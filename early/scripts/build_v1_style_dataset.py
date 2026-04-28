import argparse
import asyncio
import hashlib
import json
import os
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APIStatusError, AsyncOpenAI, InternalServerError


SYSTEM_PROMPT_DIRECT = (
    "Direct JSON mode. Output ONLY raw JSON. No intro, no markdown, no reasoning text. "
    "Start with '{' and end with '}'. Use valid JSON with double quotes."
)

STYLE_REWRITE_PROMPT = """
你是一个专门制作“强人格口语 SFT / DPO 数据”的中文数据工程助手。

任务：
我会给你一条问答样本，它来自户晨风相关语料，但当前版本往往是“总结稿、整理稿、旁白稿”，
不是“户晨风本人会怎么开口回答”。

你的任务不是做摘要，也不是做客观分析，而是：
把它改写成“户晨风本人在直播里或面对面时，会怎么直接回答用户”的版本。

核心目标：
1. 回答要像本人直接说话，不像编辑部整理稿。
2. 可以直接、现实、带判断、带口语，但不要乱编新事实。
3. 要保留原样本的核心立场、事实、逻辑顺序。
4. 如果原样本本身太虚、太空、太像二手转述，宁可丢弃，不要硬写。

强制风格要求：
1. answer 必须直接回答问题，不准绕成“用户认为/用户提到/这涉及到/关键在于/需要分析”。
2. answer 开头禁止出现：
   - 用户
   - 用户认为
   - 用户提到
   - 这涉及到
   - 关键在于
   - 需要分析
   - 不能一概而论
   - 这个问题
3. 尽量用更像说话的句子：
   - 可以有“我跟你说”“说白了”“本质上”“你别扯”“很简单”“就这么回事”
   - 但不要每句都装腔，不要堆口头禅
4. 不要写成百科、评论员、咨询顾问、论文摘要。
5. 不要输出 `<think>`、`</think>`、英文思维链、分析步骤。
6. question 保持自然用户提问口吻，优先第二人称或自然问法。

丢弃标准：
1. 原答案事实基础太弱，无法可靠重写
2. 直播现场残留太强，脱离上下文后不成立
3. 原答案太像总结标题、不是一个真实回答
4. 你无法在不编造的前提下改成“本人说话”

输出 JSON：
{{
  "should_keep": true,
  "drop_reason": "",
  "rewritten_question": "重写后的自然问题",
  "rewritten_answer": "重写后的本人化回答",
  "style_score": 0.86,
  "rewrite_tags": ["direct_answer", "persona_strong", "colloquial", "anti_summary"]
}}

样本：
question: {question}
answer: {answer}
bucket: {bucket}
date: {date}
"""


LIVE_TERMS = ["直播间", "PK", "连麦", "弹幕", "超管", "刷礼物", "上麦", "挂麦"]
BANNED_PREFIXES = [
    "用户",
    "用户认为",
    "用户提到",
    "用户的问题",
    "这个问题",
    "这涉及到",
    "关键在于",
    "需要分析",
    "不能一概而论",
]
BANNED_PHRASES = [
    "用户认为",
    "用户提到",
    "这涉及到",
    "关键在于",
    "需要分析",
    "不能一概而论",
    "</think>",
    "<think>",
    "here's a thinking process",
]
WEAK_STYLE_MARKERS = [
    "这反映了",
    "这种观点",
    "总体来看",
    "综合来看",
    "本质上是",
    "需要结合",
    "值得注意的是",
]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_api_key(args_api_key: str, api_key_file: str) -> str:
    if args_api_key:
        return args_api_key
    key_path = Path(api_key_file)
    if key_path.exists():
        return key_path.read_text(encoding="utf-8").strip()
    return ""


def process_json_output(output_str: str) -> dict[str, Any] | None:
    if not output_str:
        return None
    start = output_str.find("{")
    end = output_str.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(output_str[start:end + 1])
    except json.JSONDecodeError:
        return None


def normalize_question(question: str) -> str:
    q = question.strip()
    q = re.sub(r"\s+", "", q)
    replacements = [
        (r"^户晨风如何评价", "你如何评价"),
        (r"^户晨风怎么看待", "你怎么看待"),
        (r"^户晨风怎么看", "你怎么看"),
        (r"^户晨风为什么", "你为什么"),
        (r"^户晨风有没有", "你有没有"),
        (r"^户晨风会不会", "你会不会"),
        (r"^户晨风能不能", "你能不能"),
        (r"^户晨风是否", "你是否"),
        (r"^户晨风是什么", "你是什么"),
        (r"^户晨风在", "你在"),
        (r"^户晨风", "你"),
    ]
    for pattern, replacement in replacements:
        q = re.sub(pattern, replacement, q)
    if not q.endswith(("？", "?")):
        q += "？"
    return q


def normalize_compare_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，。！？、,.!?：:；;“”\"'‘’（）()【】\\-]", "", text)
    return text


def make_row_uid(row: dict[str, Any]) -> str:
    base = "||".join(
        [
            str(row.get("date", "")),
            str(row.get("segment_id", "")),
            str(row.get("question", "")),
            str(row.get("answer", "")),
        ]
    )
    return hashlib.md5(base.encode("utf-8")).hexdigest()


def has_heavy_repetition(text: str) -> bool:
    if not text:
        return False
    compact = re.sub(r"\s+", "", text)
    for span in [8, 12, 16]:
        counts: dict[str, int] = {}
        for idx in range(0, max(len(compact) - span + 1, 0)):
            chunk = compact[idx: idx + span]
            counts[chunk] = counts.get(chunk, 0) + 1
            if counts[chunk] >= 4:
                return True
    return False


def directness_score(answer: str) -> float:
    score = 1.0
    head = answer[:20]
    if any(head.startswith(prefix) for prefix in BANNED_PREFIXES):
        score -= 0.7
    for phrase in BANNED_PHRASES:
        if phrase in answer:
            score -= 0.35
    for marker in WEAK_STYLE_MARKERS:
        if marker in answer:
            score -= 0.12
    if "我" not in answer and "你" not in answer:
        score -= 0.15
    if has_heavy_repetition(answer):
        score -= 0.5
    return max(0.0, min(score, 1.0))


def rule_filter(row: dict[str, Any]) -> tuple[bool, str]:
    question = str(row.get("question", "")).strip()
    answer = str(row.get("answer", "")).strip()

    if not question or not answer:
        return False, "empty_question_or_answer"
    if len(question) < 5 or len(question) > 60:
        return False, "question_length_out_of_range"
    if len(answer) < 45:
        return False, "answer_too_short"
    if len(answer) > 420:
        return False, "answer_too_long"
    if any(term in question for term in LIVE_TERMS) or any(term in answer for term in LIVE_TERMS):
        return False, "live_context_residue"
    if answer.count("。") + answer.count("！") + answer.count("？") == 0 and len(answer) > 180:
        return False, "answer_poorly_structured"
    return True, "accepted"


def passes_style_gate(question: str, answer: str, style_score: float) -> tuple[bool, str]:
    if not question or not answer:
        return False, "empty_after_rewrite"
    if len(answer) < 35:
        return False, "rewritten_answer_too_short"
    if len(answer) > 420:
        return False, "rewritten_answer_too_long"
    if any(answer.startswith(prefix) for prefix in BANNED_PREFIXES):
        return False, "summary_prefix_residue"
    if any(phrase in answer for phrase in BANNED_PHRASES):
        return False, "summary_phrase_residue"
    if has_heavy_repetition(answer):
        return False, "rewritten_answer_repetitive"
    if style_score < 0.55:
        return False, "style_score_too_low"
    return True, "accepted"


async def get_response_async(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    max_retries: int = 3,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_DIRECT},
        {"role": "user", "content": prompt},
    ]
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=2048,
                response_format={"type": "json_object"},
            )
            return {"content": response.choices[0].message.content or ""}
        except (APIConnectionError, InternalServerError):
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(5)
        except APIStatusError as exc:
            if exc.status_code in {500, 502, 503, 504} and attempt < max_retries - 1:
                await asyncio.sleep(6)
                continue
            raise
        except Exception as exc:
            error_str = str(exc).lower()
            if any(err in error_str for err in ["connection closed", "peer closed connection", "incomplete chunked read", "502", "bad gateway"]):
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(8)
            else:
                raise
    return {"content": ""}


async def rewrite_row(
    client: AsyncOpenAI,
    model: str,
    row: dict[str, Any],
    raw_log_path: Path,
) -> dict[str, Any]:
    prompt = STYLE_REWRITE_PROMPT.format(
        question=row["question"],
        answer=row["answer"],
        bucket=row.get("bucket", row.get("meta", {}).get("bucket", "")),
        date=row.get("date", row.get("meta", {}).get("date", "")),
    )
    result = await get_response_async(client, model, prompt)
    raw = result.get("content", "")

    with raw_log_path.open("a", encoding="utf-8") as f:
        f.write(f"========== {row.get('row_uid', '')} | {row.get('segment_id', '')} ==========\n")
        f.write(raw)
        f.write("\n==================================================\n\n")

    parsed = process_json_output(raw)
    if not parsed:
        return {**row, "v1_drop_reason": "rewrite_invalid_json", "rewrite_status": "failed_invalid_json"}

    should_keep = bool(parsed.get("should_keep", True))
    if not should_keep:
        return {
            **row,
            "v1_drop_reason": str(parsed.get("drop_reason", "rewrite_llm_drop")).strip() or "rewrite_llm_drop",
            "rewrite_status": "dropped_by_llm",
        }

    rewritten_question = normalize_question(str(parsed.get("rewritten_question", row["question"])).strip() or row["question"])
    rewritten_answer = str(parsed.get("rewritten_answer", row["answer"])).strip() or row["answer"]
    style_score = parsed.get("style_score")
    try:
        style_score = float(style_score)
    except Exception:
        style_score = directness_score(rewritten_answer)

    keep, reason = passes_style_gate(rewritten_question, rewritten_answer, style_score)
    if not keep:
        return {
            **row,
            "question_v1": rewritten_question,
            "answer_v1": rewritten_answer,
            "style_score": style_score,
            "rewrite_tags": parsed.get("rewrite_tags", []),
            "rewrite_status": "rewritten_but_dropped",
            "v1_drop_reason": reason,
        }

    return {
        **row,
        "question_v1": rewritten_question,
        "answer_v1": rewritten_answer,
        "style_score": style_score,
        "rewrite_tags": parsed.get("rewrite_tags", []),
        "rewrite_status": "rewritten",
    }


async def process_rewrites(
    candidates: list[dict[str, Any]],
    client: AsyncOpenAI,
    model: str,
    output_dir: Path,
    concurrency: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_log_path = output_dir / "v1_style_raw.txt"
    rewritten_progress_path = output_dir / "v1_style_rewritten_progress.jsonl"
    dropped_progress_path = output_dir / "v1_style_rewrite_dropped_progress.jsonl"
    rewritten_rows: list[dict[str, Any]] = []
    dropped_rows: list[dict[str, Any]] = []
    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    progress_lock = asyncio.Lock()
    progress = {"done": 0, "total": len(candidates)}

    async def worker(row: dict[str, Any]) -> None:
        async with semaphore:
            try:
                result = await rewrite_row(client, model, row, raw_log_path)
                async with write_lock:
                    if "v1_drop_reason" in result:
                        dropped_rows.append(result)
                        append_jsonl(dropped_progress_path, result)
                    else:
                        rewritten_rows.append(result)
                        append_jsonl(rewritten_progress_path, result)
            except Exception as exc:
                failure = {
                    **row,
                    "rewrite_status": "rewrite_request_failed",
                    "v1_drop_reason": f"request_failed:{type(exc).__name__}",
                }
                async with write_lock:
                    dropped_rows.append(failure)
                    append_jsonl(dropped_progress_path, failure)
            finally:
                async with progress_lock:
                    progress["done"] += 1
                    done = progress["done"]
                    total = progress["total"]
                    if done % 50 == 0 or done == total:
                        print(f"[PROGRESS] rewrite_done={done}/{total}", flush=True)

    await asyncio.gather(*(worker(row) for row in candidates))
    return rewritten_rows, dropped_rows


def recover_from_raw_log(
    raw_log_path: Path,
    candidate_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    if not raw_log_path.exists():
        return [], [], set()

    text = raw_log_path.read_text(encoding="utf-8")
    blocks = [block.strip() for block in text.split("==================================================") if block.strip()]
    if not blocks:
        return [], [], set()

    indexed: dict[str, list[dict[str, Any]]] = {}
    for row in candidate_rows:
        indexed.setdefault(str(row.get("segment_id", "")), []).append(row)

    used_uids: set[str] = set()
    recovered_rewritten: list[dict[str, Any]] = []
    recovered_dropped: list[dict[str, Any]] = []

    for block in blocks:
        lines = [line for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        header = lines[0].strip()
        header_match = re.match(r"^==========\s+(.+?)\s+\|\s+(.*?)\s+==========$", header)
        raw_uid = ""
        segment_id = ""
        if header_match:
            raw_uid = header_match.group(1).strip()
            segment_id = header_match.group(2).strip()
        else:
            legacy_match = re.match(r"^==========\s+(.*?)\s+==========$", header)
            if legacy_match:
                segment_id = legacy_match.group(1).strip()

        obj = process_json_output(block)
        if not obj:
            continue

        row = None
        if raw_uid:
            for candidate in indexed.get(segment_id, []):
                if candidate.get("row_uid") == raw_uid and candidate["row_uid"] not in used_uids:
                    row = candidate
                    break

        if row is None:
            candidates = [c for c in indexed.get(segment_id, []) if c["row_uid"] not in used_uids]
            if not candidates:
                continue
            rewritten_q = normalize_compare_text(str(obj.get("rewritten_question", "")))
            best_score = -1.0
            for candidate in candidates:
                score = SequenceMatcher(
                    None,
                    normalize_compare_text(candidate["question"]),
                    rewritten_q,
                ).ratio()
                if score > best_score:
                    best_score = score
                    row = candidate

        if row is None:
            continue

        used_uids.add(row["row_uid"])
        should_keep = bool(obj.get("should_keep", True))
        if not should_keep:
            recovered_dropped.append(
                {
                    **row,
                    "rewrite_status": "dropped_by_llm",
                    "v1_drop_reason": str(obj.get("drop_reason", "rewrite_llm_drop")).strip() or "rewrite_llm_drop",
                }
            )
            continue

        rewritten_question = normalize_question(str(obj.get("rewritten_question", row["question"])).strip() or row["question"])
        rewritten_answer = str(obj.get("rewritten_answer", row["answer"])).strip() or row["answer"]
        style_score = obj.get("style_score")
        try:
            style_score = float(style_score)
        except Exception:
            style_score = directness_score(rewritten_answer)

        keep, reason = passes_style_gate(rewritten_question, rewritten_answer, style_score)
        if not keep:
            recovered_dropped.append(
                {
                    **row,
                    "question_v1": rewritten_question,
                    "answer_v1": rewritten_answer,
                    "style_score": style_score,
                    "rewrite_tags": obj.get("rewrite_tags", []),
                    "rewrite_status": "rewritten_but_dropped",
                    "v1_drop_reason": reason,
                }
            )
            continue

        recovered_rewritten.append(
            {
                **row,
                "question_v1": rewritten_question,
                "answer_v1": rewritten_answer,
                "style_score": style_score,
                "rewrite_tags": obj.get("rewrite_tags", []),
                "rewrite_status": "rewritten",
            }
        )

    return recovered_rewritten, recovered_dropped, used_uids


def split_by_date(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    rows = sorted(rows, key=lambda x: x.get("date", "9999-99-99"))
    train, val, test = [], [], []
    for row in rows:
        d = row.get("date", "")
        if d <= "2023-10-31":
            train.append(row)
        elif d <= "2023-11-30":
            val.append(row)
        else:
            test.append(row)
    return train, val, test


def to_sharegpt(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": row["question_v1"]},
            {"role": "assistant", "content": row["answer_v1"]},
        ],
        "meta": {
            "date": row.get("date", ""),
            "bucket": row.get("bucket", row.get("meta", {}).get("bucket", "")),
            "segment_id": row.get("segment_id", ""),
            "rewrite_status": row.get("rewrite_status", "rewritten"),
            "rewrite_tags": row.get("rewrite_tags", []),
            "style_score": row.get("style_score", None),
        },
    }


def to_preference_pair(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt": [
            {"role": "user", "content": row["question_v1"]},
        ],
        "chosen": [
            {"role": "assistant", "content": row["answer_v1"]},
        ],
        "rejected": [
            {"role": "assistant", "content": row["answer"]},
        ],
        "meta": {
            "date": row.get("date", ""),
            "bucket": row.get("bucket", row.get("meta", {}).get("bucket", "")),
            "segment_id": row.get("segment_id", ""),
            "rewrite_tags": row.get("rewrite_tags", []),
            "style_score": row.get("style_score", None),
        },
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Build v1_style dataset with aggressive persona rewrite")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="qwen3.5-35b")
    parser.add_argument("--base_url", type=str, default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--api_key_file", type=str, default="")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(input_path)
    normalized_rows: list[dict[str, Any]] = []
    rule_dropped: list[dict[str, Any]] = []

    for row in rows:
        meta = row.get("meta", {})
        if "messages" in row:
            question = row["messages"][0]["content"]
            answer = row["messages"][1]["content"] if len(row["messages"]) > 1 else ""
            current = {
                "question": normalize_question(question),
                "answer": answer.strip(),
                "date": meta.get("date", ""),
                "bucket": meta.get("bucket", ""),
                "segment_id": meta.get("segment_id", row.get("segment_id", "")),
                "meta": meta,
            }
        else:
            current = {
                **row,
                "question": normalize_question(str(row.get("question", "")).strip()),
                "answer": str(row.get("answer", "")).strip(),
            }

        keep, reason = rule_filter(current)
        if keep:
            current["row_uid"] = make_row_uid(current)
            normalized_rows.append(current)
        else:
            rule_dropped.append({**current, "v1_drop_reason": reason})

    if args.max_rows > 0:
        normalized_rows = normalized_rows[:args.max_rows]

    write_jsonl(output_dir / "v1_style_candidates.jsonl", normalized_rows)
    write_jsonl(output_dir / "v1_style_rule_dropped.jsonl", rule_dropped)

    recovered_rewritten, recovered_dropped, completed_uids = recover_from_raw_log(
        output_dir / "v1_style_raw.txt",
        normalized_rows,
    )
    remaining_rows = [row for row in normalized_rows if row["row_uid"] not in completed_uids]

    rewritten_rows: list[dict[str, Any]] = list(recovered_rewritten)
    rewrite_dropped: list[dict[str, Any]] = list(recovered_dropped)
    if not args.dry_run and normalized_rows:
        api_key = read_api_key(args.api_key, args.api_key_file)
        client = AsyncOpenAI(api_key=api_key, base_url=args.base_url, timeout=180.0)
        new_rewritten, new_dropped = await process_rewrites(
            remaining_rows,
            client,
            args.model,
            output_dir,
            args.concurrency,
        )
        rewritten_rows.extend(new_rewritten)
        rewrite_dropped.extend(new_dropped)

    rewritten_rows = sorted(rewritten_rows, key=lambda x: (x.get("date", ""), x.get("segment_id", "")))
    split_train, split_val, split_test = split_by_date(rewritten_rows)
    preference_rows = [to_preference_pair(row) for row in rewritten_rows if row["answer_v1"].strip() != row["answer"].strip()]

    write_jsonl(output_dir / "v1_style_rewritten.jsonl", rewritten_rows)
    write_jsonl(output_dir / "v1_style_rewrite_dropped.jsonl", rewrite_dropped)
    write_jsonl(output_dir / "v1_style_clean_full.jsonl", rewritten_rows)
    write_jsonl(output_dir / "v1_style_train.jsonl", [to_sharegpt(r) for r in split_train])
    write_jsonl(output_dir / "v1_style_val.jsonl", [to_sharegpt(r) for r in split_val])
    write_jsonl(output_dir / "v1_style_test.jsonl", [to_sharegpt(r) for r in split_test])
    write_jsonl(output_dir / "v1_style_dpo_pairs.jsonl", preference_rows)

    report = {
        "input_rows": len(rows),
        "rule_kept_rows": len(normalized_rows),
        "rule_dropped_rows": len(rule_dropped),
        "recovered_rows_from_raw": len(recovered_rewritten),
        "remaining_rows_after_resume": len(remaining_rows),
        "rewritten_rows": len(rewritten_rows),
        "rewrite_dropped_rows": len(rewrite_dropped),
        "final_rows": len(rewritten_rows),
        "dpo_pairs_rows": len(preference_rows),
        "train_rows": len(split_train),
        "val_rows": len(split_val),
        "test_rows": len(split_test),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_url": args.base_url,
        "model": args.model,
        "aggressive_persona_rewrite": True,
    }
    with (output_dir / "v1_style_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[INFO] input rows: {len(rows)}")
    print(f"[INFO] rule kept: {len(normalized_rows)}")
    print(f"[INFO] rule dropped: {len(rule_dropped)}")
    print(f"[INFO] recovered from raw: {len(recovered_rewritten)}")
    print(f"[INFO] remaining after resume: {len(remaining_rows)}")
    print(f"[INFO] rewritten rows: {len(rewritten_rows)}")
    print(f"[INFO] rewrite dropped: {len(rewrite_dropped)}")
    print(f"[INFO] dpo pairs: {len(preference_rows)}")
    print(f"[INFO] train/val/test: {len(split_train)}/{len(split_val)}/{len(split_test)}")
    print(f"[INFO] report saved to: {output_dir / 'v1_style_report.json'}")


if __name__ == "__main__":
    asyncio.run(main())
