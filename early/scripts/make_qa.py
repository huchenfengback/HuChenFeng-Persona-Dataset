import argparse
import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from openai import APIConnectionError, AsyncOpenAI, InternalServerError


SYSTEM_PROMPT_DIRECT = (
    "Direct JSON mode. Output ONLY raw JSON. No intro, no markdown, no reasoning text. "
    "Start with '{' and end with '}'. Use valid JSON with double quotes."
)


QA_EXTRACTION_PROMPT = """
你是一个严格的数据清洗助手。你的任务是把“户晨风直播转写片段”整理成适合后续 SFT 的中文问答对。

目标：
1. 抽取“适合独立成立”的问题。
2. 抽取“户晨风对这个问题的回答”。
3. 保留他的表达风格、口语感、论证路径。
4. 删除明显依赖直播现场的噪声。

请遵守这些规则：
1. 只保留能脱离直播现场后仍然成立的问答。
2. 问题必须是完整中文句子，不能是残缺短语。
3. 回答要以户晨风的回答为基础，但去掉明显无意义的口头噪声。
4. 不要编造原文没有的信息。
5. 遇到以下内容尽量丢弃：
   - 纯寒暄、PK、礼物感谢、连麦调侃
   - 强依赖“弹幕/热搜/刚才那个/这个直播间/对面主播”的内容
   - 明显不适合作为通用问答训练的攻击性或低信息量段落
6. 如果片段里没有足够稳定的问答，就返回空数组。
7. 输出必须是 JSON 对象，格式如下：
{{
  "qa_pairs": [
    {{
      "question": "完整问题",
      "answer": "基于原文整理后的回答",
      "style_tags": ["口语化", "直接", "现实向"],
      "should_keep": true,
      "confidence": 0.92,
      "source_basis": "简短说明这个问答来自片段中的哪部分"
    }}
  ]
}}

质量标准：
1. 回答要尽量像“可训练样本”，而不是直播流水账。
2. 回答不要过短，尽量包含观点或解释。
3. 如果只是情绪表达、没有实质内容，should_keep 设为 false。

以下是原始转写片段：
{transcript}
"""


SPEAKER_PATTERN = re.compile(r"(户晨风|某网友|网友|对方|女生|男生|路人|观众|粉丝)[：:]")

NOISE_PATTERNS = [
    r"谢谢.*礼物",
    r"点点关注",
    r"兄弟们",
    r"直播间",
    r"PK",
    r"连麦",
    r"弹幕",
]


@dataclass
class Segment:
    file_path: str
    segment_id: str
    source_text: str
    date_str: str


def process_qa_output(output_str: str) -> Any:
    if not output_str:
        return None

    start_idx = output_str.find("{")
    end_idx = output_str.rfind("}")
    if start_idx == -1 or end_idx == -1:
        return None

    json_str = output_str[start_idx:end_idx + 1]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            json_str = re.sub(r'(?<=[a-zA-Z0-9])"(?=[a-zA-Z0-9 ])', r'\"', json_str)
            if "': '" in json_str or "': [" in json_str:
                json_str = json_str.replace("'", '"')
            json_str = re.sub(r"[\x00-\x1F\x7F]", "", json_str)
            return json.loads(json_str)
        except Exception:
            return None


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def split_turns(text: str) -> list[tuple[str, str]]:
    text = normalize_text(text)
    matches = list(SPEAKER_PATTERN.finditer(text))
    if not matches:
        return []

    turns: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        speaker = match.group(1)
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        content = re.sub(r"\s+", " ", content)
        if content:
            turns.append((speaker, content))
    return turns


def build_segments(file_path: str, text: str, window_size: int = 8, min_chars: int = 120) -> list[Segment]:
    turns = split_turns(text)
    if not turns:
        return []

    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", Path(file_path).stem)
    date_str = date_match.group(1) if date_match else ""

    segments: list[Segment] = []
    for idx, (speaker, _) in enumerate(turns):
        if speaker != "户晨风":
            continue

        left = max(0, idx - 2)
        right = min(len(turns), idx + window_size)
        window = turns[left:right]
        segment_text = "\n".join(f"{spk}：{content}" for spk, content in window)
        if len(segment_text) < min_chars:
            continue

        segments.append(
            Segment(
                file_path=file_path,
                segment_id=f"{Path(file_path).stem}_seg_{idx:04d}",
                source_text=segment_text,
                date_str=date_str,
            )
        )
    return segments


def pair_is_noise(question: str, answer: str) -> bool:
    merged = f"{question} {answer}"
    if len(question.strip()) < 6 or len(answer.strip()) < 20:
        return True
    if answer.count("？") + answer.count("?") > 3:
        return True
    return any(re.search(pat, merged, flags=re.IGNORECASE) for pat in NOISE_PATTERNS)


def get_era_range(era: str) -> tuple[str | None, str | None]:
    if era == "early":
        return "2023-03-01", "2023-12-31"
    if era == "mid":
        return "2024-01-01", "2024-12-31"
    if era == "late":
        return "2025-01-01", "2025-12-31"
    return None, None


def parse_date_from_path(path: Path) -> date | None:
    match = re.search(r"(\d{4}-\d{2}-\d{2})", path.stem)
    if not match:
        return None
    try:
        return date.fromisoformat(match.group(1))
    except ValueError:
        return None


def in_date_range(path: Path, date_start: str | None, date_end: str | None) -> bool:
    path_date = parse_date_from_path(path)
    if path_date is None:
        return False
    if date_start and path_date < date.fromisoformat(date_start):
        return False
    if date_end and path_date > date.fromisoformat(date_end):
        return False
    return True


def classify_pair(question: str, answer: str, confidence: float, profile: str) -> tuple[bool, str]:
    if not question or not answer:
        return False, "empty_question_or_answer"
    if pair_is_noise(question, answer):
        return False, "noise_pattern_or_too_short"

    q_len = len(question.strip())
    a_len = len(answer.strip())

    if q_len < 8:
        return False, "question_too_short"
    if q_len > 80:
        return False, "question_too_long"
    if a_len < 60:
        return False, "answer_too_short"
    if a_len > 650:
        return False, "answer_too_long"

    if profile == "professional_v1":
        if confidence < 0.75:
            return False, "low_confidence"
        if any(token in answer for token in ["直播间", "PK", "连麦", "弹幕", "谢谢礼物"]):
            return False, "live_context_heavy"

    return True, "accepted"


def to_sharegpt_record(question: str, answer: str, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "meta": meta,
    }


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
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
            return {"content": content}
        except (APIConnectionError, InternalServerError):
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(5)
        except Exception as exc:
            error_str = str(exc).lower()
            if any(err in error_str for err in ["incomplete chunked read", "peer closed connection", "connection closed"]):
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(8)
            else:
                raise
    return {"content": ""}


async def generate_qa_for_segment(
    client: AsyncOpenAI,
    model: str,
    segment: Segment,
    raw_log_path: Path,
    profile: str,
) -> dict[str, Any]:
    prompt = QA_EXTRACTION_PROMPT.format(transcript=segment.source_text)
    result = await get_response_async(client, model, prompt)
    raw_response = result.get("content", "")

    with raw_log_path.open("a", encoding="utf-8") as f:
        f.write(f"========== {segment.segment_id} ==========\n")
        f.write(raw_response)
        f.write("\n==================================================\n\n")

    parsed = process_qa_output(raw_response)
    if not isinstance(parsed, dict):
        return {
            "accepted_pairs": [],
            "rejected_pairs": [
                {
                    "question": "",
                    "answer": "",
                    "confidence": 0.0,
                    "reject_reason": "invalid_json_or_empty_response",
                    "segment_id": segment.segment_id,
                    "file_path": segment.file_path,
                    "date": segment.date_str,
                    "source_text": segment.source_text,
                }
            ],
        }

    pairs = parsed.get("qa_pairs", [])
    if not isinstance(pairs, list):
        return {
            "accepted_pairs": [],
            "rejected_pairs": [
                {
                    "question": "",
                    "answer": "",
                    "confidence": 0.0,
                    "reject_reason": "missing_qa_pairs_field",
                    "segment_id": segment.segment_id,
                    "file_path": segment.file_path,
                    "date": segment.date_str,
                    "source_text": segment.source_text,
                }
            ],
        }

    accepted_pairs: list[dict[str, Any]] = []
    rejected_pairs: list[dict[str, Any]] = []
    for item in pairs:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        should_keep = bool(item.get("should_keep", True))
        confidence = float(item.get("confidence", 0.0) or 0.0)
        style_tags = item.get("style_tags", [])
        source_basis = item.get("source_basis", "")

        if not question or not answer:
            rejected_pairs.append(
                {
                    "question": question,
                    "answer": answer,
                    "style_tags": style_tags,
                    "confidence": confidence,
                    "source_basis": source_basis,
                    "reject_reason": "empty_question_or_answer",
                    "segment_id": segment.segment_id,
                    "file_path": segment.file_path,
                    "date": segment.date_str,
                    "source_text": segment.source_text,
                }
            )
            continue
        if not should_keep:
            rejected_pairs.append(
                {
                    "question": question,
                    "answer": answer,
                    "style_tags": style_tags,
                    "confidence": confidence,
                    "source_basis": source_basis,
                    "reject_reason": "llm_marked_should_not_keep",
                    "segment_id": segment.segment_id,
                    "file_path": segment.file_path,
                    "date": segment.date_str,
                    "source_text": segment.source_text,
                }
            )
            continue

        keep, reason = classify_pair(question, answer, confidence, profile)
        pair_record = {
            "question": question,
            "answer": answer,
            "style_tags": style_tags,
            "confidence": confidence,
            "source_basis": source_basis,
            "segment_id": segment.segment_id,
            "file_path": segment.file_path,
            "date": segment.date_str,
            "source_text": segment.source_text,
        }
        if keep:
            accepted_pairs.append(pair_record)
        else:
            rejected_pairs.append({**pair_record, "reject_reason": reason})
    return {"accepted_pairs": accepted_pairs, "rejected_pairs": rejected_pairs}


async def process_segments(
    segments: list[Segment],
    client: AsyncOpenAI,
    model: str,
    output_dir: Path,
    concurrency: int,
    profile: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_log_path = output_dir / "qa_raw_responses.txt"
    segment_dir = output_dir / "segments_json"
    segment_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"

    semaphore = asyncio.Semaphore(concurrency)
    accepted_pairs: list[dict[str, Any]] = []
    rejected_pairs: list[dict[str, Any]] = []
    empty_segments: list[dict[str, Any]] = []
    stats = {
        "total_segments": len(segments),
        "processed_segments": 0,
        "skipped_segments": 0,
        "accepted_pairs": 0,
        "rejected_pairs": 0,
        "empty_segments": 0,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    def save_progress() -> None:
        with progress_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    async def worker(segment: Segment) -> None:
        async with semaphore:
            out_path = segment_dir / f"{segment.segment_id}.json"
            if out_path.exists():
                stats["skipped_segments"] += 1
                stats["processed_segments"] += 1
                try:
                    existing = json.loads(out_path.read_text(encoding="utf-8"))
                    accepted_pairs.extend(existing.get("accepted_qa_pairs", []))
                    rejected_pairs.extend(existing.get("rejected_qa_pairs", []))
                    if not existing.get("accepted_qa_pairs", []):
                        empty_segments.append(
                            {
                                "segment_id": existing.get("segment_id", segment.segment_id),
                                "file_path": existing.get("file_path", segment.file_path),
                                "date": existing.get("date", segment.date_str),
                                "source_text": existing.get("source_text", segment.source_text),
                                "reason": "no_accepted_pairs_in_existing_record",
                            }
                        )
                except Exception:
                    pass
                if stats["processed_segments"] % 50 == 0:
                    print(
                        f"[PROGRESS] processed={stats['processed_segments']}/{stats['total_segments']} "
                        f"skipped={stats['skipped_segments']} accepted={len(accepted_pairs)} rejected={len(rejected_pairs)}",
                        flush=True,
                    )
                    save_progress()
                return

            result = await generate_qa_for_segment(client, model, segment, raw_log_path, profile)
            segment_accepted = result["accepted_pairs"]
            segment_rejected = result["rejected_pairs"]
            record = {
                "segment_id": segment.segment_id,
                "file_path": segment.file_path,
                "date": segment.date_str,
                "source_text": segment.source_text,
                "accepted_qa_pairs": segment_accepted,
                "rejected_qa_pairs": segment_rejected,
            }
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            accepted_pairs.extend(segment_accepted)
            rejected_pairs.extend(segment_rejected)
            if not segment_accepted:
                empty_segments.append(
                    {
                        "segment_id": segment.segment_id,
                        "file_path": segment.file_path,
                        "date": segment.date_str,
                        "source_text": segment.source_text,
                        "reason": "no_accepted_pairs",
                    }
                )
            stats["processed_segments"] += 1
            stats["accepted_pairs"] = len(accepted_pairs)
            stats["rejected_pairs"] = len(rejected_pairs)
            stats["empty_segments"] = len(empty_segments)
            if stats["processed_segments"] % 50 == 0:
                print(
                    f"[PROGRESS] processed={stats['processed_segments']}/{stats['total_segments']} "
                    f"skipped={stats['skipped_segments']} accepted={len(accepted_pairs)} rejected={len(rejected_pairs)}",
                    flush=True,
                )
                save_progress()

    await asyncio.gather(*(worker(seg) for seg in segments))
    stats["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    stats["accepted_pairs"] = len(accepted_pairs)
    stats["rejected_pairs"] = len(rejected_pairs)
    stats["empty_segments"] = len(empty_segments)
    save_progress()
    return accepted_pairs, rejected_pairs, empty_segments


def collect_markdown_files(input_dir: Path, date_start: str | None = None, date_end: str | None = None) -> list[Path]:
    files: list[Path] = []
    for p in input_dir.rglob("*.md"):
        if not p.is_file():
            continue
        if "_pipeline" in p.parts or ".git" in p.parts:
            continue
        if not in_date_range(p, date_start, date_end):
            continue
        files.append(p)
    return sorted(files)


def deduplicate_pairs(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for item in pairs:
        key = (item["question"], item["answer"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def save_outputs(
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    empty_segments: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    accepted = deduplicate_pairs(accepted)

    full_jsonl = output_dir / "huchenfeng_qa_full.jsonl"
    sft_jsonl = output_dir / "huchenfeng_qa_sft.jsonl"
    rejected_jsonl = output_dir / "huchenfeng_qa_rejected.jsonl"
    empty_segments_jsonl = output_dir / "huchenfeng_segments_empty.jsonl"

    with full_jsonl.open("w", encoding="utf-8") as f_full, sft_jsonl.open("w", encoding="utf-8") as f_sft:
        for item in accepted:
            f_full.write(json.dumps(item, ensure_ascii=False) + "\n")
            sharegpt = to_sharegpt_record(
                item["question"],
                item["answer"],
                {
                    "style_tags": item.get("style_tags", []),
                    "confidence": item.get("confidence", 0.0),
                    "segment_id": item.get("segment_id", ""),
                    "file_path": item.get("file_path", ""),
                },
            )
            f_sft.write(json.dumps(sharegpt, ensure_ascii=False) + "\n")

    with rejected_jsonl.open("w", encoding="utf-8") as f_rejected:
        for item in rejected:
            f_rejected.write(json.dumps(item, ensure_ascii=False) + "\n")

    with empty_segments_jsonl.open("w", encoding="utf-8") as f_empty:
        for item in empty_segments:
            f_empty.write(json.dumps(item, ensure_ascii=False) + "\n")


def compute_dataset_stats(files: list[Path], window_size: int, min_chars: int) -> dict[str, Any]:
    segments = 0
    total_chars = 0
    for file_path in files:
        text = file_path.read_text(encoding="utf-8")
        total_chars += len(text)
        segments += len(build_segments(str(file_path), text, window_size=window_size, min_chars=min_chars))
    return {
        "markdown_files": len(files),
        "total_chars": total_chars,
        "candidate_segments": segments,
        "window_size": window_size,
        "min_chars": min_chars,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="把户晨风转写仓库清洗成 QA / SFT 数据")
    parser.add_argument("--input_dir", type=str, required=True, help="HuChenFeng 仓库根目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--model", type=str, default="qwen3.5-35b")
    parser.add_argument("--base_url", type=str, default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--api_key_file", type=str, default="")
    parser.add_argument("--max_files", type=int, default=0, help="仅处理前 N 个 markdown 文件，0 表示全部")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--min_chars", type=int, default=120)
    parser.add_argument("--dry_run", action="store_true", help="只统计规模，不调用模型")
    parser.add_argument("--date_start", type=str, default=None, help="起始日期，格式 YYYY-MM-DD")
    parser.add_argument("--date_end", type=str, default=None, help="结束日期，格式 YYYY-MM-DD")
    parser.add_argument("--era", type=str, default="all", choices=["all", "early", "mid", "late"])
    parser.add_argument("--profile", type=str, default="professional_v1", choices=["base", "professional_v1"])
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    era_start, era_end = get_era_range(args.era)
    date_start = args.date_start or era_start
    date_end = args.date_end or era_end
    files = collect_markdown_files(input_dir, date_start=date_start, date_end=date_end)
    if args.max_files > 0:
        files = files[:args.max_files]

    stats = compute_dataset_stats(files, window_size=args.window_size, min_chars=args.min_chars)
    stats["era"] = args.era
    stats["profile"] = args.profile
    stats["date_start"] = date_start
    stats["date_end"] = date_end
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "dataset_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[INFO] markdown files: {stats['markdown_files']}")
    print(f"[INFO] total chars: {stats['total_chars']}")
    print(f"[INFO] candidate segments: {stats['candidate_segments']}")
    print(f"[INFO] manifest saved to: {manifest_path}")

    if args.dry_run:
        return

    segments: list[Segment] = []
    for file_path in files:
        text = file_path.read_text(encoding="utf-8")
        segments.extend(
            build_segments(
                str(file_path),
                text,
                window_size=args.window_size,
                min_chars=args.min_chars,
            )
        )

    api_key = args.api_key
    key_path = Path(args.api_key_file)
    if key_path.exists():
        api_key = key_path.read_text(encoding="utf-8").strip() or api_key

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=args.base_url,
        timeout=120.0,
    )

    print(f"[INFO] actual segments queued: {len(segments)}")

    accepted, rejected, empty_segments = await process_segments(
        segments=segments,
        client=client,
        model=args.model,
        output_dir=output_dir,
        concurrency=args.concurrency,
        profile=args.profile,
    )
    save_outputs(accepted, rejected, empty_segments, output_dir)
    print(f"[INFO] accepted qa pairs: {len(deduplicate_pairs(accepted))}")
    print(f"[INFO] rejected qa pairs: {len(rejected)}")
    print(f"[INFO] empty segments: {len(empty_segments)}")
    print(f"[INFO] saved to: {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
