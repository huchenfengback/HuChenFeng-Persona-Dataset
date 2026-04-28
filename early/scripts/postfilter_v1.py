import argparse
import json
import re
import time
from pathlib import Path


OVERUSED_OPENERS = [
    "说白了",
    "我跟你说",
    "说实话",
    "我说句实话",
]

OVERUSED_PHRASES = [
    "说白了",
    "我跟你说",
    "说实话",
    "我说句实话",
    "就这么回事",
]

AGGRESSIVE_MARKERS = [
    "别扯",
    "扯淡",
    "做梦",
    "纯粹是",
    "根本不值",
    "混出个啥",
    "别整那些虚的",
    "那是做梦",
    "那是扯淡",
]

SUMMARY_RESIDUES = [
    "用户认为",
    "用户提到",
    "这涉及到",
    "关键在于",
    "需要分析",
    "不能一概而论",
    "<think>",
    "</think>",
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


def count_matches(text: str, phrases: list[str]) -> int:
    return sum(text.count(phrase) for phrase in phrases)


def too_repetitive(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    for span in [8, 12]:
        seen: dict[str, int] = {}
        for i in range(0, max(len(compact) - span + 1, 0)):
            chunk = compact[i:i + span]
            seen[chunk] = seen.get(chunk, 0) + 1
            if seen[chunk] >= 4:
                return True
    return False


def strict_filter(row: dict) -> tuple[bool, str]:
    answer = row["answer_v1"].strip()
    question = row["question_v1"].strip()
    style_score = float(row.get("style_score", 0.0) or 0.0)

    if not question or not answer:
        return False, "empty_after_v1"
    if len(answer) < 45:
        return False, "answer_too_short_strict"
    if len(answer) > 360:
        return False, "answer_too_long_strict"
    if style_score < 0.82:
        return False, "style_score_below_strict_threshold"
    if any(token in answer for token in SUMMARY_RESIDUES):
        return False, "summary_residue"
    if too_repetitive(answer):
        return False, "answer_repetitive"

    opener_hits = sum(answer.startswith(prefix) for prefix in OVERUSED_OPENERS)
    phrase_hits = count_matches(answer, OVERUSED_PHRASES)
    aggressive_hits = count_matches(answer, AGGRESSIVE_MARKERS)

    if opener_hits >= 1 and phrase_hits >= 2:
        return False, "overused_persona_template"
    if phrase_hits >= 3:
        return False, "overused_persona_phrases"
    if aggressive_hits >= 2:
        return False, "too_aggressive"

    # Filter out answers that still sound too generic and detached.
    generic_markers = ["要具体看", "需要结合实际", "不能一概而论", "这个得看情况", "看具体情况"]
    if any(marker in answer for marker in generic_markers):
        return False, "too_generic_after_v1"

    return True, "accepted"


def split_by_date(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    rows = sorted(rows, key=lambda x: x.get("date", x.get("meta", {}).get("date", "9999-99-99")))
    train, val, test = [], [], []
    for row in rows:
        d = row.get("date", row.get("meta", {}).get("date", ""))
        if d <= "2023-10-31":
            train.append(row)
        elif d <= "2023-11-30":
            val.append(row)
        else:
            test.append(row)
    return train, val, test


def to_sharegpt(row: dict) -> dict:
    return {
        "messages": [
            {"role": "user", "content": row["question_v1"]},
            {"role": "assistant", "content": row["answer_v1"]},
        ],
        "meta": {
            "date": row.get("date", ""),
            "bucket": row.get("bucket", row.get("meta", {}).get("bucket", "")),
            "segment_id": row.get("segment_id", ""),
            "rewrite_status": row.get("rewrite_status", ""),
            "rewrite_tags": row.get("rewrite_tags", []),
            "style_score": row.get("style_score", None),
            "strict_kept": True,
        },
    }


def to_preference_pair(row: dict) -> dict:
    return {
        "prompt": [{"role": "user", "content": row["question_v1"]}],
        "chosen": [{"role": "assistant", "content": row["answer_v1"]}],
        "rejected": [{"role": "assistant", "content": row["answer"]}],
        "meta": {
            "date": row.get("date", ""),
            "bucket": row.get("bucket", row.get("meta", {}).get("bucket", "")),
            "segment_id": row.get("segment_id", ""),
            "style_score": row.get("style_score", None),
            "strict_kept": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-filter v1_style into a stricter training set")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(input_path)
    kept: list[dict] = []
    dropped: list[dict] = []

    for row in rows:
        keep, reason = strict_filter(row)
        if keep:
            kept.append(row)
        else:
            dropped.append({**row, "strict_drop_reason": reason})

    train, val, test = split_by_date(kept)
    dpo_pairs = [to_preference_pair(row) for row in kept if row["answer_v1"].strip() != row["answer"].strip()]

    write_jsonl(output_dir / "v1_clean_full.jsonl", kept)
    write_jsonl(output_dir / "v1_dropped.jsonl", dropped)
    write_jsonl(output_dir / "v1_train.jsonl", [to_sharegpt(r) for r in train])
    write_jsonl(output_dir / "v1_val.jsonl", [to_sharegpt(r) for r in val])
    write_jsonl(output_dir / "v1_test.jsonl", [to_sharegpt(r) for r in test])
    write_jsonl(output_dir / "v1_dpo_pairs.jsonl", dpo_pairs)

    report = {
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "dropped_rows": len(dropped),
        "dpo_pairs_rows": len(dpo_pairs),
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "style_score_threshold": 0.82,
    }
    with (output_dir / "v1_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[INFO] input rows: {len(rows)}")
    print(f"[INFO] kept rows: {len(kept)}")
    print(f"[INFO] dropped rows: {len(dropped)}")
    print(f"[INFO] dpo pairs: {len(dpo_pairs)}")
    print(f"[INFO] train/val/test: {len(train)}/{len(val)}/{len(test)}")
    print(f"[INFO] report saved to: {output_dir / 'v1_report.json'}")


if __name__ == "__main__":
    main()
