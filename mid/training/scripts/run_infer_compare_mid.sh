#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${CONFIG:-$ROOT/training/configs/train_mid.env}"

if [[ ! -f "$CONFIG" ]]; then
  echo "[ERROR] Missing config: $CONFIG"
  echo "[INFO] Copy training/configs/train_mid.example.env to training/configs/train_mid.env and edit it first."
  exit 1
fi

source "$CONFIG"
mkdir -p "$LOG_DIR" "$ROOT/training/evals"

CUDA_VISIBLE_DEVICES="$COMPARE_GPU_IDS" "$TRAIN_PYTHON" \
  "$ROOT/training/common/infer_compare_lora.py" \
    --base_model_path "$BASE_MODEL_PATH" \
    --adapter_path "$OUTPUT_DIR" \
    --input_jsonl "$TEST_FILE" \
    --output_jsonl "$ROOT/training/evals/mid_compare_test.jsonl" \
    --max_samples 20 \
    --max_new_tokens 256 \
    --batch_size "$COMPARE_BATCH_SIZE" \
    --load_in_4bit
