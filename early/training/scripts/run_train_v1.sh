#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${CONFIG:-$ROOT/training/configs/train_v1.env}"

if [[ ! -f "$CONFIG" ]]; then
  echo "[ERROR] Missing config: $CONFIG"
  echo "[INFO] Copy training/configs/train_v1.example.env to training/configs/train_v1.env and edit it first."
  exit 1
fi

source "$CONFIG"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

IFS=',' read -r -a GPU_ARRAY <<< "$TRAIN_GPU_IDS"
NUM_GPUS="${#GPU_ARRAY[@]}"

CUDA_VISIBLE_DEVICES="$TRAIN_GPU_IDS" "$TRAIN_PYTHON" -m accelerate.commands.launch \
  --num_processes "$NUM_GPUS" \
  "$ROOT/training/common/train_lora_sft.py" \
    --model_name_or_path "$BASE_MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --val_file "$VAL_FILE" \
    --test_file "$TEST_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --max_seq_length "$MODEL_MAX_LENGTH" \
    --num_train_epochs "$NUM_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --per_device_train_batch_size "$TRAIN_BATCH_SIZE" \
    --per_device_eval_batch_size "$EVAL_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
    --warmup_ratio "$WARMUP_RATIO" \
    --logging_steps "$LOGGING_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --eval_steps "$EVAL_STEPS" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --load_in_4bit

