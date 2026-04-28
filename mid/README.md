# HuChenFeng Persona SFT Dataset：Mid

`mid/` 是我整理的中期风格版本。这个版本更强调“本人化回答”：问题保持自然用户提问，回答尽量保留原始语料中的立场强度、表达节奏和口语判断，而不是改写成通用助手或摘要稿。

## 目录结构

```text
mid/
├── data/
│   ├── mid_clean_full.jsonl
│   ├── mid_train.jsonl
│   ├── mid_val.jsonl
│   ├── mid_test.jsonl
│   ├── mid_dpo_pairs.jsonl
│   └── mid_report.json
├── scripts/
│   └── build_mid_dataset.py
├── training/
│   ├── common/
│   │   ├── train_lora_sft.py
│   │   └── infer_compare_lora.py
│   ├── configs/
│   │   └── train_mid.example.env
│   └── scripts/
│       ├── run_train_mid.sh
│       └── run_infer_compare_mid.sh
└── docs/
    └── DATA_CARD.md
```

## 数据格式

训练、验证和测试文件采用 ShareGPT 风格的 `messages` 格式：

```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "meta": {
    "date": "2024-01-01",
    "bucket": "general_persona",
    "style_score": 0.85
  }
}
```

`mid_dpo_pairs.jsonl` 是我额外整理的偏好优化数据，可继续用于 DPO 或类似偏好学习实验。

## 数据规模

统计信息见 `data/mid_report.json`。当前版本：

- 完整清洗集：32,312 条
- 训练集：27,322 条
- 验证集：4,990 条
- 测试集：0 条
- DPO 偏好对：30,970 条

## 训练

先复制配置文件：

```bash
cp training/configs/train_mid.example.env training/configs/train_mid.env
```

请在配置文件中填写本地训练环境对应的 `BASE_MODEL_PATH`、`TRAIN_PYTHON`、`TRAIN_GPU_IDS` 等参数后运行：

```bash
bash training/scripts/run_train_mid.sh
```

## 推理对比

```bash
bash training/scripts/run_infer_compare_mid.sh
```
