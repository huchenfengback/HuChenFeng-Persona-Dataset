# HuChenFeng Persona SFT Dataset：Early

`early/` 是我整理的早期风格版本，包含 `HuChenFeng` 人格风格 SFT 数据和对应训练流程。

这个版本的目标是训练模型学习一种更直接、更口语化、更有个人表达风格的中文回答方式。它不是通用助手数据集，也不是事实校正数据集。

## 包含内容

```text
early/
├── data/
│   ├── v1_clean_full.jsonl
│   ├── v1_train.jsonl
│   ├── v1_val.jsonl
│   ├── v1_test.jsonl
│   ├── v1_dpo_pairs.jsonl
│   └── v1_report.json
├── scripts/
│   ├── make_qa.py
│   ├── build_v1_style_dataset.py
│   └── postfilter_v1.py
├── training/
│   ├── common/
│   │   ├── train_lora_sft.py
│   │   └── infer_compare_lora.py
│   ├── configs/
│   │   └── train_v1.example.env
│   └── scripts/
│       ├── run_train_v1.sh
│       └── run_infer_compare_v1.sh
└── docs/
    └── DATA_CARD.md
```

## 数据格式

`data/v1_train.jsonl`、`data/v1_val.jsonl`、`data/v1_test.jsonl` 使用 ShareGPT 风格的 `messages` 格式：

```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "meta": {
    "date": "2023-03-10",
    "bucket": "general_persona",
    "style_score": 0.89
  }
}
```

`data/v1_clean_full.jsonl` 是清洗后的完整数据，保留可用于训练和分析的问答内容与基础元信息。

`data/v1_dpo_pairs.jsonl` 是我额外整理的偏好优化数据，可继续用于 DPO 或类似实验。

## 数据规模

统计信息见 `data/v1_report.json`。

当前版本：

- 完整清洗集：13,549 条
- 训练集：6,291 条
- 验证集：3,968 条
- 测试集：3,290 条
- DPO 偏好对：13,486 条

## 流程说明

整体流程如下：

1. `scripts/make_qa.py`
   - 从原始转写中切片并抽取候选 QA。
2. `scripts/build_v1_style_dataset.py`
   - 将候选 QA 改写为更直接、更口语化的人格风格数据。
3. `scripts/postfilter_v1.py`
   - 使用规则和风格分数过滤，得到最终 v1 数据。
4. `training/scripts/run_train_v1.sh`
   - 使用 v1 数据进行 LoRA SFT 训练。
5. `training/scripts/run_infer_compare_v1.sh`
   - 对比 base model 和 LoRA adapter 的输出效果。

训练前需要复制并修改配置文件：

```bash
cp training/configs/train_v1.example.env training/configs/train_v1.env
```

请在配置文件中填写本地训练环境对应的参数：

- `BASE_MODEL_PATH`
- `TRAIN_PYTHON`
- `TRAIN_GPU_IDS`
- `COMPARE_GPU_IDS`
- 数据路径与输出路径

## 训练

```bash
bash training/scripts/run_train_v1.sh
```

## 推理对比

```bash
bash training/scripts/run_infer_compare_v1.sh
```

数据体量较大时，我会将完整数据集发布到 Hugging Face Dataset；GitHub 仓库主要维护代码、说明文档和数据入口。
