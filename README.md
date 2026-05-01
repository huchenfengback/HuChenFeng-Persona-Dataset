# HuChenFeng Persona SFT Dataset

<p align="center">
  <b>面向中文人格风格复刻的 QA / SFT / Preference 数据整理项目</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/language-Chinese-red" alt="language">
  <img src="https://img.shields.io/badge/objective-Persona%20SFT-blue" alt="objective">
  <img src="https://img.shields.io/badge/data-QA%20%7C%20DPO-green" alt="data">
  <img src="https://img.shields.io/badge/model-Qwen3.5--35B-blueviolet" alt="model">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="license">
</p>

---

## 项目简介

我整理了一套用于训练 **HuChenFeng 风格中文人格模型** 的开源数据与训练流程。

它的核心目标不是训练一个"通用正确助手"，而是把公开转写语料整理成可训练的问答数据，让模型学习更直接、更口语化、更有个人表达节奏的中文回答方式。

仓库当前包含两个时间段版本：

- `early/`：早期风格版本。
- `mid/`：中期风格版本。

我后续计划继续整理 `late/` 版本，并在数据质量稳定后开放对应的 Hugging Face Dataset 与 LoRA adapter 权重。

---

## 📖 可读版本 QA（推荐普通读者从这里开始）

如果你想像看书一样浏览户晨风与网友的问答对话，可以直接查看以下可读版本：

| 阶段 | 可读版本入口 | 问答数量 | 日期范围 |
|---|---|---|---|
| **早期** | [early/readable/index.html](early/readable/index.html) | 13,549 条 | 2023-03-10 ~ 2023-12-31 |
| **中期** | [mid/readable/index.html](mid/readable/index.html) | 32,312 条 | 2024-01-01 ~ 2024-10-06 |

> 可读版本仅展示网友提问与户晨风的回答，按日期拆分为独立页面，GitHub 可直接渲染 HTML。如需训练格式数据（JSONL），请参考各阶段的 `data/` 目录。

---

## 项目简介（技术向）

---

## 在线阅读与项目入口

- 原始公开转写项目参考：[Olcmyk/HuChenFeng](https://github.com/Olcmyk/HuChenFeng)
- 仓库定位：在公开转写资料基础上，进一步构建可用于 SFT / Preference Learning 的训练数据与脚本流程。

---

## 数据来源

本仓库的数据来源于公开的 HuChenFeng 相关转写资料。我在公开转写文本基础上做了切片、QA 抽取、本人化重写、规则过滤、日期切分和训练格式整理。

原始转写材料通常不是天然适合训练的 SFT 数据：其中会混有直播上下文、弹幕现场信息、第三方总结腔、片段断裂、重复表达和人称漂移。因此我没有直接把原始文本喂给模型训练，而是先把它整理成更稳定的问答格式。

我当前使用的公开转写快照来自 [Olcmyk/HuChenFeng](https://github.com/Olcmyk/HuChenFeng) 中按日期组织的 Markdown 文件。这个快照里可用的转写日期覆盖 `2023-03-10` 到 `2025-09-14`。本仓库当前先开放其中两个阶段：

| 阶段 | 使用的公开转写日期 | 当前状态 |
|---|---|---|
| `early` | `2023-03-10` 到 `2023-12-31` | 已开放数据与训练流程 |
| `mid` | `2024-01-01` 到 `2024-10-06` | 已开放数据与训练流程 |
| `late` | 计划优先整理 `2025-01-09` 到 `2025-09-14` | 计划中 |

当前公开快照中，我本地未看到连续的 `2024-10-07` 到 `2024-12-31` 转写文件。因此 `late` 版本会先从 2025 年可用转写开始整理；如果原始仓库后续补充这段资料，我会再考虑补进 `late` 或单独做一个过渡版本。

---

## 为什么分为 early / mid

我把数据拆成不同时间段，是因为同一个人物在不同阶段的表达方式、常见话题和回答结构会有变化。如果把所有时间段直接混在一起，模型容易学到一个平均化、模糊化的人设。

- `early`：更偏早期语料中的表达节奏，回答更强调直接、强判断和口语化。
- `mid`：更偏中期语料中的本人化复刻，重写时更强调保留原始立场、事实细节和说话力度。
- `late`：计划中的后续版本，用于继续补齐更晚期语料中的风格变化。

这种拆分也方便后续训练多个 LoRA adapter，分别对比不同阶段的风格差异。

我在切分训练集、验证集和测试集时也尽量按日期切，而不是随机切。这样可以减少同一天、同一场直播中近重复内容同时出现在训练集和验证集里的问题。

| 版本 | 训练集日期 | 验证集日期 | 测试集日期 |
|---|---|---|---|
| `early` | `2023-03-10` 到 `2023-10-31` | `2023-11-01` 到 `2023-11-28` | `2023-12-03` 到 `2023-12-31` |
| `mid` | `2024-01-01` 到 `2024-08-31` | `2024-09-04` 到 `2024-10-06` | 当前版本暂未单独切出测试集 |

---

## 项目特色

- **分阶段数据集**：按时间段拆分为 `early` 与 `mid`，便于训练不同阶段的人格风格模型。
- **训练格式友好**：主数据采用 ShareGPT 风格 `messages` 格式，可直接用于多数 SFT 框架。
- **包含偏好数据**：我额外提供 `prompt / chosen / rejected` 格式数据，方便继续做 DPO 或类似偏好优化实验。
- **完整训练流程**：包含 LoRA SFT 脚本、推理对比脚本和可编辑配置模板。
- **面向研究复现**：保留数据构建、后处理、训练和评估所需的核心文件结构。

---

## 清洗与重写流程

我使用 Qwen3.5-35B-A3B 作为主要的数据清洗与重写辅助模型，通过 OpenAI-compatible API 调用它完成候选 QA 抽取和本人化改写。脚本中默认模型名写作 `qwen3.5-35b`，实际运行时可替换成任何兼容接口的中文大模型。

整体流程分为三层：

1. 候选 QA 抽取
   使用 `early/scripts/make_qa.py` 对公开转写文本进行切片，并从片段中抽取可独立成立的问答对。
2. 人格风格重写
   使用风格 prompt 将候选回答改写成更自然的本人直接回答，减少总结稿、旁白稿和第三方整理腔。
3. 规则过滤与训练格式化
   过滤直播现场残留、过短/过长回答、重复表达、摘要腔开头和低风格分样本，并输出 ShareGPT SFT 格式和 DPO 偏好格式。

### early 清洗脚本

`early/` 主要包含三步：

- `early/scripts/make_qa.py`
  从公开转写中抽取候选 QA，重点要求问题和回答脱离直播上下文后仍然成立。
- `early/scripts/build_v1_style_dataset.py`
  使用较强的人格风格 prompt，将回答改写得更直接、更口语化、更有早期表达力度。
- `early/scripts/postfilter_v1.py`
  对重写后的数据做二次过滤，去掉摘要腔、人称漂移、重复和低风格分样本。

### mid 清洗脚本

`mid/` 使用单独的本人化重写流程：

- `mid/scripts/build_mid_dataset.py`
  这个脚本更强调 faithful persona rewrite，即尽量保留原始语料中的立场、事实细节、情绪强度和回答节奏，不把回答改写成“更理性、更温和”的通用助手语气。

`mid` 的目标不是让回答更客观，而是更接近原始人物在该阶段的表达方式。

---

## 目录结构

```text
opensource_release/
├── early/
│   ├── data/
│   ├── scripts/
│   ├── training/
│   └── docs/
├── mid/
│   ├── data/
│   ├── scripts/
│   ├── training/
│   └── docs/
├── ACKNOWLEDGEMENTS.md
├── LICENSE
└── README.md
```

---

## 数据规模

| 版本 | 完整清洗集 | 训练集 | 验证集 | 测试集 | 偏好对 |
|---|---:|---:|---:|---:|---:|
| `early` | 13,549 | 6,291 | 3,968 | 3,290 | 13,486 |
| `mid` | 32,312 | 27,322 | 4,990 | 0 | 30,970 |

---

## 模型与 LoRA 计划

当前仓库先开放数据、清洗脚本和 LoRA 训练流程。训练脚本默认面向 Qwen 系列因果语言模型，可通过配置文件替换为其他兼容的 base model。

我已经基于这些数据训练过 LoRA adapter。当前 LoRA 版本可以初步学习到部分口语化表达和人格风格，但仍然存在一些客观局限：

- 对深层观点路径的复刻还不稳定，有时只学到表层语气。
- 部分回答会偏通用助手腔，需要继续提高训练数据的本人化密度。
- 对事实细节和人物经历的边界控制还需要更强的评测集。
- 不同时间段风格混合后容易出现表达漂移，因此我选择继续保留分阶段版本。

后续我计划开放 Hugging Face 上的 LoRA adapter 权重，并继续探索更强的 HuChenFeng-enhanced 版本，包括更严格的偏好数据、人工评测集和可能的 DPO / RLHF / GRPO 实验。

---

## 数据格式

### SFT 数据

`early/data/*_train.jsonl` 和 `mid/data/*_train.jsonl` 采用 ShareGPT 风格：

```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "meta": {
    "date": "...",
    "bucket": "...",
    "style_score": 0.85
  }
}
```

### Preference 数据

偏好数据采用如下格式：

```json
{
  "prompt": [{"role": "user", "content": "..."}],
  "chosen": [{"role": "assistant", "content": "..."}],
  "rejected": [{"role": "assistant", "content": "..."}],
  "meta": {"date": "...", "bucket": "..."}
}
```

---

## 快速开始

### 训练 early 版本

```bash
cd early
cp training/configs/train_v1.example.env training/configs/train_v1.env
# 编辑 train_v1.env，填写模型路径、Python 环境和 GPU 配置
bash training/scripts/run_train_v1.sh
```

### 训练 mid 版本

```bash
cd mid
cp training/configs/train_mid.example.env training/configs/train_mid.env
# 编辑 train_mid.env，填写模型路径、Python 环境和 GPU 配置
bash training/scripts/run_train_mid.sh
```

### 推理对比

```bash
# early
cd early
bash training/scripts/run_infer_compare_v1.sh

# mid
cd mid
bash training/scripts/run_infer_compare_mid.sh
```

---

## 推荐用途

- 中文人格风格 SFT 实验
- 中文口语化 QA 数据构建研究
- LoRA 微调流程复现
- 偏好优化数据构造实验
- 不同时间段人物表达风格对比研究

---

## 注意事项

- 这个数据集优化目标是表达风格，不保证所有观点事实正确。
- 数据中可能包含强烈判断、口语化表达或尖锐立场。
- 如果将模型用于产品或公开服务，请额外做事实性、安全性和风格边界评估。
- 数据体量较大时，我会将完整数据集发布到 Hugging Face Dataset；GitHub 仓库主要维护代码、说明文档和数据入口。

---

## 许可证

本仓库中的代码、脚本与整理流程采用 [MIT License](./LICENSE)。

数据使用请同时尊重原始公开资料来源、平台规则以及相关内容权利边界。

---

## 致谢

感谢公开转写与资料整理社区的工作。本仓库参考并受益于：

- [Olcmyk/HuChenFeng](https://github.com/Olcmyk/HuChenFeng)

感谢 Qwen / QwenLM 团队提供优秀的开源模型生态。这个仓库的数据清洗、重写实验和 LoRA 训练流程都受益于 Qwen 系列模型的中文能力。

也感谢中文开源社区在数据整理、模型微调和人格风格研究方向上的持续探索。
