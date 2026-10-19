# UniFusion: 大模型兼容融合小模型的统一框架

> **验证实验代码框架** — 验证"隐状态级模型调用"是否比文本级Tool Use更有效

## 项目概述

UniFusion 探索一种新的大模型与小模型融合方式：**在隐状态空间（hidden state space）进行模型间的调用与融合**，而不是传统的文本序列化调用。

### 核心假设

传统Tool Use的瓶颈：大模型生成文本 → 调用小模型 → 小模型返回文本 → 大模型解析文本。这个过程信息损失大、延迟高、无法深度融合。

UniFusion的方案：大模型在中间层提取隐状态 → 通过轻量适配层投影到小模型空间 → 小模型继续前向传播 → 输出投影回大模型空间 → 门控融合后大模型继续推理。

### 八维框架

| 维度 | 说明 |
|------|------|
| 兼容性 | 异构模型的输入输出对齐 |
| 直接使用 | 动态调用专业小模型 |
| 深度融合 | 能力内化到大模型参数 |
| 经验互鉴 | 双向知识流动 |
| 可解释可控 | 能力溯源与权重调控 |
| 效率资源 | 推理延迟与显存优化 |
| 安全隔离 | 错误隔离与回滚 |
| 防遗忘 | 融合不退化通用能力 |

## 目录结构

> 注意：`main.py` 以「包」的方式导入本目录（`from UniFusionModel.config import ...`），
> 所以**仓库目录名必须保持 `UniFusionModel`**，并且运行时需要其父目录在 `PYTHONPATH` 上
> （`main.py` 会自动加，用脚本运行时脚本也会自动加）。

```
UniFusionModel/
├── config/              # 配置系统
│   ├── default_config.py
│   └── __init__.py
├── models/              # 核心模型
│   ├── adapter.py       # 双向适配层（P_in / P_out）
│   ├── gating.py        # 门控融合层
│   ├── unifusion.py     # UniFusion主模型
│   └── __init__.py
├── trainers/            # 训练
│   ├── calibrate.py     # 适配层校准训练
│   └── __init__.py
├── eval/                # 评测
│   ├── evaluate.py      # 多任务评测（HumanEval/GSM8K/MMLU）
│   └── __init__.py
├── utils/               # 工具
│   ├── hooks.py         # 隐状态Hook机制
│   ├── layer_mapping.py # 层映射策略
│   └── __init__.py
├── data/                # 数据
│   └── prepare_data.py  # 校准数据准备
├── scripts/             # 运行脚本
│   ├── run_code_experiment.sh
│   └── smoke_test.sh
├── checkpoints/         # 模型保存
├── logs/                # 日志与结果
├── main.py              # 主入口
├── requirements.txt
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 冒烟测试（验证框架可运行）

```bash
bash scripts/smoke_test.sh
```

这会用1.5B小模型跑10步校准+5样本评测，快速验证所有模块正常工作。

### 3. 完整实验（代码专业模型融合）

```bash
bash scripts/run_code_experiment.sh
```

默认配置：
- 大模型：Qwen2.5-7B-Instruct（GPU 0）
- 小模型：Qwen2.5-Coder-7B-Instruct（GPU 1）
- 校准数据：200条代码样本
- 训练步数：500步
- 评测：HumanEval + GSM8K + MMLU，对比 native vs call 模式

> **为什么大小模型必须同族？** 小模型是直接复用大模型的 `input_ids` 做前向的，
> 两者必须共用同一个 tokenizer。因此默认组合是 Qwen2.5（通用）+ Qwen2.5-Coder（代码），
> 二者 tokenizer 完全一致。换模型时请一并核对 tokenizer 是否相同。

### 4. 自定义运行

```bash
# 准备数据
python main.py prepare --domain code --num_samples 200

# 校准训练
python main.py calibrate \
    --large_model Qwen/Qwen2.5-7B-Instruct \
    --small_model Qwen/Qwen2.5-Coder-7B-Instruct \
    --large_device cuda:0 \
    --small_device cuda:1 \
    --train_data data/calibration_code.jsonl \
    --output_dir checkpoints/my_experiment \
    --max_steps 500

# 评测
python main.py evaluate \
    --large_model Qwen/Qwen2.5-7B-Instruct \
    --small_model Qwen/Qwen2.5-Coder-7B-Instruct \
    --large_device cuda:0 \
    --small_device cuda:1 \
    --adapter checkpoints/my_experiment/adapter_final.pt \
    --tasks humaneval,gsm8k,mmlu \
    --modes native,call \
    --num_samples 100
```

## 核心模块说明

### 适配层 (BidirectionalAdapter)

- `P_in`: 大模型隐状态 → 小模型输入空间（线性/MLP/残差MLP）
- `P_out`: 小模型输出 → 大模型隐空间
- 参数量：约10-50M（远小于模型本身）
- 初始化：接近恒等映射，训练稳定

### 门控融合 (GatingFusion)

三种门控策略：
- `scalar`: 全局标量，最简单
- `vector`: 逐维向量，更精细
- `input_dependent`: 根据输入动态决定（推荐）

融合公式：`h_out = g * h_enhanced + (1-g) * h_native`

### 层映射 (Layer Mapping)

- `proportional`: 按层数比例线性映射（默认）
- `semantic`: 基于表征相似度自动寻找最佳匹配层
- `manual`: 手动指定

## 校准训练目标

自监督校准，不需要任务标注：

1. **隐状态MSE对齐**：P_in(h_large) 逼近小模型同层真实隐状态
2. **端到端LM Loss**：融合后的模型在专业文本上的语言建模损失
3. **原生保持正则**：融合后输出分布不偏离大模型原生分布太远

## 评测任务

| 任务 | 指标 | 目的 |
|------|------|------|
| HumanEval | Pass@1 | 验证代码专业能力增强 |
| GSM8K | 准确率 | 验证推理能力 |
| MMLU | 准确率 | 检测通用能力是否退化 |

## 8卡H800资源分配

```
GPU 0: 大模型 (7B, ~14GB BF16) + 适配层训练
GPU 1: 小模型 (7B, ~14GB BF16)
GPU 2-3: 备用（可用于数据并行或第二个小模型）
GPU 4-7: 并行评测 / 多实验同时跑
```

单实验只需2卡，剩余6卡可并行跑其他配置（数学、医学等）。

## 验证成功的标准

实验成功的核心指标：
1. ✅ 适配层训练收敛，loss稳定下降
2. ✅ call模式在专业任务（HumanEval）上优于native模式
3. ✅ call模式在通用任务（MMLU）上不退化超过1%
4. ✅ 门控值在专业输入上偏向小模型，在通用输入上偏向大模型
5. ✅ 推理延迟增加 < 50%（相比纯大模型）

## 后续扩展

验证实验通过后，可扩展的方向：

- [ ] 融合模式（Fuse Mode）：LoRA渐进式内化
- [ ] 多专业模型同时融合（能力正交化）
- [ ] 推理时双向知识流动
- [ ] 异构架构兼容（CNN/RNN小模型）
- [ ] KV Cache优化（加速生成）
- [ ] 自动模式切换（Call ↔ Fuse）

## 常见问题

**Q: 为什么不直接用Model Merging？**
A: Model Merging要求同架构，且是静态合并。UniFusion支持异构模型，且是动态调用。

**Q: 为什么不直接用Tool Use？**
A: Tool Use是文本级调用，信息损失大。UniFusion在隐状态空间调用，更高效、融合更深。

**Q: 适配层训练需要多少数据？**
A: 50-200条专业领域文本即可，属于少样本校准。

**Q: 支持哪些模型？**
A: 当前实现支持Llama/Qwen/DeepSeek等Decoder-Only Transformer。其他架构需要适配层的小修改。

## 引用

如果使用本代码，请引用相关工作：
- Task Arithmetic (Ilharco et al., 2022)
- BTS: Branch-Train-Stitch (Zhang et al., 2025)
- CTCD: Co-Training and Co-Distillation (Lee et al., 2023)
- SemAlign: Latent Semantic Alignment (Gu et al., 2026)
