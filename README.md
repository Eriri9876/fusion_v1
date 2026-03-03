# 融合多专家路由 (MoE) 的长时序预测鲁棒性实证研究
**Empirical Study on LSTF Model Fusion: MLF meets DUET**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)
![Task](https://img.shields.io/badge/Task-Long--term%20Time%20Series%20Forecasting-success)

本仓库包含了针对长序列时间序列预测（LSTF）任务的模型融合实证研究代码与完整实验报告。本项目以 MLF 为主干网络，探索了引入 DUET 混合专家机制（MoE）进行跨通道特征掩码注入的有效性与初始化敏感性。

## 📑 核心研究结论与报告

我们实施了严格的控制变量与底线测试，深入剖析了复杂时序模型中动态路由架构的跨 Seed 失效问题：
- **静态残差融合**：在固定融合强度（$\alpha=0.05$）下，静态通道掩码注入能带来微弱但稳定的指标增益。
- **动态联合训练**：解冻 MoE 门控网络后，由于时序数据的强周期性与缺乏负载均衡约束，模型表现出极端的初始化（Seed）敏感性，出现专家负载崩塌。

👉 **[点击此处阅读完整的《实证研究与失效分析报告》PDF] (此处替换为你的PDF链接，例如 ./Empirical_Study_on_MLF_DUET_Fusion.pdf)**

---

## 📂 仓库结构导航

- `models/`, `layers/`, `data_provider/`：模型核心组件与数据加载器。
- `main_MLF_longterm.py`：长序列预测任务的主训练与评估入口。
- `archive/`：历史调试补丁记录（Patch & Debug logs）。
- `scripts/`：数据预处理与修复脚本。

---

## 🚀 快速复现指南 (Quick Start)

为保证实验的可复现性，我们在 **[Releases](#)** 中提供了跨 3 个 Seed (1986, 2020, 2024) 的 Baseline 与 Fusion 预训练权重。

### 0. 环境准备 (Windows PowerShell)
```powershell
conda activate mlf
chcp 65001
$env:PYTHONIOENCODING="utf-8"
请确保 ETTh1_fixed.csv 数据集已放置于 .\dataset\ 目录下。
```

### 1. 评估 Baseline 模型
下载 Release 中的 Baseline 权重至 checkpoints/ 目录，并执行：

```PowerShell
$env:MLF_SEED="2024"
$env:MLF_PRETRAINED_CKPT="checkpoints/baseline/baseline_seed_2024/0_checkpoint.pth"

python -u .\main_MLF_longterm.py `
  --use_duet 0 --is_training 0 --itr 1 `
  --root_path .\dataset --data_path ETTh1_fixed.csv `
  --pred_len 96 --batch_size 8 --num_workers 0
```
### 2. 评估 Fusion 模型 (Stage 1)
下载 Release 中的 Fusion 权重至 checkpoints/ 目录，并执行：

```PowerShell
$env:MLF_SEED="2024"
$env:MLF_PRETRAINED_CKPT="checkpoints/fusion_stage1/ckpt_route1_duetonly_seed2024/0_checkpoint.pth"

python -u .\main_MLF_longterm.py `
  --use_duet 1 --is_training 0 --itr 1 `
  --root_path .\dataset --data_path ETTh1_fixed.csv `
  --pred_len 96 --batch_size 8 --num_workers 0
```
---
## 🙏 致谢
本研究的基础代码架构与基线模型参考了学术界现有的优秀开源成果。
