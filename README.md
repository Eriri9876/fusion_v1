# README (Deliver Package)

## 0. 环境准备（Windows PowerShell）

从Release里面下载checkpoint

进入项目根目录（你截图是 `...\deliver\`）后执行：

```powershell
conda activate mlf
chcp 65001
$env:PYTHONIOENCODING="utf-8"
```

确认数据在这里（相对 `deliver` 目录）：

```text
.\dataset\ETTh1_fixed.csv
```

---

## 1. 只 测试 Baseline

### 1.1 设置 seed 与 ckpt 路径（改成实际路径）


```powershell
$env:MLF_SEED="2024"
$env:MLF_PRETRAINED_CKPT="checkpoints/baseline_seed_2024_0_checkpoint.pth"
```

### 1.2 运行 eval

```powershell
python -u .\main_MLF_longterm.py `
  --use_duet 0 --is_training 0 --itr 1 `
  --root_path .\dataset --data_path ETTh1_fixed.csv `
  --pred_len 96 --batch_size 8 --num_workers 0 `
  --checkpoints .\runs\eval_baseline_seed2024
```

输出里会出现：

```text
final test {...}
```

---

## 2. 只 测试 Fusion Stage1

### 2.1 每次更换变量时清理一下前面的可能干扰的变量
```powershell
Remove-Item Env:MLF_PRETRAINED_CKPT -ErrorAction SilentlyContinue
Remove-Item Env:PRELOAD_CKPT -ErrorAction SilentlyContinue
```

### 2.2 设置 seed 与 ckpt 路径（改成实际路径）

```powershell
$env:MLF_SEED="2024"
$env:MLF_PRETRAINED_CKPT="checkpoints/fusion_seed2024_0_checkpoint.pth"
```

### 2.3 运行 eval

```powershell
python -u .\main_MLF_longterm.py `
  --use_duet 1 --is_training 0 --itr 1 `
  --root_path .\dataset --data_path ETTh1_fixed.csv `
  --pred_len 96 --batch_size 8 --num_workers 0 `
  --checkpoints .\runs\eval_fusion_stage1_seed2024
```

---

## 3. 训练 Fusion Stage1（从 preload ckpt 开始训练，最好采用baseline为基准权重）

### 3.1 清理前面的变量

```powershell
Remove-Item Env:MLF_PRETRAINED_CKPT -ErrorAction SilentlyContinue
Remove-Item Env:PRELOAD_CKPT -ErrorAction SilentlyContinue
```

### 3.2 设置 seed 与 preload ckpt（改成实际路径）

```powershell
$env:MLF_SEED="2024"
$env:PRELOAD_CKPT="checkpoints/baseline_seed_2024_0_checkpoint.pth"
$env:ROUTE1_DUET_ONLY="1"
$env:ROUTE1_STAGE="1"
```

### 3.3 运行训练

```powershell
python -u .\main_MLF_longterm.py `
  --use_duet 1 --is_training 1 --itr 1 `
  --root_path .\dataset --data_path ETTh1_fixed.csv `
  --pred_len 96 --batch_size 8 --num_workers 0 `
  --train_epochs 10 --learning_rate 0.0005 `
  --lradj none `
  --checkpoints .\runs\train_fusion_stage1_seed2024
```

---

## 4. 常用参数说明（最重要的几个）

- `--use_duet 0/1`：是否使用 fusion 模块  
- `--is_training 0/1`：0=只测试，1=训练  
- `--lradj none`：关闭学习率自动对半衰减  

### 环境变量

- `$env:MLF_SEED`：随机种子  
- `$env:MLF_PRETRAINED_CKPT`：评估时要加载的权重  
- `$env:PRELOAD_CKPT`：训练时要 preload 的权重  
- `$env:ROUTE1_STAGE`：1=stage1，2=stage2  
