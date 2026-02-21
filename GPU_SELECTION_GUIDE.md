# ZO2 单GPU选择使用指南

由于 ZO2 目前只支持单GPU训练，本指南说明如何在多GPU环境中限制使用指定的GPU。

## 修改说明

已修改以下文件以支持GPU选择:

1. **`example/mezo_runner/mezo.sh`** - Shell脚本入口
2. **`example/mezo_runner/run.py`** - Python训练脚本
3. **`example/demo/train_zo2_with_hf_opt.py`** - Demo脚本

## 使用方法

### 方法1: 使用 mezo.sh (推荐)

通过环境变量 `GPU_ID` 指定要使用的GPU:

```bash
# 使用 GPU 0 (默认)
bash example/mezo_runner/mezo.sh

# 使用 GPU 1
GPU_ID=1 bash example/mezo_runner/mezo.sh

# 使用 GPU 2
GPU_ID=2 bash example/mezo_runner/mezo.sh

# 同时指定任务和GPU
TASK=SST2 GPU_ID=1 bash example/mezo_runner/mezo.sh
```

### 方法2: 直接运行 Python 脚本

#### Demo 脚本:

```bash
# 使用 GPU 0 (默认)
cd example/demo
python train_zo2_with_hf_opt.py

# 使用 GPU 1
GPU_ID=1 python train_zo2_with_hf_opt.py

# 使用 GPU 2
GPU_ID=2 python train_zo2_with_hf_opt.py
```

#### MeZO Runner:

```bash
# 使用 GPU 0 (默认)
cd example/mezo_runner
python run.py --model_name facebook/opt-125m --task_name SST2 ...

# 使用 GPU 1
GPU_ID=1 python run.py --model_name facebook/opt-125m --task_name SST2 ...
```

### 方法3: 命令行直接设置 (最灵活)

在运行任何脚本之前设置 `CUDA_VISIBLE_DEVICES`:

```bash
# 使用 GPU 0
CUDA_VISIBLE_DEVICES=0 python run.py ...

# 使用 GPU 1
CUDA_VISIBLE_DEVICES=1 python run.py ...

# 使用 GPU 3
CUDA_VISIBLE_DEVICES=3 bash mezo.sh
```

## 验证GPU选择

运行脚本后，您应该看到类似的输出:

```
Using GPU: 0
```

您也可以在 Python 中验证:

```python
import torch
print(f"可见的GPU数量: {torch.cuda.device_count()}")  # 应该输出 1
print(f"GPU名称: {torch.cuda.get_device_name(0)}")
```

或使用命令行工具:

```bash
# 运行训练时，在另一个终端查看GPU使用情况
nvidia-smi
```

只有您指定的GPU会显示使用中。

## 常见问题

### Q: 为什么要在导入 torch 之前设置 CUDA_VISIBLE_DEVICES?

**A:** PyTorch 在第一次导入时会初始化CUDA上下文，之后设置 `CUDA_VISIBLE_DEVICES` 将不会生效。

### Q: 如果我有4个GPU，想用第3个(GPU 2)怎么办?

**A:**
```bash
GPU_ID=2 bash mezo.sh
```
或
```bash
CUDA_VISIBLE_DEVICES=2 python run.py ...
```

### Q: 可以使用多个GPU吗?

**A:** 不可以。ZO2目前只支持单GPU。如果您设置 `CUDA_VISIBLE_DEVICES=0,1`，PyTorch会看到2个GPU，但ZO2的 `_zo2_unsupported_conditions` 会抛出错误:
```
NotImplementedError: Currently ZO2 only support one working device
```

### Q: 修改后会影响原有功能吗?

**A:** 不会。这些修改只是添加了GPU选择功能，默认行为不变(使用GPU 0)。

## 示例: 完整训练命令

```bash
# 在 GPU 1 上训练 SST2 任务，使用 OPT-1.3B 模型
GPU_ID=1 TASK=SST2 MODEL=facebook/opt-1.3b bash example/mezo_runner/mezo.sh

# 在 GPU 2 上运行 demo
cd example/demo
GPU_ID=2 python train_zo2_with_hf_opt.py --model_name facebook/opt-125m
```

## 技术细节

修改实现原理:

1. **mezo.sh**: 在脚本开头设置 `export CUDA_VISIBLE_DEVICES=${GPU_ID:-0}`
2. **run.py & train_zo2_with_hf_opt.py**: 在导入 torch 之前检查并设置 `os.environ["CUDA_VISIBLE_DEVICES"]`

这样确保了在 PyTorch 初始化之前就限制了可见的GPU，达到单GPU运行的效果。
