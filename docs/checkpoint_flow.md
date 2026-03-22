# Checkpoint 模式流程图与代码分析

## 模式总览

```
BATCHDIFF_CKPT 参数值
├── -1  ：Full Checkpoint（禁用 log，使用默认 Trainer 保存）
├──  0  ：Log-based（每次保存都累积从初始模型开始的全部 update log）
└── >=1 ：Full + Log（每 N 步保存完整模型，中间步只保存 update log）
              └─ 可选: enable_shadow（CPU 实时 shadow 模型，用于即时恢复）

enable_shadow 只对 batch_size>=1 有效（默认关闭）。
batch_size<=0 时 enable_shadow 强制关闭。
```

---

## 模式 1: `batch_size = -1`（Full Checkpoint / 禁用）

### 目标
禁用差分检查点，使用 HuggingFace Trainer 的默认保存机制。

### 流程

```
训练开始 (on_train_begin)
│
├─ batch_size < 0 → 直接 return（不缓存模型、不注册 hook）
│
├─ 保存检查点: trainer._save_checkpoint()
│   ├─ batch_size < 0 → 跳过自定义逻辑
│   └─ super()._save_checkpoint(model, trial) → 默认 Trainer 保存
│
└─ 结果：标准 checkpoint 目录（model.safetensors + optimizer.pt + 其他）
```

---

## 模式 2: `batch_size = 0`（Log-based）

### 目标
从初始预训练模型开始，累积**所有** update log。恢复时通过 replay 重建模型。
**无 shadow**（enable_shadow 强制关闭）。

### 保存流程

```
训练开始 (on_train_begin)
│
├─ [首次训练] _cache_initial_model(model)
│   ├─ _detect_tied_weights() → 记录 tied weight groups
│   ├─ 记录 _trainable_param_names
│   ├─ model.state_dict() → base_checkpoint_state (CPU 副本)
│   ├─ base_checkpoint_path = "__initial__"
│   └─ base_checkpoint_step = 0
│
├─ [恢复训练] _init_for_resume(model, state, batchdiff_resume)
│   ├─ _detect_tied_weights()、记录 _trainable_param_names
│   ├─ 加载 optimizer.pt（一次性读取所有元数据）
│   ├─ 恢复 pending_grad、全量 update_history
│   ├─ base = "__initial__"、base_step = 0
│   └─ 缓存 model.state_dict() → base_checkpoint_state (CPU)
│
├─ 注册 _zo_update_hook (post-hook)
│
│── 每一步: _zo_update_hook(model, inputs, loss)
│   ├─ 获取 seed, applied_grad, new_grad, lr, wd, zo_eps
│   ├─ _pending_grad = new_grad
│   └─ if applied_grad != 0: update_history.append({step, seed, grad, lr, wd, zo_eps})
│
└─ 保存检查点: trainer._save_checkpoint()
    │
    ├─ os.makedirs(output_dir)
    ├─ 保存 trainer_state.json, scheduler.pt, rng_state
    │
    ├─ batch_size = 0 → is_full_step = False
    ├─ 复制 update_history (with update_lock)
    ├─ 构建 optimizer_state（原始 optimizer state + zo 元数据）
    │   └─ 元数据: zo_update_history, base_checkpoint="__initial__",
    │             current_step, batch_size=0, num_updates,
    │             tied_weights, model_dtype, pending_grad,
    │             trainable_param_names, is_full_checkpoint=False,
    │             zo_eps, rng_device
    ├─ torch.save(optimizer_state, optimizer.pt)
    ├─ is_full_step=False → 不调用 super()._save_checkpoint()
    └─ return（只保存了 optimizer.pt，无模型文件）

    on_save callback:
    ├─ save_count += 1
    └─ 无其他操作（base 永远是初始模型，不需要更新）
```

### 恢复流程

```
恢复 resume_from_batch_diff(checkpoint_path)
│
├─ 尝试加载 model.safetensors / pytorch_model.bin → 都不存在（log checkpoint）
├─ 加载 optimizer.pt → 有 zo_update_history
│
├─ 提取元数据: batch_size=0, base_checkpoint="__initial__", updates, ...
├─ base_checkpoint_ref = "__initial__"
├─ is_full_checkpoint = False
│
├─ _load_base_state("__initial__", pretrained_model_name, ...)
│   └─ AutoModelForCausalLM.from_pretrained(pretrained_model_name)
│
├─ _tie_state_dict_inplace(reconstructed, tied_groups)
├─ _replay_updates_on_state(reconstructed, updates, ...)  ← 重放所有 update
└─ return reconstructed
```

### 涉及的核心函数

| 函数 | 文件 | 作用 |
|------|------|------|
| `_cache_initial_model` | callback | 首次训练时缓存模型到 CPU |
| `_init_for_resume` | callback | 恢复训练时从 optimizer.pt 加载所有元数据 |
| `_zo_update_hook` | callback | 每步记录 {seed, grad, lr, wd, zo_eps} |
| `_save_checkpoint` | trainer.py | 保存 optimizer.pt（含全量 update log） |
| `resume_from_batch_diff` | callback | 从 base + replay 重建模型 |
| `_load_base_state` | callback | 加载基础模型（pretrained 或 checkpoint） |
| `_replay_updates_on_state` | callback | 按序重放 update log |

---

## 模式 3: `batch_size >= 1`（Full + Log）

### 目标
每 `batch_size` 步保存一次完整模型，中间步只保存 update log。
恢复时：完整检查点直接加载，日志检查点从基础模型 + replay 重建。

可选 `enable_shadow=True`：启用 CPU shadow 模型，用于即时恢复（默认关闭）。

### 保存流程

```
训练开始 (on_train_begin)
│
├─ [首次训练] _cache_initial_model(model)
│   ├─ base_checkpoint_step = 0
│   └─ if enable_shadow: _refresh_shadow_from_base()
│
├─ [恢复训练] _init_for_resume(model, state, batchdiff_resume)
│   ├─ 加载 optimizer.pt（一次性读取所有元数据）
│   ├─ 恢复 pending_grad
│   ├─ 完整检查点恢复 → base = resume_path, base_step = global_step, history = []
│   ├─ 日志检查点恢复 → 继承 base_checkpoint_path/step，加载 update_history
│   ├─ 缓存 model.state_dict() → base_checkpoint_state (CPU)
│   └─ if enable_shadow: _refresh_shadow_from_base()
│
├─ 注册 _zo_update_hook
├─ if enable_shadow: _start_shadow_thread()
│
│── 每一步: _zo_update_hook → update_history.append(...)
│
├─ Shadow 线程（如果 enable_shadow=True）：
│   ├─ [SHADOW_PIPELINE=0] 串行: 逐条 _apply_update_to_shadow()
│   └─ [SHADOW_PIPELINE=1] 流水线: P producer 预生成 z → ring buffer → consumer 更新 shadow
│
└─ 保存检查点: trainer._save_checkpoint()
    │
    ├─ batch_size >= 1
    ├─ steps_since_base = global_step - base_checkpoint_step
    ├─ is_full_step = (steps_since_base >= batch_size)
    │
    ├─ 构建 optimizer_state + zo 元数据
    │   └─ base_checkpoint = batchdiff_callback.base_checkpoint_path
    ├─ torch.save(optimizer_state, optimizer.pt)
    │
    ├─ if is_full_step:                          ← 完整检查点
    │   ├─ super()._save_checkpoint(model, trial)   → 保存 model.safetensors 等
    │   ├─ base_checkpoint_path = output_dir
    │   ├─ base_checkpoint_step = global_step
    │   └─ update_history = []                      → 清空 log
    │
    └─ else:                                     ← 日志检查点
        └─ return（只有 optimizer.pt，无模型文件）

    on_save callback:
    ├─ save_count += 1
    ├─ if is_full_step:
    │   ├─ _update_base_and_shadow(model, step)  ← 更新 base_checkpoint_state
    │   └─ if enable_shadow: _refresh_shadow_from_base()
    └─ else (log step):
        └─ 无操作（shadow worker 异步追赶中）
```

### 保存示例（batch_size=50, save_steps=10）

```
Step 10:  [Log]  optimizer.pt (10 updates, base="__initial__")
Step 20:  [Log]  optimizer.pt (20 updates, base="__initial__")
Step 30:  [Log]  optimizer.pt (30 updates, base="__initial__")
Step 40:  [Log]  optimizer.pt (40 updates, base="__initial__")
Step 50:  [Full] optimizer.pt (50 updates) + model.safetensors
                 → base 更新为 checkpoint-50, history 清空
Step 60:  [Log]  optimizer.pt (10 updates, base=checkpoint-50)
Step 70:  [Log]  optimizer.pt (20 updates, base=checkpoint-50)
...
Step 100: [Full] optimizer.pt (50 updates) + model.safetensors
                 → base 更新为 checkpoint-100, history 清空
```

### 恢复流程

```
恢复 resume_from_batch_diff(checkpoint_path)
│
├─ 加载 optimizer.pt → 提取元数据
├─ is_full_checkpoint?
│
├─ YES (完整检查点):
│   └─ load_batch_diff_checkpoint(ckpt_dir) → 直接加载 model.safetensors
│
└─ NO (日志检查点):
    ├─ base_checkpoint_ref = "checkpoint-50"（上一个完整检查点）
    ├─ _load_base_state(base_checkpoint_ref, ...)
    │   └─ load_batch_diff_checkpoint("checkpoint-50") → 加载基础模型
    ├─ _tie_state_dict_inplace(...)
    ├─ _replay_updates_on_state(reconstructed, updates)
    └─ return reconstructed
```

### 恢复后初始化（`_init_for_resume`）

恢复时不再调用 `_cache_initial_model`，而是使用 `_init_for_resume`：
- 从 `optimizer.pt` 一次性加载所有元数据（pending_grad、update_history、base 信息）
- `batch_size>=1` 完整检查点恢复 → `base_step = global_step`，`history = []`
- `batch_size>=1` 日志检查点恢复 → 从元数据继承 `base_checkpoint_path` 和 `base_checkpoint_step`
- `batch_size=0` → 加载全量 history，`base = "__initial__"`

---

## 公共函数调用关系

```
trainer._save_checkpoint()
├─ [batch_size >= 0] 自定义逻辑
│   ├─ torch.save(optimizer_state, optimizer.pt)      ← 所有模式共用
│   ├─ [is_full_step] super()._save_checkpoint()      ← 只有 batch_size>=1 的完整步
│   └─ callback.on_save()
│       ├─ [batch_size=0] 无操作（base 永远是初始模型）
│       └─ [batch_size>=1, full step] _update_base_and_shadow()
│           └─ [enable_shadow] _refresh_shadow_from_base()
│
└─ [batch_size == -1] super()._save_checkpoint()       ← 默认 Trainer 保存

trainer._load_from_checkpoint()
├─ [batchdiff_resume] skip（模型已通过 replay 恢复）
└─ [标准恢复] super()._load_from_checkpoint()

trainer._load_optimizer_and_scheduler()
├─ 加载 optimizer.pt
├─ 检测 zo_update_history → 剥离 zo 元数据
├─ optimizer.load_state_dict(cleaned_state)
└─ 加载 scheduler.pt

resume_from_batch_diff()
├─ load optimizer.pt → 提取元数据
├─ [is_full_checkpoint] load_batch_diff_checkpoint()    → 直接返回
├─ _load_base_state()                                   → 加载基础模型
├─ _tie_state_dict_inplace()
└─ _replay_updates_on_state()
    └─ _apply_single_update() × N          ← 串行重放（默认）
        └─ _generate_z_for_replay()
    （并行 replay 选项见文末「不常用功能」）

_start_shadow_thread()
├─ [SHADOW_PIPELINE=0] _shadow_worker()              ← 串行 shadow
│   └─ loop: _apply_update_to_shadow()
│       └─ _apply_single_update()                     ← 每步重新生成 z
│           └─ _generate_z_for_replay()
│
└─ [SHADOW_PIPELINE=1] _shadow_worker_pipelined()     ← 流水线 shadow
    ├─ P producer threads (ring buffer + threading.Event)
    │   └─ _generate_z_for_one_step()                 ← 预生成 z，释放 GIL
    └─ consumer (main shadow thread):
        └─ _apply_single_update_with_pregenerated_z() ← 用 buffered z 更新
            ├─ [adam_state=None] SGD 路径
            └─ [adam_state≠None] Adam 路径 (m/v/t on CPU)
```

---

# `mezo.sh` 参数速查表

> 所有参数均通过**环境变量**传入，格式: `KEY=VALUE bash mezo.sh`

## 训练基础参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MODEL` | `facebook/opt-1.3b` | HuggingFace 模型名 |
| `TASK` | (必填) | 任务名: SST2, SQuAD, CB, Copa, ReCoRD, DROP 等 |
| `MODE` | `ft` | 训练模式: `ft`(全参数), `prefix`(prefix tuning), `lora` |
| `DTYPE` | `fp16` | 模型精度: `fp16`(float16), `bf16`(bfloat16), `fp32`(float32) |
| `LR` | `1e-5` | 学习率 |
| `EPS` | `1e-3` | ZO perturbation epsilon |
| `ZO_METHOD` | `mezo-sgd` | ZO 优化器: `mezo-sgd`(SGD) 或 `mezo-adam`(Adam) |
| `BS` | `16` | per_device_train_batch_size |
| `SEED` | `0` | 全局随机种子 (控制数据顺序 + ZO perturbation seed 生成) |
| `STEPS` | `20000` | max_steps |
| `EVAL_STEPS` | `4000` | 每隔多少步做一次 evaluation |
| `LOGGING_STEPS` | `10` | 每隔多少步记录一次 training loss |
| `SAVE_STEPS` | `20000` | 每隔多少步保存一次 checkpoint |
| `TRAIN` | `1000` | 训练样本数 |
| `DEV` | `500` | 验证集样本数 (CB/Copa 自动改为 100) |
| `EVAL` | `1000` | 测试集样本数 |

## MeZO-Adam 参数

> 以下参数仅在 `ZO_METHOD=mezo-adam` 时生效

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ADAM_BETA1` | `0.9` | Adam 一阶矩衰减系数 β1 |
| `ADAM_BETA2` | `0.999` | Adam 二阶矩衰减系数 β2 |
| `ADAM_EPS` | `1e-8` | Adam epsilon (分母稳定项)，与 `EPS`(ZO perturbation ε) 不同 |

## 输出与控制

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `OUTPUT_ROOT` | `/lvs0/rccs-hpbdrt/minqiu/ZO_ckpt` | checkpoint 输出根目录. 可改为 `/tmp/ZO_ckpt`(本地NVMe) 或 `/dev/shm/ZO_ckpt`(DRAM) 测性能 |
|`FORCE_FSYNC` | `0`| FORCE_FSYNC=1时会强制将ckpt同步写入Local SSD, 但是其他模式下不会触发|
| `TRAIN_NAME` | `Test_staging_8` | 实验名前缀，拼入 output_dir |
| `DO_EVAL` | `1` | 设为 `0` 跳过训练后的 evaluation 阶段 (`--no_eval`) |
| `GPU_ID` | `0` | 使用哪块 GPU (设置 CUDA_VISIBLE_DEVICES) |
| `WANDB_PROJECT` | `NonStopZO2` | Weights & Biases 项目名 |

## L0-L3 Checkpoint 层级

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BATCHDIFF_CKPT` | `-1` | **核心开关**. `-1`=L0 Full Checkpoint; `0`=Log-based(累积所有updates); `N>=1`=Full+Log(每N步做一次full，中间存log) |
| `ENABLE_SHADOW` | `0` | 设为 `1` 启用 CPU Shadow Model. **对 `BATCHDIFF_CKPT>=0` 有效**（`=0` log-based 模式也可开启），仅 `=-1`(禁用模式) 时强制关闭 |
| `INSTANT_RECOVER` | `0` | 设为 `1` 启用即时恢复. 需要 `ENABLE_SHADOW=1`. GPU 故障时直接从 shadow 拷贝到 GPU，无需 replay |
| `GPU_FAIL_STEP` | `-1` | 在第 N 步模拟 GPU 故障. `-1`=不注入. 可独立使用或叠加任意层级 |
| `ASYNC_ANCHOR` | `0` | **异步 anchor**. 设为 `1` 启用异步写入 full checkpoint. **仅 `BATCHDIFF_CKPT>=1` 时有效**. GPU→CPU 异步拷贝 + 后台线程写盘，训练不阻塞. 若前一次写盘未完成则跳过本次 anchor (redo log 保证可恢复) |
| `OUTPUT_LOG` | `""` | **仅 `ASYNC_ANCHOR=1` 时有效**. log checkpoint 输出目录. 设置后 log checkpoints (optimizer.pt) 写入此目录, full checkpoints 仍在 `OUTPUT_ROOT`. 可设为 `/dev/shm/ZO_ckpt`(DRAM) 加速频繁的 log 写入 |
| `SHADOW_PIPELINE` | `0` | 设为 `1` 启用 **pipelined shadow**. P 个 producer 线程并行预生成 z，1 个 consumer 串行更新 shadow model，通过 ring buffer 连接. **需要 `ENABLE_SHADOW=1`**. GIL 不阻塞: PyTorch C++ ops 和 zo_rng C extension 均释放 GIL. 支持 SGD 和 Adam |
| `SHADOW_PIPELINE_WORKERS` | `2` | P = producer 线程数 = ring buffer slot 数. P>=2 才有 overlap. 使用 `calibrate_shadow_pipeline()` 确定最优 P. 每个 slot 占用 1×model_size 内存 |

> 并行 replay 参数见文末「不常用功能」。

## Resume / Replay 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `RESUME_CKPT` | `""` | L0 标准恢复: HuggingFace `resume_from_checkpoint` 路径 |
| `BATCHDIFF_RESUME` | `""` | L1+ 差分恢复: 指定 full checkpoint 路径，自动扫描并重放后续 differential checkpoints. **优先于 RESUME_CKPT** |
| `BATCHDIFF_REPLAY_DEVICE` | `cuda` | Replay 设备: `cpu` 或 `cuda`. **使用 `ZO_RNG_DEVICE=zo_rng` 时可安全设为 `cpu`**; 否则 replay 设备必须与训练设备一致 |
| `BATCHDIFF_SIMULATE_PERTURBATION` | `1` | `1`=replay 时模拟 [+1,-2,+1] perturbation 序列以精确还原 fp16 舍入; `0`=跳过, ~4x 更快但非 bitwise exact |
| `BATCHDIFF_REPLAY_FP32` | `0` | 设为 `1` 使用 fp32 精度做 replay. **仅对 `BATCHDIFF_REPLAY_DEVICE=cpu` 有效**（GPU replay 时忽略此参数）. 动机: CPU 上 `torch.normal(fp16)` 比 fp32 慢约 22x，upcast 到 fp32 做 replay 可大幅加速. **注意: 会引入累积误差**——训练在 fp16 下每步都有 fp16 舍入，而 fp32 replay 的舍入路径不同，步数越多偏差越大，重建结果**非 bitwise exact** |

> 并行 replay 参数（`PARALLEL_RECOVERY`、`CLOSEDFORM_RECOVERY` 等）见文末「不常用功能」。

## 确定性与随机数控制

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `DETERMINISTIC` | `0` | 设为 `1` 启用 `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8`. 强制 cuBLAS matmul 等所有 PyTorch 算子使用确定性实现，保证同一硬件上两次独立训练产生 bitwise 相同的 loss 轨迹. **注意: 不影响 replay 正确性**（replay 不涉及 cuBLAS），仅影响训练 forward pass. 代价: ~5-15% 性能下降 |
| `ZO_RNG_DEVICE` | `native` | ZO 扰动噪声 z 的生成设备. `native`=在参数所在设备上生成(快); `cpu`=始终在 CPU 生成再传到 GPU(跨设备可移植，但非常慢！); **`zo_rng`=使用 zo_rng 库生成跨设备 bit-exact 的噪声，使 `BATCHDIFF_REPLAY_DEVICE=cpu` 可以精确还原 GPU 训练** |

> **关于确定性的说明**:
> - 无论 `DETERMINISTIC` 设为何值，`cudnn.deterministic=True` 和 `cudnn.benchmark=False` **始终开启**
> - `DETERMINISTIC=1` 额外强制 cuBLAS 使用确定性 GEMM kernel，但**不保证跨 GPU 架构一致**（如 A100 vs H200），只保证同一硬件多次运行一致
> - **Replay (seed log 回放) 的正确性不依赖 `DETERMINISTIC` 设置**，因为 replay 全程只做 RNG + 逐元素运算，不涉及 cuBLAS matmul
> - **`ZO_RNG_DEVICE=zo_rng` 是唯一能保证 CPU replay = GPU 训练的选项**。原理: 使用 Philox4x32 PRNG + 多项式近似的 Box-Muller 变换，整个流程只用 IEEE 754 浮点加减乘除，在 CPU 和 GPU 上产生完全相同的结果
>
> **BF16 支持说明**:
> - `DTYPE=bf16` 完全支持. 训练、replay、shadow 均可使用 bfloat16
> - **CPU 上 z 生成速度**（Qwen3-1.7B, 903M params 为例）:
>   - `torch.randn`: fp32 ~12s, fp16 ~59s, **bf16 ~50s** — bf16 和 fp16 同样慢（PyTorch CPU 缺乏 bf16 native randn）
>   - `zo_rng.randn`: fp32 ~1.5s, fp16 ~2.3s, **bf16 ~2.7s** — 全 dtype 均快，bf16 仅比 fp32 慢 1.8x
> - **结论: Shadow pipeline (`SHADOW_PIPELINE=1`) 和 CPU replay 在使用 bf16 时必须搭配 `ZO_RNG_DEVICE=zo_rng`**，否则 CPU 端 z 生成会成为瓶颈（50s/step vs 2.7s/step）

---

# 不常用功能

> 以下功能为实验性质或仅用于特定场景，一般情况下不需要开启。

## 并行 Replay 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `PARALLEL_RECOVERY` | `0` | 设为 `1` 启用流水线 producer-consumer replay. P 个 producer 线程/stream 并行生成 z，1 个 consumer 串行更新参数，通过 ring buffer 连接. 结果与串行 replay **bitwise exact**. GPU+native 使用 CUDA streams; 其他情况使用 CPU threads |
| `PARALLEL_RECOVERY_WORKERS` | `1` | P = producer 数 = ring buffer 大小. P>=2 才有 overlap. 使用 `calibrate_parallel_recovery()` 确定最优 P. 每个 slot 占用 model_size 内存 (ZO2: 2×model_size) |
| `CLOSEDFORM_RECOVERY` | `0` | 设为 `1` 启用闭合形式并行 replay. 将 ZO-SGD 递推展开为独立项之和: `p_n = sp[0]*p_0 - Σ sp[t+1]*lr_t*grad_t*z_t`, W 个 worker 并行累加. 与串行 replay 近似相等 (非 bitwise exact). 不支持 perturbation simulation |
| `CLOSEDFORM_WORKERS` | `1` | W = 并行 worker 数. 内存 = 1×accum_buffer + W×z_buffer (共享累加, 无 per-worker partial sum). W=1 即为串行闭合形式 |
| `CLOSEDFORM_PRECISION` | `mixed` | 精度模式: `fp32`=全程 fp32(最准确); `fp16`=保持原始 dtype 累加(最快但累积误差大); `mixed`=参数保持原始 dtype, 累加用 fp32(默认, 推荐) |

## 并行 Replay 函数调用关系

```
_replay_updates_on_state()
├─ [默认: PARALLEL_RECOVERY=0, CLOSEDFORM_RECOVERY=0] 串行:
│   └─ _apply_single_update() × N
│       └─ _generate_z_for_replay()
│
├─ [PARALLEL_RECOVERY=1] 流水线 producer-consumer (bitwise exact):
│   └─ _parallel_replay_updates_on_state()
│       ├─ pre-compute seeds_info
│       ├─ [GPU+native] _pipelined_replay_gpu()
│       │   ├─ P CUDA streams (ring buffer)
│       │   ├─ pre-fill: schedule first P z on separate streams
│       │   └─ loop: default_stream.wait_event → apply update → schedule next z
│       └─ [CPU / other] _pipelined_replay_cpu()
│           ├─ P producer threads (ring buffer + threading.Event)
│           └─ main thread consumer: wait ready → apply update → signal free
│
└─ [CLOSEDFORM_RECOVERY=1] 闭合形式并行 (近似, 无 perturbation):
    └─ _closedform_replay_on_state()
        ├─ pre-compute suffix product sp[i] = Π_{j=i}^{n-1} (1-lr_j*wd_j)
        ├─ build term list: (coeff_wd, coeff_nowd, seed) per non-zero grad step
        ├─ [GPU+native] _closedform_gpu()
        │   └─ batches of W CUDA streams generate z, sync, accumulate into shared buffer
        ├─ [CPU / other] _closedform_cpu()
        │   └─ W threads generate z in parallel, lock-accumulate into shared buffer
        └─ finalize: p_n = sp[0]*p_0 - total_sum
```
