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
│   └─ 持续消费 update_history，逐条 _apply_update_to_shadow()
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
    └─ _apply_single_update() × N
        └─ _generate_z_for_replay()
```
