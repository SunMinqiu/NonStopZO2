# Log-Based Checkpoint 架构与流程

本文档只覆盖和 checkpoint / replay / RNG / shadow / failure injection / async anchor 直接相关的逻辑。
不展开训练算法本身，也不展开 `zo` / `zo2` 的推导细节。

## 参数与开关速查

### 训练入口参数

这些参数来自两层：

- `transformers.TrainingArguments`
- `example/mezo_runner/run.py` 里的 `OurArguments`

#### 训练 / 优化

| 参数 | 默认值 | 来源 | 说明 |
|------|--------|------|------|
| `per_device_train_batch_size` | HF 默认 | `TrainingArguments` | 每卡 batch size |
| `gradient_accumulation_steps` | HF 默认 `1` | `TrainingArguments` | 梯度累积步数；决定一次 optimizer update 对应多少 dataloader step |
| `max_steps` | HF 默认 `-1` | `TrainingArguments` / demo 脚本 | 训练总 update 步数；大于 0 时优先于 `num_train_epochs` |
| `num_train_epochs` | HF 默认 `3.0` | `TrainingArguments` | epoch-based 训练轮数 |
| `learning_rate` | HF 默认 | `TrainingArguments` | 优化器学习率 |
| `weight_decay` | HF 默认 | `TrainingArguments` / demo 脚本 | 权重衰减 |
| `lr_scheduler_type` | HF 默认 | `TrainingArguments` | 学习率调度器类型 |
| `trainer` | `"none"` | `OurArguments` | `none` / `regular` / `zo` |
| `zo_method` | `"mezo-sgd"` | `OurArguments` | ZO 方法名 |
| `zo_mode` | `ZO_MODE` 或 `"zo"` | `OurArguments` | `zo` / `zo2` |
| `zo_eps` | `1e-3` | `OurArguments` | MeZO 扰动半径 |
| `adam_beta1` | `0.9` | `OurArguments` | MeZO-Adam beta1 |
| `adam_beta2` | `0.999` | `OurArguments` | MeZO-Adam beta2 |
| `adam_eps` | `1e-8` | `OurArguments` | MeZO-Adam epsilon |

#### 长度 / token / 生成

| 参数 | 默认值 | 来源 | 说明 |
|------|--------|------|------|
| `max_length` | `2048` | `OurArguments` | mezo_runner 里模型可接收的最大输入长度 |
| `max_seq_length` | `None` | `zo2/trainer/hf_trl/sft_trainer.py` | SFTTrainer 数据截断长度 |
| `max_new_tokens` | `50` | `OurArguments` | 生成任务的新 token 上限 |
| `block_size` | `1024` / 各模型 preset | `zo2/model/nanogpt/model.py` | NanoGPT 上下文长度 |
| `max_position_embeddings` | 模型 config | HF model config | OPT/Qwen 等模型原生最大上下文长度 |

#### 输出 / 恢复 / 保存路径

| 参数 | 默认值 | 来源 | 说明 |
|------|--------|------|------|
| `output_dir` | HF 默认 | `TrainingArguments` | 主输出目录；checkpoint、initial model、默认 trace 都写这里 |
| `overwrite_output_dir` | HF 默认 `False` | `TrainingArguments` | 允许覆盖已有输出目录 |
| `resume_from_checkpoint` | `None` | `TrainingArguments` / runner | 标准 HF resume 入口 |
| `RESUME_CKPT` | `""` | `mezo.sh` env | shell 层 `resume_from_checkpoint` 别名；只适用于标准 HF full checkpoint resume |
| `log_output_dir` | `""` | `OurArguments` | shadow 关闭或 async anchor 下单独放 log checkpoints 的目录 |
| `shadow_resume` | `""` | `OurArguments` | 显式指定 shadow snapshot 或 flat header 用于 soft recovery |
| `RESET_OUTPUT_DIR` | `1` | `mezo.sh` env | 训练前是否清空 `OUTPUT_DIR`；若同时传 `LOG_BASED_RESUME` / `RESUME_CKPT` 则报错；若 `INSTANT_RECOVER=1` 则忽略，不删除 `OUTPUT_DIR` |
| `save_strategy` | HF 默认 | `TrainingArguments` | checkpoint 触发策略 |
| `save_steps` | HF 默认 | `TrainingArguments` | `save_strategy=steps` 时的保存间隔 |
| `save_total_limit` | HF 默认 | `TrainingArguments` | 保留 checkpoint 数量上限 |
| `logging_steps` | HF 默认 | `TrainingArguments` | Trainer 日志步数间隔；也会影响部分诊断默认周期 |

#### 数据 / 任务 / 模型选择

| 参数 | 默认值 | 来源 | 说明 |
|------|--------|------|------|
| `task_name` | `"SST2"` | `OurArguments` | 任务名 |
| `model_name` | `"facebook/opt-125m"` | `OurArguments` | HuggingFace 模型名 |
| `num_train` | `0` | `OurArguments` | 训练样本数或 ICL demo 数 |
| `num_dev` | `None` | `OurArguments` | dev 样本数 |
| `num_eval` | `None` | `OurArguments` | eval 样本数 |
| `num_train_sets` | `None` | `OurArguments` | 抽样训练集份数 |
| `train_set_seed` | `None` | `OurArguments` | 训练样本抽样 seed |
| `load_float16` | `False` | `OurArguments` | 以 fp16 加载模型 |
| `load_bfloat16` | `False` | `OurArguments` | 以 bf16 加载模型 |
| `load_int8` | `False` | `OurArguments` | 以 int8 加载模型 |
| `offloading_device` | `"cpu"` | `OurArguments` | ZO2 offload 设备 |
| `working_device` | `"cuda:0"` | `OurArguments` | ZO2 工作设备 |

### checkpoint / shadow / resume

| 参数 | 默认值 | 来源 | 说明 |
|------|--------|------|------|
| `log_based_ckpt` / `LOG_BASED_CKPT` | `-1` | `OurArguments` / env | `-1`=L0；`0`=Log-based；`N>=1`=Full+Log |
| `enable_shadow` / `ENABLE_SHADOW` | `0` | `OurArguments` / env | `LOG_BASED_CKPT>=0` 时可启用 shadow |
| `instant_recover` / `INSTANT_RECOVER` | `0` | `OurArguments` / env | 开启即时恢复流程 |
| `gpu_fail_step` / `GPU_FAIL_STEP` | `-1` | `OurArguments` / env | 在给定步数触发故障注入 |
| `log_based_resume` / `LOG_BASED_RESUME` | `""` | `OurArguments` / env | 明确指定 replay 恢复入口 |
| `resume_from_checkpoint` / `RESUME_CKPT` | `None` / `""` | `TrainingArguments` / env | 标准 HF full checkpoint 恢复入口；`mezo.sh` 会把 `RESUME_CKPT` 转成 `--resume_from_checkpoint` |
| `log_based_replay_device` / `LOG_BASED_REPLAY_DEVICE` | `cuda` | `OurArguments` / env | replay 执行设备 |
| `log_based_simulate_perturbation` / `LOG_BASED_SIMULATE_PERTURBATION` | `1` | `OurArguments` / env | replay 是否模拟 perturbation-restore |
| `log_based_replay_fp32` / `LOG_BASED_REPLAY_FP32` | `0` | `OurArguments` / env | CPU replay 时临时 upcast 到 fp32 |
| `async_anchor` / `ASYNC_ANCHOR` | `0` | `OurArguments` / env | 启用 async anchor |
| `SHADOW_WAIT_READY` | `1` | env | 训练启动时是否等待 shadow ready |
| `SHADOW_PIPELINE` | `0` | env | 是否启用 pipelined shadow |
| `SHADOW_PIPELINE_WORKERS` | `2` | env | shadow pipeline producer 数 |
| `SHADOW_COMMIT_INTERVAL` | `1` | env | shadow durable commit 的步数间隔 |
| `SHADOW_FLAT_COMMIT` | `0` | env | 是否使用 `/dev/shm/zo_ckpt/*.flat*` 单 buffer shadow |
| `SHADOW_RESERVE_THREADS` | `1` | env | shadow 为训练预留的核数 |
| `SHADOW_CONSUMER_THREADS` | `auto` | env | shadow consumer 的 ATen 线程数 |

### RNG / persistence / deterministic

| 参数 | 默认值 | 来源 | 说明 |
|------|--------|------|------|
| `zo_rng_device` / `ZO_RNG_DEVICE` | `native` | `OurArguments` / env | `native` / `cpu` / `zo_rng` |
| `ZO_RNG_TRAIN_THREADS` | unset | env | 训练进程 zo_rng 线程数 |
| `ZO_RNG_NUM_THREADS` | 环境决定 | env | zo_rng 线程池初始大小 |
| `deterministic` | `False` | `OurArguments` | 是否开启 deterministic algorithms |
| `FORCE_FSYNC` | `0` | env | 对 initial model、optimizer.pt、full checkpoint 启用 fsync |
| `PARALLEL_RECOVERY` | `0` | env | legacy pipelined replay 开关 |
| `PARALLEL_RECOVERY_WORKERS` | `1` | env | legacy pipelined replay worker 数 |
| `CLOSEDFORM_RECOVERY` | `0` | env | legacy closed-form replay 开关 |
| `CLOSEDFORM_WORKERS` | `1` | env | legacy closed-form worker 数 |
| `CLOSEDFORM_PRECISION` | `mixed` | env | legacy closed-form 精度模式 |

### 日志开关

运行时日志分两层：

- 3 个分类总开关只负责轻量日志，默认都关闭。
- 会拖慢训练 / shadow / replay 的热路径日志和重诊断日志，全部单独拆成独立开关，默认都关闭。

#### 轻量分类开关

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ZO_LOG_TIME` | `0` | 轻量时间日志；checkpoint / resume / replay / shadow / async anchor 的阶段耗时摘要，不含每步 timing |
| `ZO_LOG_RESOURCE` | `0` | 轻量资源日志；CPU / GPU / RSS / pinned buffer / thread env 摘要，不含每步资源采样和 thread snapshot |
| `ZO_LOG_CONSISTENCY` | `0` | 轻量一致性日志；恢复来源、VERIFY、模式自动探测、shadow-vs-base 判定等状态说明，不含 checksum / exact hash / batch / RNG 诊断 |

#### 热路径 / 重诊断开关

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ZO_LOG_MEMORY_DEBUG` | `0` | 训练侧 `[MemDebug]`；包含 `torch.cuda.synchronize()`，会拖慢训练 |
| `ZO_LOG_THREAD_SNAPSHOT` | `0` | `[ThreadSnap]`；扫描 `/proc/.../task` 和线程池，启动期和诊断期有额外开销 |
| `ZO_LOG_BATCH_DEBUG` | `0` | `[BATCH]`；对 batch tensor 做 `cpu()/hash/head` 摘要 |
| `ZO_LOG_STATE_DIAG` | `0` | `STATE-CKSUM` / `ADAM-BRIEF` / `ADAM-CKSUM` / `[CKSUM]`；会遍历整份 model / Adam state |
| `ZO_LOG_STATE_EXACT` | `0` | `STATE-EXACT` / `ADAM-EXACT` / `DIAG-EXACT`；会做完整 hash / exact compare |
| `ZO_LOG_RNG_DIAG` | `0` | `[RNG-DIAG]`；会抓取并 hash CPU / CUDA / numpy / python RNG state |
| `ZO_LOG_Z_DIAG` | `0` | `[Z-CKSUM]`；会汇总 Adam replay/update 里生成的 `z` |
| `ZO_LOG_Z_EXACT` | `0` | `[Z-EXACT]`；会对 `z` 做完整 hash |
| `ZO_LOG_OPT_STEP` | `0` | `[OPT]`；训练热路径每步打印 seed / applied_grad / new_grad |
| `ZO_LOG_SHADOW_SEND_TIME` | `0` | `[ShadowSend]`；训练热路径每步记录 enqueue 时间 |
| `ZO_LOG_REPLAY_STEP_TIME` | `0` | `[Replay] update ... time=...` / legacy `[PipelinedReplay] update ...`；每步 replay timing |
| `ZO_LOG_SHADOW_STEP_TIME` | `0` | `[Shadow Timing]`；shadow 每步 apply / zgen / commit timing |
| `ZO_LOG_SHADOW_STEP_RESOURCE` | `0` | `[Shadow Resource]`；shadow 每步 RSS 采样 |
| `ZO_LOG_TRAIN_STEP_RESOURCE` | `0` | `[LogBased] step=... | CPU=... | GPU alloc=...`；训练热路径周期性资源采样 |

### Trace 开关

结构化 trace 默认关闭。开启后会输出 `jsonl`，并由 `tools/visualize_trace.py` 在脚本或 Jupyter 里直接生成 figure 对象。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ZO_TRACE` | `0` | 总开关；开启结构化 trace |
| `ZO_TRACE_PATH` | `OUTPUT_DIR/zo_trace.jsonl` | trace 输出路径；不设时默认写到训练输出目录 |
| `ZO_TRACE_RUN_ID` | auto | 本次训练 / 恢复的 trace run id |
| `ZO_TRACE_TIMELINE` | `1` | 输出时序事件 |
| `ZO_TRACE_RESOURCE` | `1` | 输出资源采样事件 |
| `ZO_TRACE_RESOURCE_INTERVAL_SEC` | `1.0` | 资源采样周期；按 wall time 独立后台线程采样，不耦合训练 step |

## 0. 当前实现说明

当前实现已经切到 `flat single-buffer shadow + async anchor disk checkpoint`：

- shadow 子进程维护一份普通 CPU working copy
- committed shadow 写到 `/dev/shm/zo_ckpt/zo_shadow_latest_<hash>.flat*`
- async anchor / full checkpoint 仍然写磁盘 `checkpoint-N/model.safetensors`
- shadow flat snapshot 现在由三部分 metadata 约束：
  - `*.flat.header.json`
  - `*.flat.state.meta.json`
  - `*.flat.adam.meta.json`（Adam 模式）
- 另外还有一个代际 sidecar：
  - `*.flat.generation.meta.json`
- writer 会把实际使用的 `layout / adam_layout` 持久化进 header；loader 优先按 header 内 layout 读，不再现场重建
- shadow flat 的完整性模式现在只有两档：
  - `full_sha256`：记录并验证 `state_sha256 / adam_m_sha256 / adam_v_sha256`
  - `header_only`：只验证 `writing/ready + generation + committed_step + adam_t`
- `full_sha256` 只用于初始化，以及 rebase 后的第一次 commit；其余 commit 全部是 `header_only`
- soft recovery / resume 读 shadow flat 时，会同时校验：
  - `expected_generation == snapshot generation`
  - `generation / committed_step / adam_t`
  - 如果 `integrity_mode=full_sha256`，再校验内容哈希

也就是说，当前 shadow 协议的目标不是“总能拿到最新一步”，而是：

- 只发布 `durable_step`
- `durable_step` 对应的 `model / adam / metadata` 必须自洽
- 一旦 metadata 和内容不一致，恢复直接失败，不再 silent corruption

下面仍然提到 manifest / shm / safetensors-only shadow 的段落是旧设计，待继续清理；以代码实现为准。

## 1. 文件边界

| 文件 | 主要职责 | 关键函数 / 方法 |
|------|----------|-----------------|
| `zo2/trainer/hf_transformers/log_based_checkpoint.py` | 训练期 callback、运行期状态、shadow orchestration、兼容别名 | `LogBasedCheckpointCallback.on_train_begin` / `_cache_initial_model` / `_init_for_resume` / `_start_shadow_process` / `_zo_update_hook` / `recover_from_shadow` / `_reconstruct_on_demand` / `get_recovery_status` / `on_save` / `on_train_end` |
| `zo2/trainer/hf_transformers/log_based_resume.py` | checkpoint 读取、base model 选择、replay 恢复入口 | `load_log_based_checkpoint` / `_load_base_state` / `resume_from_log_based` |
| `zo2/trainer/hf_transformers/log_based_replay.py` | 当前仍在使用的串行 replay 主路径；按环境变量分发到 legacy replay | `_generate_z_for_replay` / `_generate_z_for_one_step` / `_apply_single_update` / `_apply_single_update_with_pregenerated_z` / `_replay_updates_on_state` |
| `zo2/trainer/hf_transformers/legacy_pipeline_closed_form_replay.py` | 已废弃但暂留的 pipelined replay 和 closed-form replay | `_parallel_replay_updates_on_state` / `_closedform_replay_on_state` / `validate_closedform_replay` |
| `zo2/trainer/hf_transformers/log_based_shadow.py` | shadow 子进程入口、串行/pipelined shadow、shadow safetensors 读写 | `_shadow_process_main` / `_shadow_process_serial` / `_shadow_process_pipelined` / `_commit_shadow_state` / `_load_shadow_replica` |
| `zo2/trainer/hf_transformers/log_based_failure_injection.py` | GPU 故障注入、`SIGKILL` 顺序控制 | `GPUFailureSimulator.set_fail_step` / `check_and_fail` / `trigger_failure` |
| `zo2/trainer/hf_transformers/log_based_utils.py` | tied weights、fsync、内存统计、线程快照 | `_detect_tied_weights` / `_tie_state_dict_inplace` / `_restore_tied_weights` / `_fsync_file` / `_fsync_directory` / `_log_memory` / `_thread_snapshot` |
| `zo2/trainer/hf_transformers/async_anchor_checkpoint.py` | async anchor 的 GPU→CPU→`/dev/shm/zo_ckpt latest`→disk 两阶段异步落盘 | `AsyncAnchorCheckpointer.try_save_full_checkpoint` / `_persist_worker` / `get_latest_published_anchor_step` / `get_latest_completed_anchor_step` / `shutdown` |
| `zo2/trainer/hf_transformers/log_based_tuning.py` | shadow / replay 的 benchmark 和线程分配标定 | `calibrate_producer_consumer` / `optimize_thread_allocation` |
| `zo2/trainer/hf_transformers/trainer.py` | HF Trainer 接线、保存 `optimizer.pt` 元数据、恢复优化器状态 | `ZOTrainer._load_from_checkpoint` / `_save_checkpoint` / `_load_optimizer_and_scheduler` |

## 2. 运行期状态

核心运行期状态都挂在 `LogBasedCheckpointCallback` 上：

| 状态 | 含义 | 定义位置 |
|------|------|----------|
| `base_checkpoint_state` | 当前 base model 的 CPU 副本；恢复和 on-demand replay 的起点 | `log_based_checkpoint.py` `LogBasedCheckpointCallback.__init__` |
| `base_checkpoint_path` / `base_checkpoint_step` | 当前 redo log 依附的 base checkpoint 路径与步数 | `log_based_checkpoint.py` `LogBasedCheckpointCallback.__init__` |
| `update_history` | 从 base 之后记录的 update log | `log_based_checkpoint.py` `LogBasedCheckpointCallback.__init__` |
| `_pending_grad` | 本步新算出、下步才会真正应用的 grad | `log_based_checkpoint.py` `_zo_update_hook` |
| `_pending_seed` | 生成 `_pending_grad` 的 seed；ZO2 恢复 RNG 用 | `log_based_checkpoint.py` `_zo_update_hook` |
| `_base_pending_seed` | 当前 base 对应的“前置 seed”；replay 第一条 update 会用到 | `trainer.py` `_save_checkpoint` / `log_based_checkpoint.py` `on_train_end` |
| `shadow_shared` / `shadow_model` | shadow 的共享内存张量 | `log_based_checkpoint.py` `_refresh_shadow_from_base` |
| `shadow_step_val` | shadow 子进程共享步数计数器 | `log_based_checkpoint.py` `_start_shadow_process` |
| `failure_simulator` | GPU 故障注入器 | `log_based_checkpoint.py` `__init__` |
| `shadow_adam_state` | shadow 端的 CPU Adam m/v/t | `log_based_checkpoint.py` `_init_shadow_adam_state` |

## 3. 模式总览

`LOG_BASED_CKPT` 控制 checkpoint 模式：

```text
LOG_BASED_CKPT = -1   -> L0: Disabled, 使用默认 Trainer full checkpoint
LOG_BASED_CKPT = 0    -> L1: Log-based, base 固定为 "__initial__"
LOG_BASED_CKPT >= 1   -> L1/L2/L3: Full + Log, 每 N 步更新 base

ENABLE_SHADOW=1       -> L2: 维护 CPU shadow
INSTANT_RECOVER=1     -> L3: 配置上启用即时恢复
GPU_FAIL_STEP>0       -> 配置故障注入，触发 SIGKILL
ASYNC_ANCHOR=1        -> full checkpoint 改成 async anchor 语义
```

注意：

- `enable_shadow` 对 `batch_size >= 0` 都有效。
- 只有 `batch_size = -1` 会强制关闭 shadow。
- `batch_size = 0` 时也允许 shadow 和 soft recovery。

### 3.1 Resume 参数优先级与约束

resume 入口分三类：

- `LOG_BASED_RESUME`
- `RESUME_CKPT` / `resume_from_checkpoint`
- `INSTANT_RECOVER=1` 时的运行期即时恢复

它们的关系是：

1. `LOG_BASED_RESUME` 优先级最高。  
   `mezo.sh` 会把它转换成 `--log_based_resume ...`，并且不会再追加 `--resume_from_checkpoint`。

2. `RESUME_CKPT` 是 shell 层别名。  
   `mezo.sh` 会把它转换成 `--resume_from_checkpoint ...`。它适用于标准 HF full checkpoint 恢复，不适用于 `LOG_BASED_CKPT>=0` 的 log-based replay 恢复。

3. `resume_from_checkpoint` 是 Python / HF Trainer 层入口。  
   如果直接调用 `run.py` 或 notebook 内部构造 `TrainingArguments`，这是标准 HF resume 参数名。

4. `LOG_BASED_CKPT>=0` 时，如果要从 log checkpoint / replay 语义恢复，应使用 `LOG_BASED_RESUME`，不要用 `RESUME_CKPT`。

5. `RESET_OUTPUT_DIR=1` 不能和 `LOG_BASED_RESUME` / `RESUME_CKPT` 同时使用。  
   `mezo.sh` 现在会在任何清理动作发生之前先做这条检查；若冲突直接报错退出。

6. `INSTANT_RECOVER=1` 时，`RESET_OUTPUT_DIR` 被忽略。  
   这是为了保留当前输出目录和恢复所需状态，不删除 `OUTPUT_DIR`。

## 4. 训练开始阶段

### 4.1 入口

文件 / 函数：

- `zo2/trainer/hf_transformers/log_based_checkpoint.py`
  - `LogBasedCheckpointCallback.on_train_begin`

### 4.2 行为

`on_train_begin` 按下面顺序初始化：

1. 记录 `output_dir` 和 `trainer`。
2. 如果 `batch_size < 0`，直接返回，只保留 L0 路径。
3. 注册 ZO post-hook：`_zo_update_hook`。
4. 探测 `model_dtype`，用于恢复时复用相同 dtype。
5. 如果 `args.log_based_resume` 非空，走 `_init_for_resume`。
6. 否则走 `_cache_initial_model`。
7. 同步 `current_step = state.global_step`。
8. 如果 `enable_shadow=True`，初始化 shadow 的 Adam state。
9. 启动 shadow 子进程：`_start_shadow_process`。
10. 打印 checkpoint / replay / shadow / thread env 配置。

## 5. 初始模型缓存与 base 建立

### 5.1 Fresh training

文件 / 函数：

- `zo2/trainer/hf_transformers/log_based_checkpoint.py`
  - `LogBasedCheckpointCallback._cache_initial_model`
- `zo2/trainer/hf_transformers/log_based_utils.py`
  - `_detect_tied_weights`
  - `_fsync_file`

### 5.2 逻辑

`_cache_initial_model` 会做这些事：

1. 检测 tied weights，记录到 `_tied_weight_groups`。
2. 记录 trainable param 名称顺序 `_trainable_param_names`。
3. 把 `model.state_dict()` clone 到 CPU，建立 `base_checkpoint_state`。
4. 如果启用了 shadow：
   - 切换到 `torch.multiprocessing.set_sharing_strategy('file_system')`
   - 把 `base_checkpoint_state` 放到 POSIX shared memory
   - 从 `resource_tracker` 注销 shm 文件，避免 `SIGKILL` 后被自动清理
5. 如果启用了 shadow，调用 `_refresh_shadow_from_base` 建立 `shadow_shared`。
6. 设置 `base_checkpoint_path="__initial__"`、`base_checkpoint_step=0`。
7. 把 initial model 落盘到 `output_dir/initial_model/`，作为后续 resume 的本地 base cache。
8. 保存时会剔除 tied duplicate key，避免和 HuggingFace `save_pretrained()` 约定冲突。

### 5.3 文档上容易漏掉的点

- `initial_model/` 是 resume 时优先读取的本地 base，不只是调试产物。
- tied weights 在这里已经开始参与“保存格式”设计。
- `FORCE_FSYNC=1` 时，initial model 也会 `fsync`。

## 6. Resume 初始化与 callback 状态重建

### 6.1 文件 / 函数

- `zo2/trainer/hf_transformers/log_based_checkpoint.py`
  - `LogBasedCheckpointCallback._init_for_resume`

### 6.2 逻辑

`_init_for_resume` 处理的是“模型权重已经被恢复好了，callback 自己的运行时状态还没补上”的阶段。

它会：

1. 重新检测 tied weights。
2. 重新记录 `_trainable_param_names`。
3. 读取目标 checkpoint 的 `optimizer.pt`。
4. 如果有 `pending_grad`，恢复到 `model.opt.projected_grad`。
5. 如果有 `pending_seed`，恢复 ZO2 用的 `last_rstate`。
6. 如果是 Adam，优先恢复 replay 后的 Adam state；否则回退 checkpoint 自带的 `adam_state`。
7. 根据 `is_full_checkpoint` 区分：
   - full checkpoint：当前 checkpoint 自己就是新的 base，`update_history=[]`
   - log checkpoint：从 metadata 继承 `base_checkpoint_path/base_checkpoint_step/update_history`
8. 如果 `batch_size==0`，强制把 base 设回 `"__initial__"`。
9. 重新缓存当前模型到 `base_checkpoint_state`。
10. 如果启用 shadow，调用 `_refresh_shadow_from_base`，让 shadow 从“已恢复后的模型”开始跟跑。

### 6.3 文件 / 函数映射

| 逻辑 | 文件 / 函数 |
|------|-------------|
| 恢复 `pending_grad` | `log_based_checkpoint.py` `LogBasedCheckpointCallback._init_for_resume` |
| 恢复 `pending_seed` / `last_rstate` | `log_based_checkpoint.py` `LogBasedCheckpointCallback._init_for_resume` |
| 恢复 replay 后的 Adam state | `log_based_checkpoint.py` `LogBasedCheckpointCallback._init_for_resume` + `log_based_replay.py` `_get_and_clear_replay_adam_state` |
| 从 full/log checkpoint 分流 | `log_based_checkpoint.py` `LogBasedCheckpointCallback._init_for_resume` |

## 7. update log 记录

### 7.1 文件 / 函数

- `zo2/trainer/hf_transformers/log_based_checkpoint.py`
  - `LogBasedCheckpointCallback._zo_update_hook`

### 7.2 逻辑

每个训练步结束后，`_zo_update_hook` 会：

1. 读出 `seed / applied_grad / new_grad / lr / wd / zo_eps`。
2. 把 `new_grad` 保存到 `_pending_grad`。
3. 把当前 seed 保存到 `_pending_seed`。
4. 把当前 step 的 update 追加到 `update_history`。
5. 如果存在 shadow queue，非阻塞地把 update 送进 shadow 子进程。

这里有两个容易被忽略的设计：

- 即使 `grad == 0` 也照样记录 entry。
  作用：保留 perturbation-restore 的数值痕迹，并维持 seed chain。
- `actual_step = current_step + 1`。
  作用：hook 调用时机在 `global_step` 自增之前，所以需要手动补一。

## 8. 训练期 checkpoint 保存

### 8.1 文件 / 函数

- `zo2/trainer/hf_transformers/trainer.py`
  - `ZOTrainer._save_checkpoint`
  - `ZOTrainer._maybe_log_save_evaluate`
- `zo2/trainer/hf_transformers/log_based_utils.py`
  - `_fsync_file`
  - `_fsync_directory`

### 8.2 统一保存逻辑

当 callback 存在且 `batch_size >= 0` 时，`ZOTrainer._save_checkpoint` 统一保存：

1. `trainer_state.json`
2. `scheduler.pt`
3. RNG state：`self._save_rng_state(output_dir)`
4. `optimizer.pt`

其中 `optimizer.pt` 不只是 optimizer state，还会额外挂这些 checkpoint metadata：

| 字段 | 含义 |
|------|------|
| `zo_update_history` | 当前 redo log |
| `base_checkpoint` | 当前 redo log 依附的 base |
| `current_step` | 当前步数 |
| `batch_size` | checkpoint mode |
| `num_updates` | redo log 条数 |
| `tied_weights` | tied groups |
| `model_dtype` | 模型 dtype |
| `pending_grad` | 下一步待应用 grad |
| `pending_seed` | 生成 `pending_grad` 的 seed |
| `base_pending_seed` | base 对应的前置 seed |
| `zo2` | 是否启用 ZO2 delayed update 语义 |
| `trainable_param_names` | replay 参数顺序 |
| `zo_eps` | replay 需要的扰动幅度 |
| `rng_device` | replay 用同一种 RNG 设备策略 |
| `is_full_checkpoint` | 这个目录是否能直接 load model 文件 |
| `zo_method` | `mezo-sgd` / `mezo-adam` |
| `adam_betas` / `adam_eps_value` | Adam replay 初始化需要的超参 |

### 8.3 L0 / Log-based / Full+Log 三种保存路径

#### L0: `LOG_BASED_CKPT=-1`

文件 / 函数：

- `trainer.py` `ZOTrainer._save_checkpoint`

逻辑：

- 直接走 `super()._save_checkpoint()`
- 如果 `FORCE_FSYNC=1`，额外对整个 checkpoint 目录执行 `_fsync_directory`

#### Log-based: `LOG_BASED_CKPT=0`

文件 / 函数：

- `trainer.py` `ZOTrainer._save_checkpoint`
- `log_based_checkpoint.py` `LogBasedCheckpointCallback.on_save`

逻辑：

- `base_checkpoint` 固定写成 `"__initial__"`
- `is_full_step=False`
- 只写 `optimizer.pt` 和常规 trainer 状态
- 不写模型文件
- `on_save` 只递增计数，不更新 base

#### Full + Log: `LOG_BASED_CKPT>=1`

文件 / 函数：

- `trainer.py` `ZOTrainer._save_checkpoint`
- `log_based_checkpoint.py` `LogBasedCheckpointCallback.on_save`
- `log_based_checkpoint.py` `LogBasedCheckpointCallback._update_base_and_shadow`

逻辑：

1. 根据 `global_step - base_checkpoint_step` 判断是否到 full step。
2. 普通 log step：
   - 只写 `optimizer.pt`
3. full step：
   - 同步模式：调用 `super()._save_checkpoint()` 写模型文件
   - 然后把 `base_checkpoint_path/base_checkpoint_step` 更新到当前 step
   - 把 `_base_pending_seed` 更新为当前 `_pending_seed`
   - 清空 `update_history`
4. `on_save` 中，如果这是 full step，再调用 `_update_base_and_shadow`

## 9. Resume / Load 分流

### 9.1 文件 / 函数

- `zo2/trainer/hf_transformers/log_based_resume.py`
  - `load_log_based_checkpoint`
  - `_load_base_state`
  - `resume_from_log_based`
- `zo2/trainer/hf_transformers/trainer.py`
  - `ZOTrainer._load_from_checkpoint`
  - `ZOTrainer._load_optimizer_and_scheduler`

### 9.2 三路分流

`resume_from_log_based()` 不是“永远 replay”。它会先分流：

1. 如果 checkpoint 目录里有 `model.safetensors` / `pytorch_model.bin`
   - 直接走 `load_log_based_checkpoint`
2. 如果有 `optimizer.pt` 但没有 `zo_update_history`
   - 视为 regular checkpoint，仍然走 `load_log_based_checkpoint`
3. 只有 `optimizer.pt` 且里面有 `zo_update_history`
   - 才进入 log-based replay 路径

### 9.3 `base` 的选择顺序

`_load_base_state()` 的优先级：

1. `base_checkpoint_ref == "__initial__"` 时
   - 先看 `output_dir/initial_model/model.safetensors`
   - 再看 `output_dir/initial_model/pytorch_model.bin`
   - 本地都没有，才回退到 `AutoModelForCausalLM.from_pretrained(pretrained_model_name)`
2. `base_checkpoint_ref != "__initial__"` 时
   - 直接 `load_log_based_checkpoint(base_checkpoint_ref)`

### 9.4 Trainer 接线层的保护逻辑

| 逻辑 | 文件 / 函数 |
|------|-------------|
| `--log_based_resume` 时跳过 HF 默认 model load | `trainer.py` `ZOTrainer._load_from_checkpoint` |
| `batch_size=-1` 却误用 log-based checkpoint 时直接报错 | `trainer.py` `ZOTrainer._load_from_checkpoint` |
| 读取 `optimizer.pt` 后剥掉 log-based metadata 再加载 optimizer | `trainer.py` `ZOTrainer._load_optimizer_and_scheduler` |

## 10. replay 主路径

### 10.1 文件 / 函数

- `zo2/trainer/hf_transformers/log_based_replay.py`
  - `_generate_z_for_replay`
  - `_generate_z_for_one_step`
  - `_apply_single_update`
  - `_apply_single_update_with_pregenerated_z`
  - `_replay_updates_on_state`

### 10.2 核心分支

`_replay_updates_on_state()` 先决定三件事：

1. 设备：CPU 还是 CUDA
2. RNG 设备策略：`native` / `cpu` / `zo_rng`
3. dtype：原 dtype 还是 `replay_in_fp32`

### 10.3 RNG 分支

| 模式 | 行为 | 文件 / 函数 |
|------|------|-------------|
| `rng_device="native"` | 在参数当前所在设备上生成 z | `log_based_replay.py` `_generate_z_for_replay` |
| `rng_device="cpu"` | 在 CPU 生成，再搬到参数设备 | `log_based_replay.py` `_generate_z_for_replay` |
| `rng_device="zo_rng"` | 用 zo_rng 生成跨设备 bit-exact z | `log_based_replay.py` `_generate_z_for_replay` / `_generate_z_for_one_step` |

如果在 CPU 上 replay、机器又有 CUDA、且 `rng_device != "zo_rng"`，代码会显式打 warning：

- `log_based_replay.py` `_replay_updates_on_state`
- `legacy_pipeline_closed_form_replay.py` `_parallel_replay_updates_on_state`
- `legacy_pipeline_closed_form_replay.py` `_closedform_replay_on_state`

### 10.4 dtype 分支

如果 `replay_in_fp32=True` 且实际 replay 设备是 CPU，代码会：

1. 把 fp16 / bf16 参数临时 upcast 到 fp32
2. replay 完后再 downcast 回原 dtype

对应函数：

- `log_based_replay.py` `_replay_updates_on_state`
- `legacy_pipeline_closed_form_replay.py` `_parallel_replay_updates_on_state`

### 10.5 Adam replay

文件 / 函数：

- `log_based_replay.py`
  - `_load_adam_state_from_base`
  - `_set_replay_adam_state`
  - `_get_and_clear_replay_adam_state`

逻辑：

- full checkpoint 直接 load model，不做 Adam replay
- log checkpoint 恢复时，如 `zo_method == mezo-adam`：
  - 先从 base checkpoint 的 `optimizer.pt` 取 Adam state
  - replay 时同步推进 `m/v/t`
  - replay 完后缓存给 callback，供 `_init_for_resume` 恢复回 optimizer

## 11. legacy replay 路径

### 11.1 文件 / 函数

- `zo2/trainer/hf_transformers/log_based_replay.py`
  - `_replay_updates_on_state`
- `zo2/trainer/hf_transformers/legacy_pipeline_closed_form_replay.py`
  - `_parallel_replay_updates_on_state`
  - `_closedform_replay_on_state`
  - `validate_closedform_replay`

### 11.2 现状

这些路径已经是 legacy / discarded：

- `PARALLEL_RECOVERY=1` 时，active replay 会转发到 `_parallel_replay_updates_on_state`
- `CLOSEDFORM_RECOVERY=1` 时，active replay 会转发到 `_closedform_replay_on_state`

文档保留它们，是为了说明为什么主 replay 文件仍然 import 它们，而不是推荐继续扩展。

## 12. shadow 生命周期

### 12.1 文件 / 函数

- `zo2/trainer/hf_transformers/log_based_checkpoint.py`
  - `_init_shadow_adam_state`
  - `_refresh_shadow_from_base`
  - `_manifest_path`
  - `_step_file_path`
  - `_unregister_shm`
  - `_save_shadow_manifest`
  - `_start_shadow_process`
  - `recover_from_shadow`
  - `get_recovery_status`
  - `on_train_end`
- `zo2/trainer/hf_transformers/log_based_shadow.py`
  - `_shadow_process_main`
  - `_shadow_process_serial`
  - `_shadow_process_pipelined`
  - `_load_shadow_from_manifest`

### 12.2 建立 shadow

`_refresh_shadow_from_base()` 会：

1. clone `base_checkpoint_state`
2. 放入 shared memory，形成 `shadow_shared`
3. 重新 tie tied weights
4. 从 `resource_tracker` 注销这些 shm 文件
5. 生成 manifest，写到 `/dev/shm/zo_ckpt/zo_shadow_manifest_<hash>.json`
6. 设置 `shadow_step = len(update_history)`

### 12.3 启动 shadow 子进程

`_start_shadow_process()` 会建立这些 IPC 通道：

| 通道 | 作用 |
|------|------|
| `update_queue` | 训练进程把 update / refresh / stop 发给 shadow |
| `shadow_step_val` | 共享步数计数器 |
| `recovery_req` | 请求 shadow 暂停 |
| `recovery_ready` | shadow 表示“现在可以 clone” |
| `recovery_done` | clone 完毕，shadow 可以继续跑 |
| `shadow_stop_event` | 训练结束时快速停机 |

### 12.4 shadow 子进程模式

#### 串行 shadow

文件 / 函数：

- `log_based_shadow.py` `_shadow_process_serial`

逻辑：

1. 先把 `shadow_shared` clone 到本地 heap `shadow_local`
2. 循环处理：
   - `stop`
   - `refresh`
   - 普通 update
3. 普通 update 时：
   - 先生成 z
   - 再应用 update
   - 再把本地结果 copy 回 `shadow_shared`
   - 写 step file

#### pipelined shadow

文件 / 函数：

- `log_based_shadow.py` `_shadow_process_pipelined`

逻辑：

1. P 个 producer 线程负责预生成 z
2. consumer 主线程按 step 顺序消费结果并更新 `shadow_shared`
3. 支持：
   - `stop`
   - `refresh`
   - producer 异常传递
   - tied weights 健康检查
   - Adam state reset

### 12.5 Shadow refresh

full checkpoint / resume 后，shadow 不是“自己推断新 base”，而是显式 `refresh`：

- `log_based_checkpoint.py` `_update_base_and_shadow`
- `log_based_shadow.py` `_shadow_process_serial`
- `log_based_shadow.py` `_shadow_process_pipelined`

训练进程把 `base_checkpoint_state` 更新好后，往 `update_queue` 发：

```python
{'cmd': 'refresh', 'new_step': len(self.update_history)}
```

shadow 收到后会：

1. 用 `base_shared` 覆盖 shadow 当前状态
2. 更新 `shadow_step_val`
3. 清空 pipeline 内部缓存
4. 如果是 Adam，清空 `m/v/t`

## 13. soft recovery 与 instant recovery

### 13.1 in-process shadow recovery

文件 / 函数：

- `log_based_checkpoint.py` `recover_from_shadow`

逻辑：

1. 请求 shadow pause
2. clone `shadow_shared`
3. 读取 `shadow_step`
4. 释放 shadow 继续跑
5. 返回“已恢复到 shadow_step 的 state_dict”

如果没有 shadow，就回退到：

- `log_based_checkpoint.py` `_reconstruct_on_demand`

### 13.2 out-of-process soft recovery

文件 / 函数：

- `log_based_resume.py` `resume_from_log_based`
- `log_based_shadow.py` `_load_shadow_from_manifest`

逻辑：

1. 新进程通过 `shadow_manifest` 读取 shadow 的 shm 文件
2. 读出 `shadow_step`
3. 只 replay `updates[shadow_step:]`

这就是“DRAM survives”的 soft recovery 路径。

### 13.3 recovery status

文件 / 函数：

- `log_based_checkpoint.py` `get_recovery_status`

提供：

- `gpu_step`
- `shadow_step`
- `shadow_available`
- `shadow_lag`
- `can_recover`
- `batch_size`

## 14. failure injection

### 14.1 文件 / 函数

- `zo2/trainer/hf_transformers/log_based_failure_injection.py`
  - `GPUFailureSimulator.set_fail_step`
  - `GPUFailureSimulator.check_and_fail`
  - `GPUFailureSimulator.trigger_failure`
- `zo2/trainer/hf_transformers/log_based_checkpoint.py`
  - `LogBasedCheckpointCallback.on_step_begin`

### 14.2 触发时机

故障检查发生在 `on_step_begin()`，不是 `_zo_update_hook()`。

原因：

- 要在下一次 ZO forward 之前就把进程打掉
- 这样 resumed run 才能拿到和原始 run 一样的数据 batch

### 14.3 kill 顺序

`trigger_failure()` 的顺序是：

1. 扫当前进程和 shadow 进程的子进程，找到 `torch_shm_manager`
2. 先 `SIGKILL` 这些 shm manager
3. 再 `SIGKILL` shadow 子进程
4. flush logging handler
5. 最后 `SIGKILL` 主进程

这段顺序是 soft recovery 成败的关键，不能只写成“模拟 GPU failure”。

## 15. tied weights

### 15.1 文件 / 函数

- `zo2/trainer/hf_transformers/log_based_utils.py`
  - `_detect_tied_weights`
  - `_tie_state_dict_inplace`
  - `_restore_tied_weights`

### 15.2 三个阶段

1. 训练开始：
   - `_detect_tied_weights(model)`
2. 保存 initial/full checkpoint：
   - 去掉 tied duplicate key
3. 读取 regular/full checkpoint 或 replay 前：
   - `_restore_tied_weights(state_dict, checkpoint_dir)`
   - `_tie_state_dict_inplace(reconstructed, tied_groups)`

## 16. RNG 与确定性

### 16.1 文件 / 函数

- `zo2/trainer/hf_transformers/log_based_replay.py`
  - `_generate_z_for_replay`
  - `_generate_z_for_one_step`
  - `_apply_single_update`
  - `_replay_updates_on_state`
- `zo2/trainer/hf_transformers/log_based_checkpoint.py`
  - `_apply_update_to_shadow`
  - `on_train_begin`
- `zo2/trainer/hf_transformers/trainer.py`
  - `_save_checkpoint` 中的 `optimizer_state['rng_device']`
  - `_save_checkpoint` 中的 `self._save_rng_state(output_dir)`

### 16.2 需要明确写到文档里的行为

- replay 会优先使用 checkpoint 里保存的 `rng_device`
- `ZO_RNG_DEVICE=zo_rng` 是跨设备精确 replay 的唯一安全选项
- `shadow` 在 `rng_device="native"` 且训练跑在 CUDA 上时，只是近似 shadow，不是 bitwise exact
- `pending_seed` 和 `base_pending_seed` 都是为了恢复 seed chain，不是冗余元数据

## 17. async anchor

### 17.1 文件 / 函数

- `zo2/trainer/hf_transformers/async_anchor_checkpoint.py`
  - `AsyncAnchorCheckpointer.try_save_full_checkpoint`
  - `_persist_worker`
  - `get_latest_completed_anchor_step`
  - `get_latest_completed_anchor_path`
  - `shutdown`
- `zo2/trainer/hf_transformers/trainer.py`
  - `ZOTrainer._save_checkpoint`
- `zo2/trainer/hf_transformers/log_based_checkpoint.py`
  - `LogBasedCheckpointCallback.on_train_end`

### 17.2 两阶段语义

#### Phase 1: 训练线程

文件 / 函数：

- `async_anchor_checkpoint.py` `AsyncAnchorCheckpointer.try_save_full_checkpoint`

逻辑：

1. 等前一个 persist 完成：`self._persist_done.wait()`
2. 等 pinned buffer 空闲：`self._buffer_free`
3. 把 GPU 参数异步 copy 到 pinned CPU buffer
4. 把 persist job 放到后台线程队列

#### Phase 2: persist 线程

文件 / 函数：

- `async_anchor_checkpoint.py` `AsyncAnchorCheckpointer._persist_worker`

逻辑：

1. 等 copy event 完成
2. clone pinned buffer 到普通 CPU 内存
3. 立刻释放 pinned buffer 给下一次 anchor
4. `fork` 子进程写 `model.safetensors`
5. 父进程 `waitpid` 等待写完
6. 成功后更新 `latest_completed_step/path`

### 17.3 base 更新语义

对应文件 / 函数：

- `trainer.py` `ZOTrainer._save_checkpoint`
- `log_based_checkpoint.py` `LogBasedCheckpointCallback.on_train_end`

注意：

- async anchor 模式下，写 `optimizer.pt` 时 `is_full_checkpoint` 永远记为 `False`
- base 不会在 enqueue anchor 时立刻更新
- 只有 anchor 真正持久化完成后，才会：
  - 更新 `base_checkpoint_path/base_checkpoint_step`
  - 截断 `update_history`
  - 记录新的 `_base_pending_seed`

这保证了恢复总能走“已持久化 base + redo log”而不是指向一个还没写完的 full checkpoint。

## 18. 清理与收尾

### 18.1 文件 / 函数

- `zo2/trainer/hf_transformers/log_based_checkpoint.py`
  - `LogBasedCheckpointCallback.on_train_end`

### 18.2 逻辑

训练结束时会：

1. `async_anchor.shutdown()`，等待最后一个 persist 完成
2. 如有必要，做最终 base trim
3. 停止 shadow：
   - `shadow_stop_event.set()`
   - `recovery_done.set()` 防止等待死锁
   - `update_queue.put_nowait({'cmd': 'stop'})` 作为 fallback
4. join shadow 子进程；超时则 terminate
5. 删除：
   - `base_checkpoint_state`
   - `shadow_shared`
   - `shadow_model`
6. 清理：
   - manifest 文件
   - step 文件
   - `/dev/shm/zo_ckpt` 中对应的 shm data file

### 18.3 2026-03-29 排障增量

这一轮排障的目标只有一个：

- 明确区分“在线 replay / shadow 计算差异”和“真正的恢复错版”
- 把 `model / adam / metadata / log` 的恢复来源和内容一致性钉死

当前代码中已经落地的增量如下。

#### A. shadow writer / loader 协议

- shadow flat snapshot 现在是两档完整性协议：
  - `full_sha256`：初始化和 rebase 后第一次 commit 使用；`state.meta / adam.meta` 记录完整哈希
  - `header_only`：普通周期 commit 使用；只验 header/meta 一致性，不重扫整份 state
- `_load_shadow_bundle_flat()` 会先检查 `*.flat.generation.meta.json` 里的 `expected_generation`
  - 如果 `expected_generation > snapshot generation`，说明 rebase 已经发生，但 rebase 后首个 durable commit 没完成；loader 直接拒绝这份 stale shadow，走 hard recovery
- 当 `integrity_mode=full_sha256` 时，loader 读完内容后会重算 hash 并验证；不一致直接报错
- writer 会把 `layout / adam_layout` 持久化到 `header`；resume/load 优先使用 header 里的 layout，而不是根据 base template 现场重建 layout

这一点是为了解决之前可能出现的两类问题：

- metadata 说自己是 `step=K`，但内容其实不是 `K`
- writer 和 loader 如果用不同 layout 解释同一份 `.bin`，会把字节切错

#### B. shadow 步数语义

shadow 进程现在明确区分三种步数：

- `applied_step`
  - CPU `working_state / adam_state` 已经 apply 到哪一步
- `desired_commit_step`
  - 当前希望落盘到哪一步
- `durable_step`
  - 真正已经写完并可恢复的是哪一步

训练 / 健康日志中的：

- `applied=...`
- `desired=...`
- `durable=...`

对应的就是这三种语义。resume 只信 `durable_step`。

#### C. commit / rebase 调度

- pipeline shadow 在写盘前会先停 producer，清掉过期预取，再对静止的 `working_state` 做 commit
- rebase 后不再把“内存里大概到哪”直接当成可恢复步数；只有 commit 成功后才发布新的 `durable_step`

#### D. resume 来源日志

resume 现在固定打印：

- `[Resume] Shadow-vs-base comparison: shadow_step=..., base_step=..., exact_match=...`
- `[Resume Sources] model_source=... model_step=... | adam_source=... adam_step=... | log_source=... log_start_step=... log_end_step=...`

这两条日志用于回答两个问题：

- 这次 soft recovery 选的是 `base` 还是 `shadow`
- replay 的起点到底是从哪一步开始

#### E. durable 边界对账

训练 / shadow 两侧都加了 durable 边界日志：

- `train_durable_ref step=N`
- `shadow_durable step=N`
- `shadow_loaded step=N`

用途分别是：

- `train_durable_ref`
  - live train 在 commit 边界的参考值
- `shadow_durable`
  - shadow 刚发布 `durable_step` 时，内存中的 snapshot
- `shadow_loaded`
  - resume 时从 flat storage 读出来的 snapshot

这一轮已经确认：

- `shadow_durable step=64`
- `shadow_loaded step=64`

在统一 hash helper 后，`STATE-EXACT / ADAM-EXACT` 可以直接比。

#### F. 本轮定位结论

结合 `/home/users/u0001609/ZO_log/29.log`，当前能确认的结论是：

- “shadow_durable 写的是 A，shadow_loaded 读出来是 B” 这条恢复错版，在当前协议下**没有新证据**
- 之前一部分“读出来不一样”的判断，来自两套不同的 `STATE-EXACT` helper；这一点已经统一
- layout mismatch 也是实打实的风险点，现已通过“writer 持久化 layout，loader 按 header layout 读取”修掉
- 如果后面再出现 `model / adam / metadata / log` 错版，优先会在 `_load_shadow_bundle_flat()` 的内容校验里直接报错，而不是静默继续恢复

## 19. 日志与诊断

### 日志分层

先区分两类：

- 轻量分类日志：受 `ZO_LOG_TIME` / `ZO_LOG_RESOURCE` / `ZO_LOG_CONSISTENCY` 控制。
- 热路径 / 重诊断日志：各自独立开关控制，默认全部关闭。

#### optimizer 层

- 时间类：
  - 无常驻时间日志。
- 系统资源类：
  - 无。
- 一致性校验类：
  - 轻量：
    - 无。
  - 热路径 / 重诊断：
    - `[OPT]` -> `ZO_LOG_OPT_STEP`
    - `[OPT-DIAG]` -> `ZO_LOG_STATE_DIAG` / `ZO_LOG_STATE_EXACT`
    - `[RNG-DIAG]` -> `ZO_LOG_RNG_DIAG`
    - `[Z-CKSUM]` -> `ZO_LOG_Z_DIAG`
    - `[Z-EXACT]` -> `ZO_LOG_Z_EXACT`

#### trainer / checkpoint orchestration 层

- 时间类：
  - `[ZOTrainer SavePath]`
  - `[ZOTrainer SaveBreakdown]`
  - `[LogBased InitBaseTiming]`
  - `[LogBased FullCkpt]`
  - `[Full Resume]` / `[No Resume]`
  - 热路径：
    - `[ShadowSend]` -> `ZO_LOG_SHADOW_SEND_TIME`
- 系统资源类：
  - `[LogBased] Memory: ...`
  - `[Thread Env — Training Process]`
  - 热路径 / 重诊断：
    - `[MemDebug]` -> `ZO_LOG_MEMORY_DEBUG`
    - `[ThreadSnap]` -> `ZO_LOG_THREAD_SNAPSHOT`
    - 每 10 步一次的 `[LogBased] step=... | CPU=... | GPU alloc=...` -> `ZO_LOG_TRAIN_STEP_RESOURCE`
- 一致性校验类：
  - `[HOOK]`
  - `[VERIFY]`
  - 热路径 / 重诊断：
    - `[BATCH]` -> `ZO_LOG_BATCH_DEBUG`
    - `[CKSUM]` -> `ZO_LOG_STATE_DIAG`
    - `train_live / train_durable_ref / train_snapshot / shadow_snapshot` 对应的 `STATE-* / ADAM-*`
      -> `ZO_LOG_STATE_DIAG` / `ZO_LOG_STATE_EXACT`

#### resume / replay 层

- 时间类：
  - `[Replay Timing]`
  - `[Resume] Target checkpoint`
  - `[Resume] Replay device`
  - `[Resume] Replaying ... updates`
  - `[Resume] Completed!`
  - legacy 路径的 `[PipelinedReplay] ... in ...` / `[ClosedForm] ... in ...`
  - 热路径：
    - `[Replay] update ... time=...` -> `ZO_LOG_REPLAY_STEP_TIME`
    - legacy `[PipelinedReplay] update ...` -> `ZO_LOG_REPLAY_STEP_TIME`
- 系统资源类：
  - `[Memory] before replay / after replay`
  - legacy 路径的 `pipelined start/done`、`closedform start/after accumulation/done`
- 一致性校验类：
  - `[Resume Sources]`
  - `[VERIFY-RESUME]`
  - `[Resume] Shadow-vs-base comparison`
  - 热路径 / 重诊断：
    - `base_ready / shadow_loaded / after_tie_before_replay / after_replay` 对应的 `DIAG-* / STATE-* / ADAM-*`
      -> `ZO_LOG_STATE_DIAG` / `ZO_LOG_STATE_EXACT`

#### shadow 层

- 时间类：
  - `[Shadow BootTiming]`
  - 热路径：
    - `[Shadow Timing]` -> `ZO_LOG_SHADOW_STEP_TIME`
- 系统资源类：
  - `[Shadow Boot]`
  - `[Shadow Flat]`
  - 热路径 / 重诊断：
    - `[Shadow Resource]` -> `ZO_LOG_SHADOW_STEP_RESOURCE`
    - `[ThreadSnap] Shadow ...` -> `ZO_LOG_THREAD_SNAPSHOT`
- 一致性校验类：
  - 热路径 / 重诊断：
    - `shadow_boot / shadow_live / shadow_durable` 对应的 `STATE-* / ADAM-*`
      -> `ZO_LOG_STATE_DIAG` / `ZO_LOG_STATE_EXACT`

#### async anchor 层

- 时间类：
  - `[AsyncAnchor] Waiting for buffer ...`
  - `[AsyncAnchor] Queued anchor step ...`
  - `[AsyncAnchor] Persisted step ...`
  - `[AsyncAnchor] Shutdown complete. Stats: ...`
  - `[AsyncAnchor] Summary: ...`
- 系统资源类：
  - `[AsyncAnchor] Init: ... pinned buffer`
  - `[AsyncAnchor] Excluding tied keys from checkpoint: ...`
- 一致性校验类：
  - 无独立校验日志，沿用 checkpoint / replay 的状态校验日志。

### 重复日志清理结果

- 已删除重复的 full checkpoint 总耗时日志，只保留 `[ZOTrainer SavePath]`。
- 已删除 resume 阶段重复的 replay 完成耗时日志，只保留 replay 层 timing 摘要。
- 已删除 shadow 子进程重复的线程 / 模式启动日志。
- shadow 每步混合日志已拆成：
  - `[Shadow Timing]`
  - `[Shadow Resource]`

### 关键日志标签

| 日志标签 | 含义 |
|----------|------|
| `[Resume Sources]` | 这次恢复实际用的是哪份 `model / adam / log` |
| `[Resume] Shadow-vs-base comparison` | `shadow_step` 对应内容是否等于当前 base checkpoint |
| `[STATE-EXACT] train_durable_ref step=N` | live train 在 durable 边界的完整 state hash |
| `[STATE-EXACT] shadow_durable step=N` | shadow 刚发布 durable snapshot 时的完整 state hash |
| `[DIAG-EXACT] shadow_loaded step=N` | resume 从 shadow flat 读出来后的完整 state hash |
| `[ADAM-EXACT] ...` | 对应时刻的完整 `m/v/t` hash |
| `[Shadow Timing] ... applied=... desired=... durable=...` | shadow 每步 apply / zgen / commit timing 与 durable 进度 |
| `[Shadow Resource] ... durable=... RSS=...` | shadow 每步 RSS 资源占用 |
| `[Replay] Replaying ...` | replay 的设备、扰动模式、rng 设备 |
| `[VERIFY-RESUME]` | 最后一条 replayed update 和 `pending_grad` 的恢复提示 |

## 20. Trace

### 20.1 目标

trace 用于两类图：

- 时序图：
  - `GPU / Train`
  - `CPU / Shadow`
- 资源图：
  - GPU / CPU / RSS / `/dev/shm`
  - 以及流程积压指标 `backlog / lag`

这里的 `backlog / lag` 定义为：

- `shadow_apply_backlog = train_step - shadow_applied_step`
- `shadow_durable_lag = train_step - shadow_durable_step`
- `anchor_lag = train_step - anchor_completed_step`
- `update_history_len = len(update_history)`

### 20.2 输出文件

- 默认输出：
  - `OUTPUT_DIR/zo_trace.jsonl`
- 可视化脚本：
  - `tools/visualize_trace.py`

用法：

```bash
ZO_TRACE=1 python your_train.py ...
python tools/visualize_trace.py /path/to/output_dir/zo_trace.jsonl
python tools/visualize_trace.py /path/to/output_dir/zo_trace.jsonl --html
```

默认行为：

- `plot_timeline(...)` 返回 `matplotlib` timeline figure
- `plot_resources(...)` 返回 `matplotlib` resources figure
- `plot_loss(...)` 返回 `matplotlib` loss-step figure
- `plot_interactive(...)` 返回 `plotly` figure

### 20.3 Schema

每条 trace 记录是一行 JSON，核心字段如下：

| 字段 | 含义 |
|------|------|
| `run_id` | 同一次训练 / 恢复的唯一标识 |
| `event_id` | 单个事件 span 的唯一 id |
| `triggered_by` | 因果关联；只做关联，不跨 lane 配对 |
| `wall_time_ns` | wall time，用于跨进程 / 跨线程统一时间轴 |
| `pid` / `tid` | 进程 / 线程 id |
| `thread_name` | 线程名 |
| `process_role` | `train` / `shadow` |
| `panel` | `gpu_train` / `cpu_shadow` |
| `lane` | panel 内子 lane |
| `event` | 事件名 |
| `phase` | `B` / `E` / `I` |
| `step` | 对应训练步 |
| `duration_ms` | `E` 事件上的持续时间 |
| `counters` | 轻量数值指标 |
| `extra` | 轻量标签字段 |

注意：

- `B` 和 `E` 只在同一个 `panel + lane` 内配对。
- 不用跨 lane 的 `B/E` 表达因果。
- `anchor` 的后台线程事件和训练线程阻塞事件都归到 `gpu_train` panel，不单独伪装成第三个进程层。

### 20.4 时序事件

#### GPU / Train panel

主 lane：

- `framework_overhead`
  - HF Trainer 在 step 之间的框架开销；包括 dataloader 取数、调度、callback 框架逻辑等。
- `train_step`
  - 单个训练 step 的主区间。

blocking lane：

- `wait_shadow_ready`
- `zo_update_hook`
- `log_send_cpu`
- `checkpoint_save`
- `checkpoint_d2h_copy`
- `checkpoint_cpu_serialize`
- `checkpoint_rng_save`
- `checkpoint_disk_persist`
- `checkpoint_model_persist`
- `full_checkpoint_refresh`
- `wait_anchor_persist`
- `wait_anchor_buffer`
- `recover_shadow`
- `replay_updates`
- `train_end_cleanup`
- `resume_begin` / `resume_end`

anchor_thread lane：

- `anchor_d2h_copy`
- `anchor_enqueue`
- `anchor_publish_latest`
- `anchor_persist`

framework lane 的 instant：

- `train_begin`
- `train_end`

counter lane 的 instant：

- `train_progress`
- `train_scalar`

#### CPU / Shadow panel

shadow_main lane：

- `shadow_process_start`
- `shadow_boot`
- `shadow_wait_update`
- `shadow_apply`
- `shadow_commit`
- `shadow_rebase`
- `shadow_stop`

resource lane 的 instant：

- `resource_sample`

### 20.5 资源采样

资源采样由独立 daemon 线程按 wall time 周期触发，不在训练热路径里主动采样 GPU/CPU 资源。
这套基础采样挂在 `ZOTrainer` 本身，因此和 `LOG_BASED_CKPT` 模式无关；即使 `LOG_BASED_CKPT=-1`，只要 `ZO_TRACE=1` 且 `ZO_TRACE_RESOURCE=1`，也会有训练进程的 CPU/GPU0 资源样本。

`gpu_train` panel 的 `resource_sample` 可能包含：

- `gpu0_alloc_mb`
- `gpu0_reserved_mb`
- `gpu0_peak_mb`
- `train_rss_mb`
- `train_cpu_percent`
- `zo_shm_used_mb`
- `update_history_len`
- `active_base_step`
- `shadow_applied_step`
- `shadow_durable_step`
- `anchor_completed_step`

`cpu_shadow` panel 的 `resource_sample` 可能包含：

- `shadow_rss_mb`
- `shadow_cpu_percent`
- `shadow_durable_step`
- `zo_shm_used_mb`

`train_progress` 额外提供：

- `shadow_apply_backlog`
- `shadow_durable_lag`
- `anchor_lag`
- `update_history_len`

`train_scalar` 提供：

- `loss`
- `learning_rate`

说明：

- `train_scalar.loss` 直接复用 HF Trainer 在 `_maybe_log_save_evaluate()` 中已经计算好的 `logs["loss"]`
- `loss-step` 图按 HF 的 logging cadence 采样
- `loss-step` 图的横轴使用 trace 事件上的 `global step`

### 20.6 可视化脚本

时间 summary 的用户侧命名：

- `checkpoint_total`
  - 来源：`checkpoint_save`
- `t_step`
  - 来源：`train_compute`
- `t_l`
  - 来源：`log_send_cpu`
- `t_d2h`
  - 来源：`anchor_d2h_copy` 与 `checkpoint_d2h_copy`
- `t_persist`
  - 来源：`anchor_persist` 与 `checkpoint_model_persist`
- `t_r`
  - 来源：`replay_step`，`device="cuda"`；按稳态平均输出
- `t_rc`
  - 来源：`replay_step`，`device="cpu"`；按稳态平均输出
- `t_cp`
  - 来源：`shadow_commit`
- `L_disk`
  - 来源：`first_step_latency.program_to_first_step_ms`
- `L_cpu`
  - 来源：`recovery_load.load_cpu_to_gpu_ms + replay cold-start residual`

`tools/visualize_trace.py` 的职责：

- 提供可直接在 Jupyter import 的函数：
  - `load_trace(...)`
  - `summarize_trace(...)`
  - `print_summary(...)`
  - `plot_timeline(...)`
  - `plot_resources(...)`
  - `plot_loss(...)`
  - `plot_interactive(...)`
- 命令行运行时直接 `show()` figure，不负责保存图片
- 命令行运行时会先打印文本 summary，再显示 figure

实现约定：

- timeline 使用 `broken_barh`
- `GPU / Train` 和 `CPU / Shadow` 两个 panel 共享同一条 x 轴
- resources 图把不同量纲拆成多个轴显示
- 资源图显示时会把 memory/storage 类指标从 MB 转成 GB
- 图例不带单位；`zo_shm_used_mb` 显示为 `DRAM`，`gpu0_*` 显示为 `GPU *`
- loss 单独按 `global_step` 画 `loss-step` 曲线，可选叠加 `learning_rate`
- 如果 `--html` 且本地安装了 `plotly`，则生成可缩放、可 hover 的交互图

## 20. 最短调用图

```text
训练开始
└─ LogBasedCheckpointCallback.on_train_begin()
   ├─ _cache_initial_model() / _init_for_resume()
   ├─ _init_shadow_adam_state()
   └─ _start_shadow_process()

每步训练
├─ on_step_begin() -> failure_simulator.check_and_fail() / trigger_failure()
├─ _zo_update_hook() -> 记录 update_history / pending_grad / pending_seed
└─ on_step_end() -> 打印 health / shadow lag

保存 checkpoint
└─ ZOTrainer._save_checkpoint()
   ├─ 写 trainer_state / scheduler / rng_state
   ├─ 写 optimizer.pt + log metadata
   ├─ [full step] super()._save_checkpoint() 或 async_anchor.try_save_full_checkpoint()
   └─ callback.on_save() -> _update_base_and_shadow()

恢复
├─ ZOTrainer._load_from_checkpoint() -> log_based_resume 时跳过默认权重加载
├─ resume_from_log_based()
│  ├─ load_log_based_checkpoint() / _load_base_state()
│  ├─ [_load_shadow_from_manifest()] 软恢复
│  └─ _replay_updates_on_state()
└─ LogBasedCheckpointCallback._init_for_resume()

训练结束
└─ LogBasedCheckpointCallback.on_train_end()
   ├─ async_anchor.shutdown()
   ├─ 停止 shadow
   └─ 清理 manifest / shm / 本地状态
```
