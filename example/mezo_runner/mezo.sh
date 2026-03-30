export WANDB_PROJECT=${WANDB_PROJECT:-NonStopZO2_New}

# GPU 选择: 只使用指定的 GPU (默认使用 GPU 0)
# 可以通过环境变量覆盖: GPU_ID=1 bash mezo.sh
export CUDA_VISIBLE_DEVICES=${GPU_ID:-0}
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

MODEL=${MODEL:-facebook/opt-1.3b}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"

BS=${BS:-16}
LR=${LR:-1e-5}
EPS=${EPS:-1e-3}
SEED=${SEED:-0}
MAX_LENGTH=${MAX_LENGTH:-2048}
# ZO 优化器选择: "mezo-sgd" (默认) 或 "mezo-adam"
ZO_METHOD=${ZO_METHOD:-mezo-sgd}
TRAIN=${TRAIN:-1000}
DEV=${DEV-500}
EVAL=${EVAL:-1000}
STEPS=${STEPS:-20000}
NUM_EPOCHS=${NUM_EPOCHS:-""}
EVAL_STRATEGY=${EVAL_STRATEGY:-steps}
EVAL_STEPS=${EVAL_STEPS:-4000}
LOGGING_STEPS=${LOGGING_STEPS:-10}

MODE=${MODE:-ft}
SAVE_STEPS=${SAVE_STEPS:-20000}

# 实验层级配置:
# L0 Baseline (默认): LOG_BASED_CKPT=-1，按 SAVE_STEPS 固定频率保存完整 checkpoint
# L1 Log-Based Checkpoint:
#   LOG_BASED_CKPT=0: Log-based (accumulate all updates from base)
#   LOG_BASED_CKPT=1: Full + Log with anchor every step
#   LOG_BASED_CKPT=N (N>=2): Full + Log with anchor every N steps
# L2 CPU Shadow (ENABLE_SHADOW=1, 需要 LOG_BASED_CKPT>=0): CPU 端实时维护 shadow model
# L3 即时恢复 (INSTANT_RECOVER=1, 需要 L2+GPU_FAIL_STEP): 故障后立即恢复
# GPU 故障注入 (GPU_FAIL_STEP=N, 旁路): 在第 N 步模拟 GPU 故障，可单独使用
# LOG_BASED_RESUME: 从指定的 full checkpoint 恢复，自动扫描并重放后续的 log checkpoints
LOG_BASED_CKPT=${LOG_BASED_CKPT:--1}
ENABLE_SHADOW=${ENABLE_SHADOW:-0}
INSTANT_RECOVER=${INSTANT_RECOVER:-0}
GPU_FAIL_STEP=${GPU_FAIL_STEP:--1}
# Failure type: "soft" (tmpfs shadow replica survives) or "hard" (DRAM lost, disk recovery)
FAILURE_TYPE=${FAILURE_TYPE:-soft}
# Shadow Pipeline: CPU 端 pipelined z 预生成 + ring buffer, 需要 ENABLE_SHADOW=1
# P 个 producer 线程并行生成 z (释放GIL)，1 个 consumer 串行更新 shadow
SHADOW_PIPELINE=${SHADOW_PIPELINE:-0}
SHADOW_PIPELINE_WORKERS=${SHADOW_PIPELINE_WORKERS:-2}
# Async Anchor: 异步写入 full checkpoint (仅 LOG_BASED_CKPT>=1 时有效)
# GPU→CPU 异步拷贝 + 后台线程写盘
ASYNC_ANCHOR=${ASYNC_ANCHOR:-0}
OUTPUT_LOG=${OUTPUT_LOG:-""}
LOG_BASED_RESUME=${LOG_BASED_RESUME:-""}
LOG_BASED_REPLAY_DEVICE=${LOG_BASED_REPLAY_DEVICE:-cuda}
LOG_BASED_SIMULATE_PERTURBATION=${LOG_BASED_SIMULATE_PERTURBATION:-1}
# 确定性随机数: DETERMINISTIC=1 启用 torch.use_deterministic_algorithms (跨进程/跨GPU可复现)
DETERMINISTIC=${DETERMINISTIC:-0}
# ZO RNG 设备: "native" (用参数所在设备, 快), "cpu" (跨GPU可移植, 慢两百来倍！),
#              "zo_rng" (跨设备 bit-exact, 支持 LOG_BASED_REPLAY_DEVICE=cpu 精确还原 GPU 训练)
ZO_RNG_DEVICE=${ZO_RNG_DEVICE:-native}

# Adam 参数 (仅 ZO_METHOD=mezo-adam 时生效)
ADAM_BETA1=${ADAM_BETA1:-0.9}
ADAM_BETA2=${ADAM_BETA2:-0.999}
ADAM_EPS=${ADAM_EPS:-1e-8}

# 模型精度: "fp16" (默认), "bf16", "fp32"
DTYPE=${DTYPE:-fp16}

TRAIN_NAME=${TRAIN_NAME:-"Test_staging_8"}
RESUME_CKPT=${RESUME_CKPT:-""}
DO_EVAL=${DO_EVAL:-1}
RESET_OUTPUT_DIR=${RESET_OUTPUT_DIR:-1}

# 路径策略: 只手动指定 TRAIN_NAME，其余路径固定派生
OUTPUT_DIR="/tmp/ZO_ckpt/$TRAIN_NAME"
SHM_ROOT="/dev/shm/ZO_ckpt/$TRAIN_NAME"
export ZO_SHM_DIR="$SHM_ROOT"
export ZO_TRACE_PATH="$OUTPUT_DIR/zo_trace.jsonl"

EXTRA_ARGS=""
# 全局规则: 持久化阶段永远不触发 replacement，不设置 save_total_limit

# Log-Based Checkpoint
if [ "$LOG_BASED_CKPT" != "-1" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --log_based_ckpt $LOG_BASED_CKPT"

    # L2: CPU Shadow (必须在 LOG_BASED_CKPT>=0 上)
    if [ "$ENABLE_SHADOW" == "1" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --enable_shadow"

        # L3: 即时恢复 (必须在 L2+GPU故障 上)
        if [ "$INSTANT_RECOVER" == "1" ] && [ "$GPU_FAIL_STEP" != "-1" ]; then
            EXTRA_ARGS="$EXTRA_ARGS --instant_recover"
        fi
    fi

    # Async Anchor (必须在 LOG_BASED_CKPT>=1 上)
    if [ "$ASYNC_ANCHOR" == "1" ] && [ "$LOG_BASED_CKPT" -ge "1" ] 2>/dev/null; then
        EXTRA_ARGS="$EXTRA_ARGS --async_anchor"
        if [ -n "$OUTPUT_LOG" ]; then
            EXTRA_ARGS="$EXTRA_ARGS --log_output_dir ${OUTPUT_LOG}/$TRAIN_NAME-$TASK-${MODEL_NAME}-$TAG"
        fi
    fi
fi

# GPU 故障注入 (旁路，可单独使用或叠加在任意层级)
if [ "$GPU_FAIL_STEP" != "-1" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --gpu_fail_step $GPU_FAIL_STEP"
fi

# Log-Based Resume (优先于 RESUME_CKPT)
if [ -n "$LOG_BASED_RESUME" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --log_based_resume $LOG_BASED_RESUME --log_based_replay_device $LOG_BASED_REPLAY_DEVICE"
    if [ "$LOG_BASED_SIMULATE_PERTURBATION" == "0" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --log_based_simulate_perturbation False"
    fi
    if [ "$LOG_BASED_REPLAY_FP32" == "1" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --log_based_replay_fp32"
    fi
elif [ -n "$RESUME_CKPT" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --resume_from_checkpoint $RESUME_CKPT"
else
    # 非 resume 模式：覆盖已有输出目录，避免 HF Trainer 意外 auto-resume
    EXTRA_ARGS="$EXTRA_ARGS --overwrite_output_dir"
fi

# 确定性随机数控制
if [ "$DETERMINISTIC" == "1" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --deterministic"
fi
if [ "$ZO_RNG_DEVICE" != "native" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --zo_rng_device $ZO_RNG_DEVICE"
fi

# 跳过训练后的评估阶段
if [ "$DO_EVAL" == "0" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --no_eval"
fi
if [ "$MODE" == "prefix" ]; then
    EXTRA_ARGS="--prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act"
elif [ "$MODE" == "lora" ]; then
    EXTRA_ARGS="--lora"
fi
# Adam 参数传递 (仅 mezo-adam 时需要)
if [ "$ZO_METHOD" == "mezo-adam" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --zo_method mezo-adam --adam_beta1 $ADAM_BETA1 --adam_beta2 $ADAM_BETA2 --adam_eps $ADAM_EPS"
fi
TAG=mezo-$MODE-$LR-$EPS-$SEED

# Output directory (used for retry loop and tmpfs shadow/anchor filenames)
RUN_HASH=$(echo -n "$OUTPUT_DIR" | md5sum | head -c 8)

# Wandb resume support (same run across retries)
[ -z "$WANDB_RUN_ID" ] && WANDB_RUN_ID=$(python3 -c "import uuid; print(uuid.uuid4().hex[:8])")
export WANDB_RUN_ID WANDB_RESUME=allow

# Validate reset/resume behavior before any cleanup
if [ "$RESET_OUTPUT_DIR" = "1" ]; then
    if [ -n "$LOG_BASED_RESUME" ] || [ -n "$RESUME_CKPT" ]; then
        echo "ERROR: RESET_OUTPUT_DIR=1 cannot be used with LOG_BASED_RESUME or RESUME_CKPT" >&2
        exit 2
    elif [ "$INSTANT_RECOVER" = "1" ]; then
        echo "INSTANT_RECOVER=1, ignoring RESET_OUTPUT_DIR and keeping OUTPUT_DIR: $OUTPUT_DIR"
    fi
fi

# Clean old tmpfs files and output directory from previous runs
if [ -d "$SHM_ROOT" ]; then
    echo "Cleaning old shm directory: $SHM_ROOT"
    rm -rf "$SHM_ROOT"
fi
mkdir -p "$SHM_ROOT"
if [ "$RESET_OUTPUT_DIR" = "1" ] && [ "$INSTANT_RECOVER" != "1" ]; then
    if [ -d "$OUTPUT_DIR" ]; then
        echo "Cleaning old output directory: $OUTPUT_DIR"
        rm -rf "$OUTPUT_DIR"
    fi
fi

TASK_ARGS=""
case $TASK in
    # For Copa, ReCoRD, SQuAD, DROP, we set --train_as_classification False; for others, set this flag to True
    CB) # It has <1000 training examples. Only use 100 for dev
        DEV=100
        ;;
    Copa) # It has <1000 training examples. Only use 100 for dev
        DEV=100
        TASK_ARGS="--train_as_classification False"
        ;;
    ReCoRD) 
        TASK_ARGS="--train_as_classification False"
        ;;
    DROP) 
        TASK_ARGS="--train_as_classification False"
        ;;
    SQuAD)
        TASK_ARGS="--train_as_classification False"
        ;;
esac

echo "========== Configuration =========="
echo "TAG: $TAG"
echo "BS: $BS, LR: $LR, EPS: $EPS, SEED: $SEED"
echo "MAX_LENGTH: $MAX_LENGTH"
echo "STEPS: $STEPS, NUM_EPOCHS: ${NUM_EPOCHS:-<unset>}, EVAL_STRATEGY: $EVAL_STRATEGY, EVAL_STEPS: $EVAL_STEPS, SAVE_STEPS: $SAVE_STEPS, LOGGING_STEPS: $LOGGING_STEPS"
echo "MODE: $MODE, DTYPE: $DTYPE, DO_EVAL: $DO_EVAL, ZO_METHOD: $ZO_METHOD"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "ZO_TRACE_PATH: $ZO_TRACE_PATH"
echo "ZO_SHM_DIR: $ZO_SHM_DIR"
echo "--- Log-Based Checkpoint ---"
echo "LOG_BASED_CKPT: $LOG_BASED_CKPT (-1=disabled, 0=log-based, N>=1=full+log)"
echo "ENABLE_SHADOW: $ENABLE_SHADOW"
echo "INSTANT_RECOVER: $INSTANT_RECOVER"
echo "FAILURE_TYPE: $FAILURE_TYPE"
echo "ASYNC_ANCHOR: $ASYNC_ANCHOR"
if [ -n "$OUTPUT_LOG" ]; then
    echo "OUTPUT_LOG: $OUTPUT_LOG"
fi
echo "GPU_FAIL_STEP: $GPU_FAIL_STEP"
if [ -n "$LOG_BASED_RESUME" ]; then
    echo "LOG_BASED_RESUME: $LOG_BASED_RESUME"
    echo "LOG_BASED_REPLAY_DEVICE: $LOG_BASED_REPLAY_DEVICE"
    echo "LOG_BASED_SIMULATE_PERTURBATION: $LOG_BASED_SIMULATE_PERTURBATION (0=skip ~4x faster, 1=bitwise exact)"
fi
echo "Extra args: $EXTRA_ARGS $TASK_ARGS"
echo "===================================="

ORIG_EXTRA_ARGS="$EXTRA_ARGS"

_run_training() {
    local num_dev_args=()
    local num_epoch_args=()
    if [ -n "$DEV" ]; then
        num_dev_args=(--num_dev "$DEV")
    fi
    if [ -n "$NUM_EPOCHS" ]; then
        num_epoch_args=(--num_train_epochs "$NUM_EPOCHS")
    fi
    python /home/users/u0001609/NonStopZO2/example/mezo_runner/run.py \
        --model_name $MODEL \
        --task_name $TASK \
        --output_dir $OUTPUT_DIR \
        --run_name $TRAIN_NAME-$TASK-${MODEL_NAME}-$TAG \
        --tag $TAG \
        --train_set_seed $SEED \
        --num_train $TRAIN \
        "${num_dev_args[@]}" \
        --num_eval $EVAL \
        --logging_steps $LOGGING_STEPS \
        --max_steps $STEPS \
        "${num_epoch_args[@]}" \
        --trainer zo \
        $([ "$DTYPE" == "fp16" ] && echo "--load_float16" || [ "$DTYPE" == "bf16" ] && echo "--load_bfloat16") \
        --max_length $MAX_LENGTH \
        --learning_rate $LR \
        --zo_eps $EPS \
        --per_device_train_batch_size $BS \
        --lr_scheduler_type "constant" \
        --eval_strategy $EVAL_STRATEGY \
        --save_strategy steps \
        --eval_steps $EVAL_STEPS \
        --save_steps $SAVE_STEPS \
        --train_as_classification \
        --report_to wandb \
        $EXTRA_ARGS \
        $TASK_ARGS \
        "$@"
}

_remaining_fail_steps_after_first() {
    local spec="$1"
    if [ -z "$spec" ] || [ "$spec" = "-1" ]; then
        echo "-1"
        return
    fi

    local IFS=','
    read -r -a parts <<< "$spec"
    local remaining=()
    local idx=0
    for part in "${parts[@]}"; do
        part="${part//[[:space:]]/}"
        [ -z "$part" ] && continue
        if [ $idx -eq 0 ]; then
            idx=$((idx + 1))
            continue
        fi
        remaining+=("$part")
        idx=$((idx + 1))
    done

    if [ ${#remaining[@]} -eq 0 ]; then
        echo "-1"
    else
        local joined
        local IFS=','
        joined="${remaining[*]}"
        echo "$joined"
    fi
}

# First run
_run_training "$@"; EXIT_CODE=$?

# Retry loop: auto-resume after SIGKILL (exit code 137)
while [ $EXIT_CODE -eq 137 ] && [ "$INSTANT_RECOVER" == "1" ]; do
    echo "===== SIGKILL detected (exit $EXIT_CODE), auto-resuming ($FAILURE_TYPE) ====="
    LATEST=$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -1)
    [ -z "$LATEST" ] && { echo "No checkpoint found in $OUTPUT_DIR"; exit 137; }
    echo "Latest checkpoint: $LATEST"

    NEXT_GPU_FAIL_STEP=$(_remaining_fail_steps_after_first "$GPU_FAIL_STEP")
    export GPU_FAIL_STEP="$NEXT_GPU_FAIL_STEP"
    echo "Remaining GPU_FAIL_STEP schedule: $GPU_FAIL_STEP"

    # Build retry args: remove overwrite_output_dir and gpu_fail_step from original
    RETRY_ARGS=$(echo "$ORIG_EXTRA_ARGS" | sed 's/--overwrite_output_dir//g; s/--gpu_fail_step [^ ]*//g')
    RETRY_ARGS="$RETRY_ARGS --gpu_fail_step $GPU_FAIL_STEP --log_based_resume $LATEST --log_based_replay_device cuda"

    SHADOW_REPLICA="$SHM_ROOT/zo_shadow_latest_${RUN_HASH}.safetensors"
    SHADOW_FLAT_HEADER="$SHM_ROOT/zo_shadow_latest_${RUN_HASH}.flat.header.json"
    SHADOW_FLAT_BUF="$SHM_ROOT/zo_shadow_latest_${RUN_HASH}.flat.bin"
    SHADOW_FLAT_ADAM_M="$SHM_ROOT/zo_shadow_latest_${RUN_HASH}.flat.adam_m.bin"
    SHADOW_FLAT_ADAM_V="$SHM_ROOT/zo_shadow_latest_${RUN_HASH}.flat.adam_v.bin"
    ANCHOR_LATEST="$SHM_ROOT/zo_anchor_latest_${RUN_HASH}.safetensors"
    REBASE_PAYLOAD_DIR="$SHM_ROOT/zo_rebase_payload_${RUN_HASH}"
    if [ "$FAILURE_TYPE" == "hard" ]; then
        # Simulate DRAM loss: delete shadow/anchor latest replicas in tmpfs
        rm -f \
            "$SHADOW_REPLICA" "$SHADOW_FLAT_HEADER" "$SHADOW_FLAT_BUF" \
            "$SHADOW_FLAT_ADAM_M" "$SHADOW_FLAT_ADAM_V" \
            "$ANCHOR_LATEST"
        rm -rf "$REBASE_PAYLOAD_DIR"
    elif [ "$SHADOW_FLAT_COMMIT" == "1" ] && [ -f "$SHADOW_FLAT_HEADER" ]; then
        RETRY_ARGS="$RETRY_ARGS --shadow_resume $SHADOW_FLAT_HEADER"
    elif [ -f "$SHADOW_REPLICA" ]; then
        # Soft failure: shadow replica survives in tmpfs
        RETRY_ARGS="$RETRY_ARGS --shadow_resume $SHADOW_REPLICA"
    fi

    EXTRA_ARGS="$RETRY_ARGS"
    _run_training "$@"; EXIT_CODE=$?
done
exit $EXIT_CODE
