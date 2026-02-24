export WANDB_PROJECT=${WANDB_PROJECT:-NonStopZO2}

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
TRAIN=${TRAIN:-1000}
DEV=${DEV:-500}
EVAL=${EVAL:-1000}
STEPS=${STEPS:-20000}
EVAL_STEPS=${EVAL_STEPS:-4000}

MODE=${MODE:-ft}
SAVE_STEPS=${SAVE_STEPS:-20000}

# 实验层级配置:
# L0 Baseline (默认): BATCHDIFF_CKPT=-1，按 SAVE_STEPS 固定频率保存完整 checkpoint
# L1 Batch Differential Checkpoint:
#   BATCHDIFF_CKPT=0: Incremental (accumulate all updates from base)
#   BATCHDIFF_CKPT=1: Pure Differential (only current step's update)
#   BATCHDIFF_CKPT=N (N>=2): Batch Differential (new full checkpoint every N steps)
# L2 CPU Shadow (ENABLE_SHADOW=1, 需要 BATCHDIFF_CKPT>=0): CPU 端实时维护 shadow model
# L3 即时恢复 (INSTANT_RECOVER=1, 需要 L2+GPU_FAIL_STEP): 故障后立即恢复
# GPU 故障注入 (GPU_FAIL_STEP=N, 旁路): 在第 N 步模拟 GPU 故障，可单独使用
# BATCHDIFF_RESUME: 从指定的 full checkpoint 恢复，自动扫描并重放后续的 differential checkpoints
BATCHDIFF_CKPT=${BATCHDIFF_CKPT:--1}
ENABLE_SHADOW=${ENABLE_SHADOW:-0}
INSTANT_RECOVER=${INSTANT_RECOVER:-0}
GPU_FAIL_STEP=${GPU_FAIL_STEP:--1}
BATCHDIFF_RESUME=${BATCHDIFF_RESUME:-""}
BATCHDIFF_REPLAY_DEVICE=${BATCHDIFF_REPLAY_DEVICE:-cpu}

TRAIN_NAME=${TRAIN_NAME:-"Amz/Test"}
RESUME_CKPT=${RESUME_CKPT:-""}
DO_EVAL=${DO_EVAL:-1}

EXTRA_ARGS=""
# 全局规则: 持久化阶段永远不触发 replacement，不设置 save_total_limit

# Batch Differential Checkpoint
if [ "$BATCHDIFF_CKPT" != "-1" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --batchdiff_ckpt $BATCHDIFF_CKPT"

    # L2: CPU Shadow (必须在 BATCHDIFF_CKPT>=0 上)
    if [ "$ENABLE_SHADOW" == "1" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --enable_shadow"

        # L3: 即时恢复 (必须在 L2+GPU故障 上)
        if [ "$INSTANT_RECOVER" == "1" ] && [ "$GPU_FAIL_STEP" != "-1" ]; then
            EXTRA_ARGS="$EXTRA_ARGS --instant_recover"
        fi
    fi
fi

# GPU 故障注入 (旁路，可单独使用或叠加在任意层级)
if [ "$GPU_FAIL_STEP" != "-1" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --gpu_fail_step $GPU_FAIL_STEP"
fi

# Batch Diff Resume (优先于 RESUME_CKPT)
if [ -n "$BATCHDIFF_RESUME" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --batchdiff_resume $BATCHDIFF_RESUME --batchdiff_replay_device $BATCHDIFF_REPLAY_DEVICE"
elif [ -n "$RESUME_CKPT" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --resume_from_checkpoint $RESUME_CKPT"
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
TAG=mezo-$MODE-$LR-$EPS-$SEED

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
echo "STEPS: $STEPS, EVAL_STEPS: $EVAL_STEPS, SAVE_STEPS: $SAVE_STEPS"
echo "MODE: $MODE, DO_EVAL: $DO_EVAL"
echo "--- Batch Differential Checkpoint ---"
echo "BATCHDIFF_CKPT: $BATCHDIFF_CKPT (-1=disabled, 0=incremental, 1=pure diff, N>=2=batch diff)"
echo "ENABLE_SHADOW: $ENABLE_SHADOW"
echo "INSTANT_RECOVER: $INSTANT_RECOVER"
echo "GPU_FAIL_STEP: $GPU_FAIL_STEP"
echo "BATCHDIFF_RESUME: $BATCHDIFF_RESUME"
echo "BATCHDIFF_REPLAY_DEVICE: $BATCHDIFF_REPLAY_DEVICE"
echo "Extra args: $EXTRA_ARGS $TASK_ARGS"
echo "===================================="

python /home/ubuntu/NonStopZO2/example/mezo_runner/run.py \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir /home/ubuntu/ZO_ckpt/$TRAIN_NAME-$TASK-${MODEL_NAME}-$TAG \
    --run_name $TRAIN_NAME-$TASK-${MODEL_NAME}-$TAG \
    --tag $TAG \
    --train_set_seed $SEED \
    --num_train $TRAIN \
    --num_dev $DEV \
    --num_eval $EVAL \
    --logging_steps 10 \
    --max_steps $STEPS \
    --trainer zo \
    --load_float16 \
    --learning_rate $LR \
    --zo_eps $EPS \
    --per_device_train_batch_size $BS \
    --lr_scheduler_type "constant" \
    --eval_strategy steps \
    --save_strategy steps \
    --eval_steps $EVAL_STEPS \
    --save_steps $SAVE_STEPS \
    --train_as_classification \
    --report_to wandb \
    $EXTRA_ARGS \
    $TASK_ARGS \
    "$@"
