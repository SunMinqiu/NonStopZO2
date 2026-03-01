#!/bin/bash
#SBATCH --job-name=mezo_train
#SBATCH -p ai-h100l
#SBATCH -N 1
#SBATCH -t 4:00:00
#SBATCH -A rccs-hpbdrt
#SBATCH -o ZO_logs/%j.log
#SBATCH -e ZO_logs/%j.log

# 创建日志目录
mkdir -p ZO_logs

# 激活 uv 环境
source /home/users/u0001609/.miao/bin/activate


# 设置项目根目录
PROJECT_ROOT=/home/users/u0001609/NonStopZO2
OUTPUT_ROOT=/lvs0/rccs-hpbdrt/minqiu/ZO_ckpt

# ========== 实验配置 ==========
export MODEL=${MODEL:-Qwen/Qwen3-1.7B}
export TASK=${TASK:-SST2}
export MODE=${MODE:-ft}
export LR=${LR:-1e-7}
export EPS=${EPS:-1e-3}
export STEPS=${STEPS:-1000}
export EVAL_STEPS=${EVAL_STEPS:-100}
export SAVE_STEPS=${SAVE_STEPS:-1}
export DO_EVAL=${DO_EVAL:-0}
export BATCHDIFF_CKPT=${BATCHDIFF_CKPT:-0}

# 其他默认参数
export WANDB_PROJECT=${WANDB_PROJECT:-NonStopZO2}
export BS=${BS:-16}
export SEED=${SEED:-0}
export TRAIN=${TRAIN:-1000}
export DEV=${DEV:-500}
export EVAL=${EVAL:-1000}
export ENABLE_SHADOW=${ENABLE_SHADOW:-0}
export INSTANT_RECOVER=${INSTANT_RECOVER:-0}
export GPU_FAIL_STEP=${GPU_FAIL_STEP:--1}
export BATCHDIFF_RESUME=${BATCHDIFF_RESUME:-""}
export BATCHDIFF_REPLAY_DEVICE=${BATCHDIFF_REPLAY_DEVICE:-cpu}
export BATCHDIFF_SIMULATE_PERTURBATION=${BATCHDIFF_SIMULATE_PERTURBATION:-1}
export RESUME_CKPT=${RESUME_CKPT:-""}

# GPU 设置 (SLURM 已分配，不需要手动设置 CUDA_VISIBLE_DEVICES)
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# 解析模型名
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"

# 构建 EXTRA_ARGS
EXTRA_ARGS=""

if [ "$BATCHDIFF_CKPT" != "-1" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --batchdiff_ckpt $BATCHDIFF_CKPT"
    if [ "$ENABLE_SHADOW" == "1" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --enable_shadow"
        if [ "$INSTANT_RECOVER" == "1" ] && [ "$GPU_FAIL_STEP" != "-1" ]; then
            EXTRA_ARGS="$EXTRA_ARGS --instant_recover"
        fi
    fi
fi

if [ "$GPU_FAIL_STEP" != "-1" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --gpu_fail_step $GPU_FAIL_STEP"
fi

if [ -n "$BATCHDIFF_RESUME" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --batchdiff_resume $BATCHDIFF_RESUME --batchdiff_replay_device $BATCHDIFF_REPLAY_DEVICE"
    if [ "$BATCHDIFF_SIMULATE_PERTURBATION" == "0" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --batchdiff_simulate_perturbation False"
    fi
elif [ -n "$RESUME_CKPT" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --resume_from_checkpoint $RESUME_CKPT"
fi

if [ "$DO_EVAL" == "0" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --no_eval"
fi

if [ "$MODE" == "prefix" ]; then
    EXTRA_ARGS="--prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act"
elif [ "$MODE" == "lora" ]; then
    EXTRA_ARGS="--lora"
fi

TAG=mezo-$MODE-$LR-$EPS-$SEED

# TASK 特殊参数
TASK_ARGS=""
case $TASK in
    CB|Copa) DEV=100 ;;
esac
case $TASK in
    Copa|ReCoRD|DROP|SQuAD) TASK_ARGS="--train_as_classification False" ;;
esac

echo "========== Configuration =========="
echo "MODEL: $MODEL"
echo "TASK: $TASK"
echo "TAG: $TAG"
echo "STEPS: $STEPS, EVAL_STEPS: $EVAL_STEPS, SAVE_STEPS: $SAVE_STEPS"
echo "EXTRA_ARGS: $EXTRA_ARGS $TASK_ARGS"
echo "===================================="

# 运行训练
python ${PROJECT_ROOT}/example/mezo_runner/run.py \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir ${OUTPUT_ROOT}/${SLURM_JOB_ID}-$TASK-${MODEL_NAME}-$TAG \
    --run_name ${SLURM_JOB_ID}-$TASK-${MODEL_NAME}-$TAG \
    --tag $TAG \
    --seed $SEED \
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
