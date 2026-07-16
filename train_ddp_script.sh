#!/usr/bin/env bash

set -euo pipefail

TASK_NAME="${ACT_TASK:-example_task}"
NUM_GPUS="${ACT_NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-12355}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)
            TASK_NAME="$2"
            shift 2
            ;;
        --nproc_per_node|--num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --master-port)
            MASTER_PORT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--task NAME] [--nproc_per_node N] [--master-port PORT]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
    esac
done

export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export CUDNN_BENCHMARK="${CUDNN_BENCHMARK:-1}"

echo "Launching Smooth ACT training: task=${TASK_NAME}, processes=${NUM_GPUS}"

torchrun \
    --standalone \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    train_ddp.py \
    --task "${TASK_NAME}"
