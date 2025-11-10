#!/bin/bash

# Launch script for multi-GPU RL training with DeepSpeed
# Usage: bash launch_rl_training.sh [algorithm] [additional_args]
# Example: bash launch_rl_training.sh ppo --learning_rate 5e-7
# Example: bash launch_rl_training.sh grpo --group_size 8

ALGORITHM=${1:-ppo}
shift

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $NUM_GPUS GPUs"

if [ $NUM_GPUS -eq 0 ]; then
    echo "Error: No GPUs detected!"
    exit 1
fi

DS_CONFIG="ds_config.json"

if [ ! -f "$DS_CONFIG" ]; then
    echo "Error: DeepSpeed config file $DS_CONFIG not found!"
    exit 1
fi

if [ "$ALGORITHM" = "ppo" ]; then
    TRAIN_SCRIPT="train_ppo.py"
elif [ "$ALGORITHM" = "grpo" ]; then
    TRAIN_SCRIPT="train_grpo.py"
else
    echo "Error: Unknown algorithm $ALGORITHM. Use 'ppo' or 'grpo'."
    exit 1
fi

echo "Starting $ALGORITHM training on $NUM_GPUS GPUs"
echo "Training script: $TRAIN_SCRIPT"
echo "DeepSpeed config: $DS_CONFIG"
echo "Additional arguments: $@"

deepspeed --num_gpus=$NUM_GPUS $TRAIN_SCRIPT \
    --deepspeed $DS_CONFIG \
    "$@"

