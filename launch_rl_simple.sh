#!/bin/bash

# Simple launch script without DeepSpeed (for debugging or single GPU)
# Usage: bash launch_rl_simple.sh [algorithm] [additional_args]
# Example: bash launch_rl_simple.sh ppo --batch_size 8
# Example: bash launch_rl_simple.sh grpo --group_size 4

ALGORITHM=${1:-ppo}
shift

if [ "$ALGORITHM" = "ppo" ]; then
    TRAIN_SCRIPT="train_ppo.py"
elif [ "$ALGORITHM" = "grpo" ]; then
    TRAIN_SCRIPT="train_grpo.py"
else
    echo "Error: Unknown algorithm $ALGORITHM. Use 'ppo' or 'grpo'."
    exit 1
fi

echo "Starting $ALGORITHM training (without DeepSpeed)"
echo "Training script: $TRAIN_SCRIPT"
echo "Additional arguments: $@"

python $TRAIN_SCRIPT "$@"

