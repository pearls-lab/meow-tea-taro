#!/bin/bash

# Script to prepare multi-turn training data for meow_tea_train
# Usage: bash scripts/prepare_ppo_data.sh

ENV_NAME="textworld" # Options: alfworld, textworld
TASKS_DIR="/home/rwang/llm-search-textgame/data/games/tw_simple_balanced"
TASK_PREFIX="tw_balanced" # Prefix for task files
OUT_DIR="data/tw_simple_balanced"
mkdir -p $OUT_DIR

# python -m meow_tea_experiments.data_generation.generate_multiturn_data \
#     --env_name $ENV_NAME \
#     --instance_dir $TASKS_DIR \
#     --instance_id_range 50001 55000 \
#     --task_prefix $TASK_PREFIX \
#     --out_dir $OUT_DIR \
#     --train_type ppo \
#     --split train

python -m meow_tea_experiments.data_generation.generate_multiturn_data \
    --env_name $ENV_NAME \
    --instance_dir $TASKS_DIR \
    --instance_id_range 40001 40100 \
    --task_prefix $TASK_PREFIX \
    --out_dir $OUT_DIR \
    --train_type ppo \
    --split validation

# python -m meow_tea_experiments.data_generation.generate_multiturn_data \
#     --env_name $ENV_NAME \
#     --instance_dir $TASKS_DIR \
#     --instance_id_range 1 100 \
#     --task_prefix $TASK_PREFIX \
#     --out_dir $OUT_DIR \
#     --train_type ppo \
#     --split test