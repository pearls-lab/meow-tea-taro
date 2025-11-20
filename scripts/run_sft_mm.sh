#!/bin/bash
set -x
export PATH="/usr/bin:$(echo $PATH | tr ':' '\n' | grep -v conda | tr '\n' ':' | sed 's/:$//')"
export HYDRA_FULL_ERROR=1

# DATA/TASK CONFIG
hf_data_repo=$HF_DATA_REPO
hf_train_data_dir=$HF_TRAIN_DATA_DIR
local_parquet_dir="local/train_parquet"

# MODEL AND TRAINING CONFIG
base_model=$BASE_MODEL
train_batch_size=$TRAIN_BATCH_SIZE
micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU
max_length=$MAX_LENGTH
nproc_per_node=$NPROC_PER_NODE
lr=$LR
save_freq=$SAVE_FREQ # per steps
test_freq=-1 # per steps
total_epochs=$TOTAL_EPOCHS

# PROJECT CONFIG
project_name=$PROJECT_NAME # TODO (optional). WandB project name.
experiment_name=$EXPERIMENT_NAME # TODO (optional). WandB experiment name.
save_path=checkpoints # The local path to save checkpoints.
save_hf_repo_id=$SAVE_HF_REPO_ID # TODO (optional). HF repo id to save the trained model. If empty, do not save.


# Step 1: Download SFT data
hf download $hf_data_repo $hf_train_data_dir --local-dir $local_parquet_dir --repo-type dataset

torchrun --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$local_parquet_dir/$hf_train_data_dir \
    data.val_files=$local_parquet_dir/$hf_train_data_dir \
    data.train_batch_size=$train_batch_size \
    data.micro_batch_size_per_gpu=$micro_batch_size_per_gpu \
    +data.chat_based.enable=true \
    +data.chat_based.messages_key=messages \
    data.max_length=$max_length \
    model.partial_pretrain=$base_model \
    optim.lr=$lr \
    trainer.default_local_dir=$save_path \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    +trainer.save_hf_repo_id=$save_hf_repo_id \
    trainer.total_epochs=$total_epochs $@ \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=false