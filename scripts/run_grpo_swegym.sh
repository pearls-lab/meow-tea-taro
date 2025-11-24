# BEFORE RUNNING THIS SCRIPT:
# git clone git@github.com:ruiyiw/SWE-agent.git
# cd SWE-agent
# pip install -e .

set -x
export HYDRA_FULL_ERROR=1

# DATA/TASK CONFIG
env_name="swegym"
hf_data_repo=$HF_DATA_REPO
hf_train_data_dir=$HF_TRAIN_DATA_DIR
hf_test_data_dir=$HF_TEST_DATA_DIR
local_parquet_dir="local/train_parquet"
train_parquet=$local_parquet_dir/$hf_train_data_dir/$TRAIN_PARQUET_FILE
test_parquet=$local_parquet_dir/$hf_test_data_dir/$TEST_PARQUET_FILE
reward_method="dense"

# MODEL CONFIG
hf_actor_repo_id=""
hf_actor_model_path=""
actor_model_path=local/model/actor
base_model=$BASE_MODEL

# AGENTIC CONFIG
# env_name=... # from above
is_multiturn=True
is_async=True
max_iter=$MAX_ITER
reward_density=$reward_method
reward_type="verified"
reward_manager="agentic_heuristics"
rollout_name="vllm"
rollout_mode="async"

# ALGORITHM CONFIG
adv_estimator=grpo
rollout_n=$ROLLOUT_N

use_kl_loss=True                                                                                                                                                                                                                                                                                    # Whether to use KL loss in objective. True for GRPO.
use_kl_in_reward=False # Whether to use KL divergence in reward calculation.

# TRAINING CONFIG
rollout_temp=$TEMP
val_rollout_temp=$TEMP
train_batch_size=16
ppo_mini_batch_size=16
max_num_batched_tokens=16384
gpu_memory_utilization=$GPU_MEMORY_UTILIZATION
num_workers=$NUM_WORKERS
max_prompt_length=6144
max_response_length=6144
actor_lr=1e-6
nnodes=1
num_epochs=$NUM_EPOCHS
save_freq=$SAVE_FREQ # per steps
test_freq=$TEST_FREQ # per steps

# PROJECT CONFIG
project_name=$PROJECT_NAME # TODO (optional). WandB project name.
experiment_name=$EXPERIMENT_NAME # TODO (optional). WandB experiment name.
save_hf_repo_id=$SAVE_HF_REPO_ID # TODO (optional). HF repo id to save the trained model. If empty, do not save.
resume_wandb_logs=True # TODO (optional, default=True). Whether to resume WandB logs if "experiment_name" exists.

# Step 1: Download RL parquet
echo "Downloading multiturn RL data for swe-gym tasks..."
hf download $hf_data_repo --include="${hf_train_data_dir}/*" --local-dir="$local_parquet_dir" --repo-type dataset
hf download $hf_data_repo --include="${hf_test_data_dir}/*" --local-dir="$local_parquet_dir" --repo-type dataset
hf download $hf_data_repo --include="swegym/sweagent_config.yaml" --local-dir="local/" --repo-type dataset
mv local/swegym/sweagent_config.yaml local/sweagent_config.yaml

# Step 2: Load models
echo "Loading models..."
# Check if actor model is specified
if [ -n "$hf_actor_repo_id" ]; then
    # If specified, download from HF path if available
    if [ -z "$hf_actor_model_path" ]; then
        # Download entire repo if path is empty/None
        hf download $hf_actor_repo_id --local-dir $actor_model_path
    else
        # Download specific path and flatten
        hf download $hf_actor_repo_id --include="${hf_actor_model_path}/*" --local-dir $actor_model_path
        mv $actor_model_path/$hf_actor_model_path/* $actor_model_path/
        rm -rf $actor_model_path/$hf_actor_model_path
        rm -rf $actor_model_path/.cache
    fi
else
    # Otherwise, use base model (from HF)
    actor_model_path=$base_model
fi

# Check if critic model is specified
if [ -n "$hf_critic_repo_id" ]; then
    # If specified, download from HF path if available
    if [ -z "$hf_critic_model_path" ]; then
        # Download entire repo if path is empty/None
        hf download $hf_critic_repo_id --local-dir $critic_model_path
    else
        # Download specific path and flatten
        hf download $hf_critic_repo_id --include="${hf_critic_model_path}/*" --local-dir $critic_model_path
        mv $critic_model_path/$hf_critic_model_path/* $critic_model_path/
        rm -rf $critic_model_path/$hf_critic_model_path
        rm -rf $critic_model_path/.cache
    fi
else
    # Otherwise, use base model (from HF)
    critic_model_path=$base_model
fi

# Step 3: Run training
echo "Starting RL training..."

python3 -m meow_tea_train.verl.trainer.main_ppo \
    data.train_files=$train_parquet \
    data.val_files=$test_parquet \
    data.return_raw_chat=True \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.train_batch_size=$train_batch_size \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    agentic.environment.name=$env_name \
    agentic.environment.is_multiturn=$is_multiturn \
    agentic.environment.is_async=$is_async \
    agentic.environment.max_iter=$max_iter \
    agentic.reward.density=$reward_density \
    agentic.reward.type=$reward_type \
    agentic.agent_loop.type="async_software" \
    +agentic.agent_loop.kwargs.trajs_save_dir="local/trajectories" \
    +agentic.agent_loop.kwargs.sweagent_config_path="local/sweagent_config.yaml" \
    actor_rollout_ref.model.path=$actor_model_path \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.use_torch_compile=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.rollout.name=$rollout_name \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    +actor_rollout_ref.rollout.agentic='${agentic}' \
    actor_rollout_ref.rollout.agent.num_workers=$num_workers \
    actor_rollout_ref.rollout.agent.default_agent_loop="swe_agent" \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="agent_loop_configs.yaml" \
    actor_rollout_ref.rollout.temperature=$rollout_temp \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.n=$rollout_n \
    actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
    actor_rollout_ref.rollout.val_kwargs.temperature=$val_rollout_temp \
    reward_model.reward_manager=$reward_manager \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.validation_data_dir="local/val_results" \
    trainer.nnodes=$nnodes \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=True \
    trainer.hf_kwargs.save_hf_repo_id=$save_hf_repo_id \
    trainer.hf_kwargs.resume_wandb_logs=$resume_wandb_logs \
    trainer.resume_mode=auto \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.total_epochs=$num_epochs $@