set -x
export HYDRA_FULL_ERROR=1

# DATA/TASK CONFIG
env_name=$ENV_NAME
task_prefix=$TASK_PREFIX
instance_id_start=$INSTANCE_ID_START
instance_id_end=$INSTANCE_ID_END
hf_data_repo=$HF_DATA_REPO
hf_instances_dir=$HF_INSTANCES_DIR
hf_train_data_dir=$HF_TRAIN_DATA_DIR
local_instances_dir="local/$hf_instances_dir"
local_train_data_dir="local/$hf_train_data_dir"
local_parquet_dir="local/train_parquet"
reward_method=$REWARD_METHOD

# MODEL CONFIG
hf_actor_repo_id=$HF_ACTOR_REPO_ID
hf_actor_model_path=$HF_ACTOR_MODEL_PATH
hf_critic_repo_id=$HF_CRITIC_REPO_ID
hf_critic_model_path=$HF_CRITIC_MODEL_PATH
actor_model_path=local/model/actor
critic_model_path=local/model/critic
base_model=$BASE_MODEL

# AGENTIC CONFIG
# env_name=... # from above
is_multiturn=True
is_async=$IS_ASYNC
max_iter=$MAX_ITER
reward_density=$reward_method
reward_type="verified"
reward_manager="agentic_verified"
rollout_name="vllm_agentic"
rollout_mode=$( [ "$is_async" = "True" ] && echo "async" || echo "sync" ) # Set 'async' if is_async=True, else 'sync'.

# ALGORITHM CONFIG
adv_estimator=gae
gamma=$GAMMA

use_kl_loss=False # Whether to use KL loss in objective. True for GRPO.
use_kl_in_reward=$USE_KL_IN_REWARD # Whether to use KL divergence in reward calculation.
kl_coef=$KL_COEF # KL coefficient for KL penalty or KL reward.
clip_ratio=$CLIP_RATIO

# TRAINING CONFIG
rollout_temp=$ROLLOUT_TEMP
val_rollout_temp=$VAL_ROLLOUT_TEMP
train_batch_size=256
ppo_mini_batch_size=256
max_num_batched_tokens=8192
gpu_memory_utilization=$GPU_MEMORY_UTILIZATION
max_prompt_length=$MAX_PROMPT_LENGTH
max_response_length=$MAX_RESPONSE_LENGTH
actor_lr=$ACTOR_LR
critic_lr=$CRITIC_LR
nnodes=1
num_epochs=$NUM_EPOCHS
save_freq=$SAVE_FREQ # per steps
test_freq=$TEST_FREQ # per steps

# PROJECT CONFIG
project_name=$PROJECT_NAME # TODO (optional). WandB project name.
experiment_name=$EXPERIMENT_NAME # TODO (optional). WandB experiment name.
save_hf_repo_id=$SAVE_HF_REPO_ID # TODO (optional). HF repo id to save the trained model. If empty, do not save.
resume_wandb_logs=True # TODO (optional, default=True). Whether to resume WandB logs if "experiment_name" exists.


# Step 1: Process RL data
echo "Processing multiturn RL data for tasks ${env_name}-${task_prefix} ${task_id_start}-${task_id_end}"
python3 -m meow_tea_train.agentic_utils.data_process.rl_data_processor \
    --env_name "$env_name" \
    --task_prefix "$task_prefix" \
    --instance_id_range "$instance_id_start" "$instance_id_end" \
    --hf_data_repo "$hf_data_repo" \
    --hf_instances_dir "$hf_instances_dir" \
    --hf_train_data_dir "$hf_train_data_dir" \
    --local_instances_dir "$local_instances_dir" \
    --local_train_data_dir "$local_train_data_dir" \
    --local_parquet_dir "$local_parquet_dir" \
    --reward_method "$reward_method"

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
    data.train_files="$local_parquet_dir/train.parquet" \
    data.val_files="$local_parquet_dir/validation.parquet" \
    data.return_raw_chat=True \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.train_batch_size=$train_batch_size \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.gamma=$gamma \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    agentic.environment.name=$env_name \
    agentic.environment.is_multiturn=$is_multiturn \
    agentic.environment.is_async=$is_async \
    agentic.environment.max_iter=$max_iter \
    agentic.reward.density=$reward_density \
    agentic.reward.type=$reward_type \
    actor_rollout_ref.model.path=$actor_model_path \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.use_torch_compile=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.clip_ratio=$clip_ratio \
    actor_rollout_ref.rollout.name=$rollout_name \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    +actor_rollout_ref.rollout.agentic='${agentic}' \
    actor_rollout_ref.rollout.temperature=$rollout_temp \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
    actor_rollout_ref.rollout.val_kwargs.temperature=$val_rollout_temp \
    critic.optim.lr=$critic_lr \
    critic.model.path=$critic_model_path \
    critic.model.use_remove_padding=True \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=32 \
    critic.use_dynamic_bsz=True \
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