# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
# export CUDA_VISIBLE_DEVICES=2 # 替换为您想使用的 GPU ID，例如 0, 1, 或 "0,1"
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
# # 激活 Conda 环境
# source /home/lvshangke/anaconda3/etc/profile.d/conda.sh 
# conda activate verl_env  # 替换为您的 Conda 环境名称




# export WANDB_API_KEY="4c3245840727bab2a2d846451f3bafbf7fa08e1e"
# export HYDRA_FULL_ERROR=1

# #WANDB_MODE="offline"

# cd /home/lvshangke/verl-main
# set -x

# export VLLM_ATTENTION_BACKEND=XFORMERS

# gsm8k_train_path=$HOME/data/gsm8k/train.parquet
# gsm8k_test_path=$HOME/data/gsm8k/test.parquet
# math_train_path=$HOME/data/math/train.parquet
# math_test_path=$HOME/data/math/test.parquet

# checkpoint_dir="/home/lvshangke/output_models/grpo-mathlighteval"

# train_files="['/home/lvshangke/dataset/train-dapo.parquet', '/home/lvshangke/dataset/train-dapo.parquet']" #糟糕 用的一个文件 用的都是math

# test_files="['$gsm8k_test_path', '$math_test_path']"

# python3 -m verl.trainer.main_ppo \
#     algorithm.adv_estimator=grpo \
#     data.train_files="/home/lvshangke/dataset/math-lighteval/train.parquet" \
#     data.val_files="/home/lvshangke/dataset/math-lighteval/test.parquet" \
#     data.train_batch_size=32 \
#     data.max_prompt_length=1024 \
#     data.max_response_length=4096 \
#     data.filter_overlong_prompts=True \
#     data.truncation='error' \
#     actor_rollout_ref.model.path=/home/shenyl/hf/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
#     actor_rollout_ref.actor.optim.lr=1e-6 \
#     actor_rollout_ref.model.use_remove_padding=True \
#     actor_rollout_ref.actor.ppo_mini_batch_size=8 \
#     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
#     actor_rollout_ref.actor.use_kl_loss=False \
#     actor_rollout_ref.actor.kl_loss_coef=0.001 \
#     actor_rollout_ref.actor.kl_loss_type=low_var_kl \
#     actor_rollout_ref.actor.entropy_coeff=0 \
#     actor_rollout_ref.model.enable_gradient_checkpointing=True \
#     actor_rollout_ref.actor.fsdp_config.param_offload=False \
#     actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
#     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
#     actor_rollout_ref.rollout.name=vllm \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
#     actor_rollout_ref.rollout.n=8 \
#     actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
#     actor_rollout_ref.ref.fsdp_config.param_offload=True \
#     algorithm.use_kl_in_reward=False \
#     trainer.critic_warmup=0 \
#     'trainer.logger=['console','wandb']' \
#     trainer.project_name='verl_grpo_cosreward' \
#     trainer.experiment_name='qwen2_1.5b_function_rm' \
#     trainer.default_local_dir=${checkpoint_dir} \
#     trainer.n_gpus_per_node=1 \
#     trainer.nnodes=1 \
#     trainer.save_freq=50 \
#     trainer.test_freq=5 \
#     trainer.total_epochs=1 $@


export CUDA_VISIBLE_DEVICES=2
#export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1


export WANDB_API_KEY="4c3245840727bab2a2d846451f3bafbf7fa08e1e"
export HYDRA_FULL_ERROR=1

#WANDB_MODE="offline"
source /home/lvshangke/anaconda3/etc/profile.d/conda.sh 

conda activate verl_env
cd /home/lvshangke/verl-main
set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet
math_train_path=$HOME/data/math/train.parquet
math_test_path=$HOME/data/math/test.parquet

train_files="['$gsm8k_train_path', '$math_train_path']"
test_files="['$gsm8k_test_path', '$math_test_path']"

#ray stop 
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="/home/lvshangke/dataset/math-lighteval-grpo/train.parquet" \
    data.val_files="/home/lvshangke/dataset/math-lighteval-grpo/test.parquet" \
    data.train_batch_size=16 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/home/shenyl/hf/model/Qwen/Qwen2.5-1.5B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='['console','wandb']' \
    trainer.project_name='verl_grpo_cosreward' \
    trainer.experiment_name='demo' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    trainer.total_epochs=2 $@
