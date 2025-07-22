export CUDA_VISIBLE_DEVICES=0,1,2,3
#export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1


#export WANDB_API_KEY="4c3245840727bab2a2d846451f3bafbf7fa08e1e"
export SWANLAB_API_KEY="bkHIjbNcQgol4Cldsp8IT"
# bkHIjbNcQgol4Cldsp8IT
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
math=/home/lvshangke/dataset/deepscaler_noprompt/math.parquet
olympiad=/home/lvshangke/dataset/deepscaler_noprompt/olympiad_bench.parquet
amc=/home/lvshangke/dataset/deepscaler_noprompt/amc.parquet
aime=/home/lvshangke/dataset/deepscaler_noprompt/aime.parquet

train_files="['$gsm8k_train_path', '$math_train_path']"
#test_files="["/home/lvshangke/dataset/deepscaler_noprompt/math.parquet","/home/lvshangke/dataset/deepscaler_noprompt/olympiad_bench.parquet","/home/lvshangke/dataset/deepscaler_noprompt/amc.parquet","/home/lvshangke/dataset/deepscaler_noprompt/aime.parquet"]"
test_files="['$math', '$olympiad', '$amc', '$aime']"

#ray stop 
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="/home/lvshangke/dataset/deepscaler_noprompt/deepscaler_train.parquet" \
    data.val_files="/home/lvshangke/dataset/deepscaler_noprompt/math.parquet" \
    data.train_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/home/shenyl/hf/model/agentica-org/DeepScaleR-1.5B-Preview  \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='['console','swanlab']' \
    trainer.project_name='verl_grpo_cosreward_new' \
    trainer.experiment_name='deepscaler_cos_only_penalty_0_28' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=60 \
    trainer.test_freq=20 \
    trainer.total_epochs=1 $@
#/home/lvshangke/verl-main/checkpoints/verl_grpo_cosreward/deepscaler_lc_truly_penalty_length_control
# verl-main/checkpoints/verl_grpo_cosreward/deepscaler_lc_1_repub_penalty_0_agg