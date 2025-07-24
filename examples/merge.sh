export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0 # 替换为您想使用的 GPU ID，例如 0, 1, 或 "0,1"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
# 激活 Conda 环境
source /home/lvshangke/anaconda3/etc/profile.d/conda.sh 
conda activate verl_env  # 替换为您的 Conda 环境名称

cd /home/lvshangke/verl-main

python3 /home/lvshangke/verl-main/scripts/model_merger.py \
    --backend fsdp \
    --is-value-model  \
    --hf_model_path /home/shenyl/hf/model/agentica-org/DeepScaleR-1.5B-Preview \
    --local_dir /home/lvshangke/verl-main/checkpoints/verl_grpo_cosreward/deepscaler_lc_4_sample_penalty_0_28/global_step_540/actor \
    --target_dir /home/lvshangke/output_models/deepscaler_lc_4_028_540