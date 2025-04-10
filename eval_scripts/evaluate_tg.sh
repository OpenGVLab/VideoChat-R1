
export CUDA_VISIBLE_DEVICES=0



MODEL_BASE=/mnt/petrelfs/share_data/yanziang/r1_ckpts/gqa_sft 
srun -p videop1 \
    --job-name=eval \
    -n8 \
    --ntasks-per-node=8 \
    --cpus-per-task=8 \
python evaluate_tg_c8.py \
     --model_base $MODEL_BASE

MODEL_BASE=/mnt/petrelfs/share_data/lixinhao/r1_ckpts/grpo_video_gqa_nothink.sh_20250326_103520_checkpoint-420

srun -p videop1 \
    --job-name=eval \
    -n8 \
    --ntasks-per-node=8 \
    --cpus-per-task=8 \
python evaluate_tg_c8.py \
     --model_base $MODEL_BASE \
     --dataset charades
