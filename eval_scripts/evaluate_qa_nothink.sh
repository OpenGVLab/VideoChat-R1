
export CUDA_VISIBLE_DEVICES=0

# MODEL_BASE=mllm/Qwen2.5-VL-7B-Instruct
MODEL_BASE=/mnt/petrelfs/share_data/lixinhao/r1_ckpts/grpo_video_qa_nothink.sh_20250326_231548
srun -p videop1 \
    --job-name=eval \
    -n8 \
    --ntasks-per-node=8 \
    --cpus-per-task=8 \
python evaluate_gqa_nothink_c8.py \
     --model_base $MODEL_BASE


