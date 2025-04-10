
export CUDA_VISIBLE_DEVICES=0

# MODEL_BASE=mllm/Qwen2.5-VL-7B-Instruct
MODEL_BASE=your_base_dir/checkpoints/run_grpo_video_qa.sh_20250326_103358/checkpoint-420
srun -p videop1 \
    --job-name=eval \
    -n8 \
    --ntasks-per-node=8 \
    --cpus-per-task=8 \
python evaluate_qa_c8.py \
     --model_base $MODEL_BASE

