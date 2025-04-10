
export CUDA_VISIBLE_DEVICES=0

# MODEL_BASE=mllm/Qwen2.5-VL-7B-Instruct
MODEL_BASE=your_base_dir/checkpoints/run_grpo_video_task.sh_20250402_002323

srun -p videop1 \
    --job-name=eval \
    -n8 \
    --ntasks-per-node=8 \
    --cpus-per-task=8 \
python evaluate_track_c8.py \
     --model_base $MODEL_BASE

srun -p videop1 \
    --job-name=eval \
    -n8 \
    --ntasks-per-node=8 \
    --cpus-per-task=8 \
python evaluate_gqa_c8.py \
     --model_base $MODEL_BASE


srun -p videop1 \
    --job-name=eval \
    -n8 \
    --ntasks-per-node=8 \
    --cpus-per-task=8 \
python evaluate_tg_c8.py \
     --model_base $MODEL_BASE



