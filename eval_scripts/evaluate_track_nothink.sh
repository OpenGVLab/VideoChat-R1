
export CUDA_VISIBLE_DEVICES=0

# MODEL_BASE=mllm/Qwen2.5-VL-7B-Instruct
MODEL_BASE=your_base_dir/checkpoints/run_sft_video_track.sh_20250331_130601
srun -p videop1 \
    --job-name=eval \
    -n8 \
    --ntasks-per-node=8 \
    --cpus-per-task=8 \
python evaluate_track_nothink_c8.py \
     --model_base $MODEL_BASE


