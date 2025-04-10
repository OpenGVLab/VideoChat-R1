
export CUDA_VISIBLE_DEVICES=0


MODEL_BASE=your_base_dir/checkpoints/run_sft_video_cls_qa_4_16.sh_20250401_170801_4shot

srun -p videop1 \
    --job-name=eval \
    -n8 \
    --ntasks-per-node=8 \
    --cpus-per-task=8 \
python evaluate_cls_quality_c8.py \
     --model_base $MODEL_BASE




MODEL_BASE=your_base_dir/checkpoints/run_sft_video_cls_qa_4_16.sh_20250401_171124_16shot

srun -p videop1 \
    --job-name=eval \
    -n8 \
    --ntasks-per-node=8 \
    --cpus-per-task=8 \
python evaluate_cls_quality_c8.py \
     --model_base $MODEL_BASE

     