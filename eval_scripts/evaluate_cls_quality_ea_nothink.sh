
export CUDA_VISIBLE_DEVICES=0


MODEL_BASE=your_base_dir/checkpoints/run_grpo_video_cls_qa_ea_ff_nothink.sh_20250327_120247+ea

srun -p videop1 \
    --job-name=eval \
    -n8 \
    --ntasks-per-node=8 \
    --cpus-per-task=8 \
python evaluate_cls_ea_nothink_c8.py \
     --model_base $MODEL_BASE




MODEL_BASE=your_base_dir/checkpoints/run_grpo_video_cls_qa_ea_ff_nothink.sh_20250327_123726+ff

srun -p videop1 \
    --job-name=eval \
    -n8 \
    --ntasks-per-node=8 \
    --cpus-per-task=8 \
python evaluate_cls_quality_nothink_c8.py \
     --model_base $MODEL_BASE

     