
# export WANDB_PROJECT=Video-GRPO
export WANDB_NAME=Qwen2.5_7b_TG

export PYTHONPATH=".:$PYTHONPATH"
OUTDIR=outputs_video

export DEBUG_MODE="true"
export LOG_PATH="./qwen2.5_7b_vl_tg_video.txt"


srun -p videop1 \
    --job-name=${WANDB_NAME} \
    -n1 \
    --gres=gpu:8 \
    --ntasks-per-node=1 \
    --cpus-per-task=128 \
    --kill-on-bad-exit=1 torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="10668" \
    src/open_r1/grpo_tg.py \
    --deepspeed scripts/zero3_offload.json \
    --output_dir $OUTDIR \
    --model_name_or_path  your_base_dir/Qwen2.5-VL-7B-Instruct \
    --train_data_path ./Charades/charades_annotation/train.json \
    --eval_data_path ./Charades/charades_annotation/val.json \
    --video_folder p2:s3://star/Charades_v1_480 \
    --dataset_name xxx \
    --max_prompt_length 8192 \
    --max_completion_length 1024 \
    --num_generations 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $WANDB_NAME \
    --report_to tensorboard \
    --save_steps 200 \
    --save_total_limit 3 \
    --save_only_model true
