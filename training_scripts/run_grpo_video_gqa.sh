
# export WANDB_PROJECT=Video-GRPO
export WANDB_NAME=$(basename $0)_$(date +"%Y%m%d_%H%M%S")

export PYTHONPATH=".:$PYTHONPATH"
OUTDIR=./checkpoints/$WANDB_NAME

export DEBUG_MODE="true"
export LOG_PATH="./logs/${WANDB_NAME}.log"


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
    src/open_r1/grpo_gqa.py \
    --deepspeed scripts/zero3_offload.json \
    --output_dir $OUTDIR \
    --model_name_or_path  your_base_dir/Qwen2.5-VL-7B-Instruct \
    --train_data_path your_base_dir/NextGQA/nextgqa_val.json \
    --eval_data_path your_base_dir/NextGQA/nextgqa_test.json \
    --video_folder p2:s3://nextqa \
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
    --save_steps 100 \
    --save_total_limit 1 \
    --save_only_model true
