
export WANDB_PROJECT=Video-GRPO
export WANDB_NAME=Qwen2.5_7b_TG

export PYTHONPATH=".:$PYTHONPATH"
OUTDIR=ckpt/cotraining

export DEBUG_MODE="true"
export LOG_PATH="./log.txt"

srun --ntasks=8 \
    --gres=gpu:8 \
    --ntasks-per-node=8 \
    --cpus-per-task=16 \
    --kill-on-bad-exit=1 \
    python  /mnt/petrelfs/yanziang/videoo1/TimeZero/src/open_r1/grpo_tasks.py \
    --deepspeed scripts/zero3_offload.json \
    --model_name_or_path Your_Model \
    --output_dir $OUTDIR \
    --training_data Videochat-R1.5/data/cotraining.json \
    --dataset_name charades \
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
    --save_steps 1000 \
    --learning_rate 1e-6 \
    --save_only_model true
