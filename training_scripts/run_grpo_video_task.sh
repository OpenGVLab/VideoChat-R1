
# export WANDB_PROJECT=Video-GRPO
export OMP_NUM_THREADS=1
export DISABLE_ADDMM_CUDA_LT=1
export TORCH_CUDNN_USE_HEURISTIC_MODE_B=1
export NCCL_SOCKET_IFNAME=bond0
# export NCCL_DEBUG="INFO"
export NCCL_IB_HCA=mlx5_0

export WANDB_NAME=$(basename $0)_$(date +"%Y%m%d_%H%M%S")

export PYTHONPATH=".:$PYTHONPATH"
OUTDIR=./checkpoints/$WANDB_NAME

export DEBUG_MODE="true"
export LOG_PATH="./logs/${WANDB_NAME}.log"

srun torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12351" \
    src/open_r1/grpo_tasks.py \
    --deepspeed scripts/zero3_offload.json \
    --output_dir $OUTDIR \
    --model_name_or_path your_base_dir/Qwen2.5-VL-7B-Instruct \
    --train_data_path_gqa /mnt/petrelfs/share_data/lixinhao/videochat-next/reason_origin/nextgqa/nextgqa_val.json \
    --train_data_path_tg ./Charades/charades_annotation/train.json \
    --train_data_path_tracking your_base_dir/track_got_train.json \
    --eval_data_path /mnt/petrelfs/share_data/lixinhao/videochat-next/reason_origin/nextgqa/nextgqa_val.json \
    --video_folder_gqa p2:s3://nextqa \
    --video_folder_tg p2:s3://star/Charades_v1_480 \
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
    --save_steps 10000 \
    --save_only_model true
