
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


# srun accelerate launch --config_file=/mnt/petrelfs/yanziang/videoo1/TimeZero/configs/zero3.yaml 
srun torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12951" \
    src/sft_track.py \
    --deepspeed scripts/zero3_offload.json \
    --model_name_or_path your_base_dir/Qwen2.5-VL-7B-Instruct \
    --preprocessed_data_path ./Charades_preprocessed_data_maxpix_3584 \
    --train_data_path your_base_dir/track_got_train.json \
    --eval_data_path your_base_dir/track_got_train.json \
    --video_folder p2:s3://nextqa \
    --dataset_name xxx \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --max_seq_length 8192 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --bf16 \
    --torch_dtype bfloat16 \
    --logging_steps 5 \
    --eval_strategy no \
    --report_to tensorboard \
    --output_dir $OUTDIR \
    --save_steps 30000 \
    --save_only_model true