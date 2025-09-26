MASTER_PORT=$((18000 + $RANDOM % 100))
TASK=mvbench_nothink


CKPT_PATH=OpenGVLab/VideoChat-R1_7B
MODEL_NAME=qwen2_5_vl_lxh
MAX_NUM_FRAMES=256
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

JOB_NAME=$(basename $0)_$(date +"%Y%m%d_%H%M%S")

NUM_GPUS=8

srun -p videop1 \
    --job-name=${JOB_NAME} \
    --ntasks=1 \
    --gres=gpu:${NUM_GPUS} \
    --ntasks-per-node=1 \
    --cpus-per-task=16 \
    --quotatype=spot \
    --kill-on-bad-exit=1 \
    accelerate launch --num_processes ${NUM_GPUS} --main_process_port 10078 -m lmms_eval \
        --model ${MODEL_NAME} \
        --model_args pretrained=$CKPT_PATH\
        --tasks $TASK \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix $TASK_SUFFIX \
        --output_path ./logs_reason/${JOB_NAME}


MASTER_PORT=$((18000 + $RANDOM % 100))
TASK=mvbench_nothink


CKPT_PATH=OpenGVLab/VideoChat-R1_7B
MODEL_NAME=qwen2_5_vl_lxh
MAX_NUM_FRAMES=256
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

JOB_NAME=$(basename $0)_$(date +"%Y%m%d_%H%M%S")

NUM_GPUS=8

srun -p videop1 \
    --job-name=${JOB_NAME} \
    --ntasks=1 \
    --gres=gpu:${NUM_GPUS} \
    --ntasks-per-node=1 \
    --cpus-per-task=16 \
    --quotatype=spot \
    --kill-on-bad-exit=1 \
    accelerate launch --num_processes ${NUM_GPUS} --main_process_port 10078 -m lmms_eval \
        --model ${MODEL_NAME} \
        --model_args pretrained=$CKPT_PATH\
        --tasks $TASK \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix $TASK_SUFFIX \
        --output_path ./logs_reason/${JOB_NAME}




MASTER_PORT=$((18000 + $RANDOM % 100))
TASK=perceptiontest_val_mc_nothink

CKPT_PATH=OpenGVLab/VideoChat-R1_7B
MODEL_NAME=qwen2_5_vl_lxh
MAX_NUM_FRAMES=256
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

JOB_NAME=$(basename $0)_$(date +"%Y%m%d_%H%M%S")

NUM_GPUS=8

srun -p videop1 \
    --job-name=${JOB_NAME} \
    --ntasks=1 \
    --gres=gpu:${NUM_GPUS} \
    --ntasks-per-node=1 \
    --cpus-per-task=16 \
    --quotatype=spot \
    --kill-on-bad-exit=1 \
    accelerate launch --num_processes ${NUM_GPUS} --main_process_port 10078 -m lmms_eval \
        --model ${MODEL_NAME} \
        --model_args pretrained=$CKPT_PATH\
        --tasks $TASK \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix $TASK_SUFFIX \
        --output_path ./logs_reason/${JOB_NAME}



