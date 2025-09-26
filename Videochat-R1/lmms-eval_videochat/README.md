
# How to use

We have modified the data loading method for lmms-eval: instead of loading from Huggingface, the data is loaded locally. Therefore, when using it, you need to **specify the data path** in the YAML file of each task. The data can be downloaded from the [lmms-eval](https://huggingface.co/lmms-lab) or the official repos of the corresponding tasks.

## Installation

You can install the package by cloning the repository and running the following command:
```bash
git clone https://github.com/OpenGVLab/VideoChat-R1
cd lmms-eval_videochat
pip install -e .
```
We provide all evaluation [scripts](scripts_reason) and [annotations](eval_annotations) here.

You could evaluate one task:
```bash
TASK=videomme_short_nothink
MODEL_NAME=qwen2_5_vl_lxh
MAX_NUM_FRAMES=512
CKPT_PATH=OpenGVLab/VideoChat-R1-7B

echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

JOB_NAME=$(basename $0)_$(date +"%Y%m%d_%H%M%S")
MASTER_PORT=$((18000 + $RANDOM % 100))
NUM_GPUS=8


accelerate launch --num_processes ${NUM_GPUS} --main_process_port ${MASTER_PORT} -m lmms_eval \
    --model ${MODEL_NAME} \
    --model_args pretrained=$CKPT_PATH,max_num_frames=$MAX_NUM_FRAMES \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/${JOB_NAME}_${MODEL_NAME}_f${MAX_NUM_FRAMES}
```
You could evaluate more tasks once like:
```bash
TASK=mvbench_nothink,videomme_short_nothink
MODEL_NAME=qwen2_5_vl_lxh
MAX_NUM_FRAMES=512
CKPT_PATH=OpenGVLab/VideoChat-Flash-Qwen2-7B_res448

echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

JOB_NAME=$(basename $0)_$(date +"%Y%m%d_%H%M%S")
MASTER_PORT=$((18000 + $RANDOM % 100))
NUM_GPUS=8


accelerate launch --num_processes ${NUM_GPUS} --main_process_port ${MASTER_PORT} -m lmms_eval \
    --model ${MODEL_NAME} \
    --model_args pretrained=$CKPT_PATH,max_num_frames=$MAX_NUM_FRAMES \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/${JOB_NAME}_${MODEL_NAME}_f${MAX_NUM_FRAMES}
```

