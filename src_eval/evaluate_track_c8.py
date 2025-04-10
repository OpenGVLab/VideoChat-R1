from data_configs import DATASETS
import argparse
import numpy as np
import json
from tqdm import tqdm
import os
import re
import pickle
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import random
import ast
import os
import json
from math import ceil


def split_data(data, num_gpus):
    """
    将数据均匀分割为 num_gpus 块。
    如果数据量不能被 num_gpus 整除，最后一块会包含多余的元素。
    如果数据是字典，则返回的每个块也是字典。
    """
    # 记录原始数据类型
    is_dict = isinstance(data, dict)

    # 确保 data 是可切片的对象
    if is_dict:
        # 如果是字典，将其转换为 (key, value) 列表
        data = list(data.items())
    elif not isinstance(data, list):
        # 如果既不是字典也不是列表，尝试将其转换为列表
        data = list(data)

    data_size = len(data)
    chunk_size = ceil(data_size / num_gpus)  # 每块的大小

    # 分割数据
    chunks = [data[i * chunk_size:(i + 1) * chunk_size] for i in range(num_gpus)]

    # 如果原始数据是字典，将每个块转换回字典
    if is_dict:
        chunks = [dict(chunk) for chunk in chunks]

    return chunks


client = None

VIDEO_INFO_CACHE = {}

def get_args():
    parser = argparse.ArgumentParser(description='Evaluation for training-free video temporal grounding (Single GPU Version)')
    parser.add_argument('--dataset', default='charades', type=str, help='Specify the dataset.')
    parser.add_argument('--split', default='default', type=str, help='Specify the split.')
    parser.add_argument("--model_base", type=str, default="/mnt/petrelfs/yanziang/videoo1/TimeZero/ckpt/grpo_tracking")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--checkpoint_dir", type=str, default="/mnt/petrelfs/yanziang/videoo1/TimeZero/eval_logs/grpo_tracking/res", help="Directory to save checkpoints")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0", help="GPU device to use")
    return parser.parse_args()

def calc_iou(candidates, gt):
    start, end = candidates[:,0], candidates[:,1]
    s, e = gt[0], gt[1]
    inter = np.minimum(end, e) - np.maximum(start, s)
    union = np.maximum(end, e) - np.minimum(start, s)
    return inter.clip(min=0) / union

def cached_process_vision_info(messages, return_video_kwargs=False):
    global VIDEO_INFO_CACHE

    video_path = None
    for msg in messages:
        for content in msg.get('content', []):
            if isinstance(content, dict) and 'video' in content:
                video_path = content['video']
                break

    cache_key = f"{video_path}_{return_video_kwargs}"
    if cache_key in VIDEO_INFO_CACHE:
        return VIDEO_INFO_CACHE[cache_key]

    result = process_vision_info(messages, return_video_kwargs=return_video_kwargs)
    VIDEO_INFO_CACHE[cache_key] = result

    return result

def inference(video_path, prompt, model, processor, max_new_tokens=2048, device="cuda:0"):

    path = video_path
    jpg_files = []
    # for root, dirs, files in os.walk(path):  
    #     for file in files:
    for file in os.listdir(path):
        if file.endswith(".jpg"):  
            full_path = os.path.join(path, file) 
            jpg_files.append(full_path)
    sorted_files = sorted(jpg_files)
    first_element = sorted_files[0]
    last_element = sorted_files[-1]
    nframes = len(sorted_files)
    # 计算中间均匀分布的六个元素的索引
    step = (nframes - 1) / 6  # 均匀间隔步长
    middle_indices = [int(i * step) for i in range(1, 6)]  # 跳过第一个和最后一个
    middle_elements = [sorted_files[i] for i in middle_indices]

    result = [first_element] + middle_elements + [last_element]
    messages = [
        {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"video": result, 
                "total_pixels": 3584 * 28 * 28, 
                "min_pixels": 16 * 28 * 28,
                },
            ]
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    image_inputs, video_inputs, video_kwargs = cached_process_vision_info(messages, return_video_kwargs=True)
    fps_inputs = video_kwargs['fps']

    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, fps=fps_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids = [output_ids[i][len(inputs.input_ids[i]):] for i in range(len(output_ids))]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]

def parse_timestamp_output(output_string):
    matches = re.findall(r"(\d+\.?\d*) (to|and) (\d+\.?\d*)", output_string)
    if not matches:
        answer_match = re.search(r"<answer>(.*?)</answer>", output_string)
        if answer_match:
            answer_content = answer_match.group(1).strip()
            answer_matches = re.findall(r"(\d+\.?\d*) (to|and) (\d+\.?\d*)", answer_content)
            if answer_matches:
                last_match = answer_matches[-1]
                return float(last_match[0]), float(last_match[2])
        return None, None

    last_match = matches[-1]
    start_time_str = last_match[0]
    end_time_str = last_match[2]

    try:
        start_time = float(start_time_str)
        end_time = float(end_time_str)
        return start_time, end_time
    except ValueError:
        return None, None



QUESTION_TEMPLATE = """Track the "[OBJECT]" in the video based on its initial coordinates "[START]". The output should be a list containing eight sublists. Each sublist includes four normalized coordinates [x0, y0, x1, y1] representing the bounding box of the object at specific time intervals.

Output your thought process within the <think> </think> tags, including analysis with either specific timestamps (xx.xx) or time ranges (xx.xx to xx.xx) in <timestep> </timestep> tags.

Provide your answer within the <answer> </answer> tags as a list of eight sublists, where each sublist contains the normalized coordinates [x0, y0, x1, y1]. For example: <answer>[[0.1, 0.5, 0.3, 0.55], [0.72, 0.25, 0.84, 0.43], ...]</answer>.
"""






def create_work_items(data, video_root):
    examples = []
    for i, info in enumerate(data):
        video_path = os.path.join(video_root, info['path'])

        example = {
            "problem": {"object":info['object'], "start":info['gt'][0]},
            "solution": {"answer":info['gt']},
            "video_path": video_path,
            # "durations": info['duration'],
        }

        examples.append(example)
    # # 随机打乱列表
    # random.shuffle(work_items)
    return examples

def setup_model(model_base, device):
    print(f"Setting up model on device {device}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_base,
        torch_dtype=torch.bfloat16,
        use_sliding_window=True,
        attn_implementation="flash_attention_2",
        device_map=device
    )
    processor = AutoProcessor.from_pretrained(model_base)
    return model, processor




def is_valid_list_of_lists(s):
    try:
        # 尝试将字符串解析为 Python 对象
        s = s.replace('\n', '')
        data = ast.literal_eval(s)

        # 检查解析后的对象是否是一个列表
        if not isinstance(data, list):
            return False

        # 检查列表的长度是否为 8
        if len(data) != 8:
            return False

        # 检查列表中的每个元素是否是长度为 4 的列表
        for element in data:
            if not (isinstance(element, list) and len(element) == 4):
                return False

        return True
    except Exception as e:
        print(f'Exception at is_valid_list_of_lists:{e}')
        return False


def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: [x0, y0, x1, y1] for the first bounding box
        box2: [x0, y0, x1, y1] for the second bounding box

    Returns:
        iou: The IoU value between the two boxes
    """
    # Extract coordinates
    x0_1, y0_1, x1_1, y1_1 = box1
    x0_2, y0_2, x1_2, y1_2 = box2

    # Calculate the coordinates of the intersection rectangle
    inter_x0 = max(x0_1, x0_2)
    inter_y0 = max(y0_1, y0_2)
    inter_x1 = min(x1_1, x1_2)
    inter_y1 = min(y1_1, y1_2)

    # Calculate the area of intersection
    inter_width = max(0, inter_x1 - inter_x0)
    inter_height = max(0, inter_y1 - inter_y0)
    inter_area = inter_width * inter_height

    # Calculate the area of both bounding boxes
    box1_area = (x1_1 - x0_1) * (y1_1 - y0_1)
    box2_area = (x1_2 - x0_2) * (y1_2 - y0_2)

    # Calculate the area of union
    union_area = box1_area + box2_area - inter_area

    # Avoid division by zero
    if union_area == 0:
        return 0.0

    # Calculate IoU
    iou = inter_area / union_area
    return iou

def average_overlap(pred, gt):
    """
    Calculate the Average Overlap (average IoU) between predicted and ground truth boxes.

    Args:
        pred: List of predicted bounding boxes, each in [x0, y0, x1, y1] format
        gt: List of ground truth bounding boxes, each in [x0, y0, x1, y1] format

    Returns:
        avg_iou: The average IoU value across all pairs of boxes
    """
    if len(pred) != len(gt):
        raise ValueError("The number of predicted boxes must match the number of ground truth boxes.")

    iou_values = []
    for p_box, g_box in zip(pred, gt):
        iou = calculate_iou(p_box, g_box)
        iou_values.append(iou)

    avg_iou = np.mean(iou_values)
    return avg_iou



def append_to_jsonl(file_path, data):
    """
    追加模式写入 JSONL 文件。

    参数:
        file_path (str): JSONL 文件路径。
        data (dict): 要写入的 JSON 对象（Python 字典）。
    """
    try:
        # 以追加模式打开文件
        with open(file_path, 'a', encoding='utf-8') as f:
            # 将数据序列化为 JSON 字符串并写入文件
            json_line = json.dumps(data, ensure_ascii=False)  # 确保非 ASCII 字符正确编码
            f.write(json_line + '\n')  # 每行一个 JSON 对象
    except Exception as e:
        print(f"写入文件时发生错误: {e}")

def process_work_items(work_items, model_base, device, checkpoint_dir, resume=False):
    model, processor = setup_model(model_base, device)

    os.makedirs(f"./eval_logs/{model_base.replace('/', '-')}_track", exist_ok=True)
    log_path = f"./eval_logs/{model_base.replace('/', '-')}_track/{device}.jsonl"
    print(log_path)

    pbar = tqdm(work_items)
    for idx, item in enumerate(pbar):
        video_path = item['video_path']

        example_prompt = QUESTION_TEMPLATE.replace("[OBJECT]", item["problem"]["object"])
        prompt = example_prompt.replace("[START]", str(item["problem"]["start"]))
        # example_prompt = QUESTION_TEMPLATE.replace("[QUESTION]", item["problem"]["question"])
        # prompt = example_prompt.replace("[OPTION]", str(item["problem"]["options"]))


        accs = []
        ious = []

        # try:
        ans = inference(video_path, prompt, model, processor, device=device)

        pattern_answer = r'<answer>(.*?)</answer>'
        match_answer = re.search(pattern_answer, ans, re.DOTALL)


        # match_glue = re.search(match_answer, ans, re.DOTALL)
        # print(f'ann:{ans}')
        iou = 0
        if match_answer:
            glue = match_answer.group(1)
            # import pdb; pdb.set_trace()
            # if is_valid_two_d_list_format(glue):
            if is_valid_list_of_lists(glue):

                glue = glue.replace('\n', '')
                pred_glue = ast.literal_eval(glue)
                iou = average_overlap(pred_glue, item["solution"]["answer"])
        else:
            iou = 0.0
        ious.append(iou)

        item_res = {'video_path': video_path, 'prompt':prompt, 'gt':item["solution"], 'pred':ans, 'iou':iou }
        append_to_jsonl(log_path, item_res)

        pbar.set_postfix({"mIoU": sum(ious)/len(ious)})

        # except Exception as e:
        #     print(f"Error processing {video_path}: {e}")

    print('=== final result ===')
    # if ious:
    print('mIoU:', sum(ious) / len(ious))
    # print("Accuacy:", sum(accs)/len(accs))

    return ious, accs

def evaluate(data, video_root, slurm_procid, args):
    work_items = create_work_items(data, video_root=video_root)

    ious, accs = process_work_items(
        work_items, 
        args.model_base, 
        f'cuda:{slurm_procid}', 
        f'{args.checkpoint_dir}_{slurm_procid}',
        args.resume
    )

    return ious, accs

if __name__=='__main__':
    args = get_args()

    # load data
    with open("/mnt/petrelfs/share_data/yanziang/got_val.json", "r") as f:
        data = json.load(f)

    slurm_procid = int(os.environ.get('SLURM_PROCID', 0))  # 当前进程的全局 ID
    print(f"slurm_procid: {slurm_procid}")
    num_gpus = 8  # 假设总共有 8 块 GPU

    data_chunks = split_data(data, num_gpus)
    current_data_chunk = data_chunks[slurm_procid]
    gpu_count = torch.cuda.device_count()
    print(f"可用的 GPU 数量: {gpu_count}")
    assert gpu_count == num_gpus, gpu_count
    # import pdb;pdb.set_trace()
    evaluate(current_data_chunk, "", slurm_procid, args)