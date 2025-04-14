from data_config import DATASETS
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
from eval_prompts import TRACK_THINK as  QUESTION_TEMPLATE

def split_data(data, num_gpus):
    
    is_dict = isinstance(data, dict)

    if is_dict:
        data = list(data.items())
    elif not isinstance(data, list):
        data = list(data)

    data_size = len(data)
    chunk_size = ceil(data_size / num_gpus)  

    chunks = [data[i * chunk_size:(i + 1) * chunk_size] for i in range(num_gpus)]

    if is_dict:
        chunks = [dict(chunk) for chunk in chunks]

    return chunks


client = None

VIDEO_INFO_CACHE = {}

def get_args():
    parser = argparse.ArgumentParser(description='Evaluation for track')
    parser.add_argument('--dataset', default='got', type=str, help='Specify the dataset.')
    parser.add_argument('--split', default='default', type=str, help='Specify the split.')
    parser.add_argument("--model_base", type=str, default="/path/to/qwen-model")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--result_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--num_gpus", type=int, default=8, help="GPU device to use")
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
    for file in os.listdir(path):
        if file.endswith(".jpg"):  
            full_path = os.path.join(path, file) 
            jpg_files.append(full_path)
    sorted_files = sorted(jpg_files)
    first_element = sorted_files[0]
    last_element = sorted_files[-1]
    nframes = len(sorted_files)
    step = (nframes - 1) / 6  
    middle_indices = [int(i * step) for i in range(1, 6)]  
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
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)

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
        s = s.replace('\n', '')
        data = ast.literal_eval(s)

        if not isinstance(data, list):
            return False

        if len(data) != 8:
            return False

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

    try:
        with open(file_path, 'a', encoding='utf-8') as f:
            json_line = json.dumps(data, ensure_ascii=False)  
            f.write(json_line + '\n')  
    except Exception as e:
        print(f"写入文件时发生错误: {e}")

def process_work_items(work_items, model_base, device, result_dir, resume=False):
    model, processor = setup_model(model_base, device)

    os.makedirs(f"{result_dir}/{model_base.replace('/', '-')}_track", exist_ok=True)
    log_path = f"{result_dir}/{model_base.replace('/', '-')}_track/{device}.jsonl"
    print(log_path)

    pbar = tqdm(work_items)
    for idx, item in enumerate(pbar):
        video_path = item['video_path']

        example_prompt = QUESTION_TEMPLATE.replace("[OBJECT]", item["problem"]["object"])
        prompt = example_prompt.replace("[START]", str(item["problem"]["start"]))


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

    print(f'=== {log_path} result ===')
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
        f'{args.result_dir}_{slurm_procid}',
        args.resume
    )

    return ious, accs

if __name__=='__main__':
    args = get_args()

    assert args.dataset in DATASETS
    dataset = DATASETS[args.dataset]
    assert args.split in dataset['splits']
    
    print('evaluate', args.dataset, args.split)
    

    slurm_procid = int(os.environ.get('SLURM_PROCID', 0))  # 当前进程的全局 ID
    print(f"slurm_procid: {slurm_procid}")
    num_gpus = args.num_gpus
    with open(dataset['splits'][args.split]['annotation_file']) as f:
        data = json.load(f)
    data_chunks = split_data(data, num_gpus)
    current_data_chunk = data_chunks[slurm_procid]
    gpu_count = torch.cuda.device_count()
    print(f"可用的 GPU 数量: {gpu_count}")
    assert gpu_count == num_gpus, gpu_count
    evaluate(current_data_chunk, dataset['video_path'], slurm_procid, args)