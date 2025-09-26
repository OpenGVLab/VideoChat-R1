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


def merge_intervals(intervals):
    """合并重叠或相邻的时间区间"""
    if not intervals:
        return []
    intervals = [list(i) for i in intervals] # tuple to list
    # 按起始时间排序
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0][:]]  # 复制第一个区间
    for current in sorted_intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            # 合并区间
            merged[-1][1] = max(last[1], current[1])
        else:
            merged.append(current[:])

    # print(merged)
    return merged

def compute_iou(list_a, list_b):
    # 合并两个列表的区间
    merged_a = merge_intervals(list_a)
    merged_b = merge_intervals(list_b)

    # 计算各自的总长度
    len_a = sum(end - start for start, end in merged_a)
    len_b = sum(end - start for start, end in merged_b)

    # 计算交集的总长度
    intersection = 0
    i = j = 0
    while i < len(merged_a) and j < len(merged_b):
        a_start, a_end = merged_a[i]
        b_start, b_end = merged_b[j]

        # 计算当前两个区间的重叠部分
        start = max(a_start, b_start)
        end = min(a_end, b_end)
        if start < end:
            intersection += end - start

        # 移动指针
        if a_end < b_end:
            i += 1
        else:
            j += 1

    # 计算并集总长度
    union = len_a + len_b - intersection
    if union == 0:
        return 1.0

    return intersection / union
VIDEO_INFO_CACHE = {}

def get_args():
    parser = argparse.ArgumentParser(description='Evaluation for training-free video temporal grounding (Single GPU Version)')
    parser.add_argument('--dataset', default='charades', type=str, help='Specify the dataset.')
    parser.add_argument('--split', default='default', type=str, help='Specify the split.')
    parser.add_argument("--model_base", type=str, default="/path/to/model")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--checkpoint_dir", type=str, default="/path/to/out_dir")
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
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
                {"type": "video", 
                "video": video_path, 
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



# GROUND_TEMPLATE = """To accurately pinpoint the event "[QUESTION]" in the video, determine the precise time period of the event.

# Output your thought process within the <think> </think> tags.

# Then, provide the start and end times (in seconds, precise to two decimal places) in the format of [(s1, e1), (s2, e2), ...] within the <answer> </answer> tags. For example: <think>...</think><answer>[(12.54, 17.83)]</answer>"""


GROUND_TEMPLATE = """To accurately pinpoint the event "[QUESTION]" in the video, determine the precise time period of the event.

Provide the start and end times (in seconds, precise to two decimal places) in the format of [(s1, e1), (s2, e2), ...] within the <answer> </answer> tags. For example: <answer>[(12.54, 17.83)]</answer>"""


# GROUND_TEMPLATE = """To accurately pinpoint the event "[EVENT]" in the video, determine the precise time period of the event.

# Provide the start and end times (in seconds, precise to two decimal places) in the format "start time to end time" within the <answer> </answer> tags. For example: "12.54 to 17.83"."""

def create_work_items(data):
    work_items = []
    for vid, ann in data.items():
        for i in range(len(ann['sentences'])):
            work_items.append({
                'vid': vid,
                'ann': ann,
                'sentence_idx': i
            })
    # 随机打乱列表
    random.shuffle(work_items)
    return work_items

def setup_model(model_base, device):
    print(f"Setting up model on device {device}")
    # import pdb; pdb.set_trace()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_base,
        trust_remote_code = True,
        torch_dtype=torch.bfloat16,
        use_sliding_window=True,
        attn_implementation="flash_attention_2",
        device_map=device
    )
    processor = AutoProcessor.from_pretrained(model_base)
    return model, processor

def get_checkpoint_path(checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    return os.path.join(checkpoint_dir, "checkpoint.pkl")

def load_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    return {'processed_items': set(), 'ious': [], 'recall': np.array([0, 0, 0])}

def save_checkpoint(checkpoint_path, state):
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(state, f)

import json

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

def process_work_items(work_items, video_dir_path, model_base, device, checkpoint_dir, resume=False, slurm_procid=0):
    ious = []
    thresh = np.array([0.3, 0.5, 0.7])
    recall = np.array([0, 0, 0])

    # 加载检查点（如果需要恢复）
    checkpoint_path = get_checkpoint_path(checkpoint_dir)
    processed_items = set()

    if resume and os.path.exists(checkpoint_path):
        checkpoint = load_checkpoint(checkpoint_path)
        processed_items = checkpoint['processed_items']
        ious = checkpoint['ious']
        recall = checkpoint['recall']
        print(f"Resuming from checkpoint with {len(processed_items)} processed items")

    model, processor = setup_model(model_base, device)

    item_ids = [f"{item['vid']}_{item['sentence_idx']}" for item in work_items]
    remaining_items = [(i, item) for i, (item, item_id) in enumerate(zip(work_items, item_ids)) 
                      if not resume or item_id not in processed_items]

    if not remaining_items:
        print("All items already processed")
        return ious, recall

    print(f"Processing {len(remaining_items)} out of {len(work_items)} items")

    pbar = tqdm(remaining_items)
    for idx, (_, item) in enumerate(pbar):
        vid = item['vid']
        ann = item['ann']
        sentence_idx = item['sentence_idx']
        item_id = f"{vid}_{sentence_idx}"

        prompt = GROUND_TEMPLATE.replace('[QUESTION]', ann['sentences'][sentence_idx])

        # 确定视频路径
        duration = ann['duration'] if 'duration' in ann else ann['video_duration']
        video_path = None
        ext = 'mp4'
        path = os.path.join(video_dir_path, f"{vid}.{ext}")

        video_path = path
                # break

        # 处理视频
        # import pdb;pdb.set_trace()
        if video_path:
        
            ans = inference(video_path, prompt, model, processor, device=device)
            # print('prompt', prompt)
            # print('ans', ans)
            # import pdb;pdb.set_trace()
            
            pattern_glue = r'<answer>(.*?)</answer>'
            match_glue = re.search(pattern_glue, ans, re.DOTALL)
            print(f'ann:{ans}')
            # import pdb; pdb.set_trace()
            iou = 0
            pred_glue = [(0,0)]
            try:
                if match_glue:
                    glue = match_glue.group(1)
                    # import pdb; pdb.set_trace()
                    # if is_valid_two_d_list_format(glue):
                    # import pdb; pdb.set_trace()
                    pred_glue = ast.literal_eval(glue)
                    # pred_glue = [(pred_glue[0][0], pred_glue[0][1])]
                    
                    iou = compute_iou(pred_glue, [item["ann"]["timestamps"][sentence_idx]])
                else:
                    # print('-----wrong---------')
                    pred_glue = ast.literal_eval(ans)
                    iou = compute_iou(pred_glue, [item["ann"]["timestamps"][sentence_idx]])
                    # iou = 0.0
            except Exception as e:
                pred_glue = [(0,0)]
                iou = 0
            
            ious.append(iou)
            


            processed_items.add(item_id)
            item_res = {'video_path': video_path, 'query':ann['sentences'][sentence_idx], 'answer':ans, 'timestamp':pred_glue, 'iou':iou, 'ans':ann['timestamps'][sentence_idx] }
            append_to_jsonl(checkpoint_path.replace('.pkl', '.jsonl'), item_res)
            if (idx + 1) % 5 == 0 or idx == len(remaining_items) - 1:
                state = {
                    'processed_items': processed_items,
                    'ious': ious,
                    'recall': recall
                }
                save_checkpoint(checkpoint_path, state)

            miou = sum(ious) / len(ious) if ious else 0
            recall_str = str(recall / len(ious) if ious else [0, 0, 0])
            pbar.set_postfix({"mIoU": miou, 'recall': recall_str})

            # except Exception as e:
            #     print(f"Error processing {vid}_{sentence_idx}: {e}")

    print('=== final result ===')
    # if ious:
    print('mIoU:', sum(ious) / len(ious))
    for th, r in zip(thresh, recall):
        print(f'R@{th}:', r / len(ious))

    return ious, recall

def evaluate(data, args, slurm_procid):
    dataset = DATASETS[args.dataset]
    video_dir_path = dataset['video_path']

    work_items = create_work_items(data)

    ious, recall = process_work_items(
        work_items, 
        video_dir_path, 
        args.model_base, 
        f'cuda:{slurm_procid}', 
        f'{args.checkpoint_dir}_{slurm_procid}',
        args.resume,
        slurm_procid
    )

    return ious, recall


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
if __name__=='__main__':
    args = get_args()
    assert args.dataset in DATASETS
    dataset = DATASETS[args.dataset]
    assert args.split in dataset['splits']

    print('evaluate', args.dataset, args.split)

    # load data
    with open(dataset['splits'][args.split]['annotation_file']) as f:
        data = json.load(f)

    slurm_procid = int(os.environ.get('SLURM_PROCID', 0))  # 当前进程的全局 ID
    num_gpus = 8  # 假设总共有 8 块 GPU
    

    data_chunks = split_data(data, num_gpus)
    
    current_data_chunk = data_chunks[slurm_procid]
    
    evaluate(current_data_chunk, args, slurm_procid)