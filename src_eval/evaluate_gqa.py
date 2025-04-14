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
from src.open_r1.my_qwen_utils import process_vision_info
import random
import ast
import os
import json
from math import ceil
from eval_prompts import GQA_THINK_ANSWER_GLUE as GQA_TEMPLATE 

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

from petrel_client.client import Client
conf_path = '~/petreloss.conf'
client = Client(conf_path)

VIDEO_INFO_CACHE = {}

def get_args():
    parser = argparse.ArgumentParser(description='Evaluation for nextgqa')
    parser.add_argument('--dataset', default='nextgqa', type=str, help='Specify the dataset.')
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
    
    result = process_vision_info(messages, return_video_kwargs=return_video_kwargs, client=client)
    VIDEO_INFO_CACHE[cache_key] = result
    
    return result

def inference(video_path, prompt, model, processor, max_new_tokens=2048, device="cuda:0"):
    messages = [
        {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"video": video_path, 
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
        video_path = os.path.join(video_root, info['video'])

        example = {
            "problem": {"question":info['question'], "options":info['options']},
            "solution": {"answer":info['answer'], "glue":info['glue']},
            "video_path": video_path,
            "durations": info['duration'],
            "video_id": i
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


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:" "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCDEFG]", s):
        return ""

    matches = re.search(r"[ABCDEFG]", s)
    if matches is None:
        return ""
    return matches[0]


def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = [list(i) for i in intervals] # tuple to list
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0][:]]
    for current in sorted_intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            merged[-1][1] = max(last[1], current[1])
        else:
            merged.append(current[:])
    
    return merged

def compute_iou(list_a, list_b):
    merged_a = merge_intervals(list_a)
    merged_b = merge_intervals(list_b)
    
    len_a = sum(end - start for start, end in merged_a)
    len_b = sum(end - start for start, end in merged_b)
    
    intersection = 0
    i = j = 0
    while i < len(merged_a) and j < len(merged_b):
        a_start, a_end = merged_a[i]
        b_start, b_end = merged_b[j]
        
        start = max(a_start, b_start)
        end = min(a_end, b_end)
        if start < end:
            intersection += end - start
        
        if a_end < b_end:
            i += 1
        else:
            j += 1
    
    union = len_a + len_b - intersection
    if union == 0:
        return 1.0
    
    return intersection / union

def is_valid_two_d_list_format(s):
    pattern = r'^\[(\(\d+(\.\d+)?,\s*\d+(\.\d+)?\)(,\s*\(\d+(\.\d+)?,\s*\d+(\.\d+)?\))*(,)?|)\]$'
    if not re.match(pattern, s):
        return False
    try:
        lst = ast.literal_eval(s)
        if not isinstance(lst, list):
            return False
        for item in lst:
            if not isinstance(item, tuple):
                return False
            if len(item) != 2:
                return False
            for num in item:
                if not isinstance(num, (int, float)):
                    return False
            if item[0] > item[1]: # 保证符合时序区间
                return False
        return True
    except:
        return False

def append_to_jsonl(file_path, data):
    
    try:
        with open(file_path, 'a', encoding='utf-8') as f:
            json_line = json.dumps(data, ensure_ascii=False)  
            f.write(json_line + '\n')  
    except Exception as e:
        print(f"写入文件时发生错误: {e}")

def process_work_items(work_items, model_base, device, result_dir, resume=False):
    model, processor = setup_model(model_base, device)
    
    os.makedirs(f"{result_dir}/{model_base.replace('/', '-')}_gqa", exist_ok=True)
    log_path = f"{result_dir}/{model_base.replace('/', '-')}_gqa/{device}.jsonl"
    print(log_path)
    pbar = tqdm(work_items)
    for idx, item in enumerate(pbar):
        video_path = item['video_path']

        example_prompt = GQA_TEMPLATE.replace("[QUESTION]", item["problem"]["question"])
        prompt = example_prompt.replace("[OPTION]", str(item["problem"]["options"]))


        accs = []
        ious = []

        try:
            ans = inference(video_path, prompt, model, processor, device=device)

            pattern_answer = r'<answer>(.*?)</answer>'
            match_answer = re.search(pattern_answer, ans, re.DOTALL)

            acc = 0.0
            if match_answer:
                answer = match_answer.group(1)
                if extract_characters_regex(answer) == extract_characters_regex(item["solution"]["answer"]):
                    acc = 1.0

            accs.append(acc)

            # IoU

            pattern_glue = r'<glue>(.*?)</glue>'
            match_glue = re.search(pattern_glue, ans, re.DOTALL)

            if match_glue:
                glue = match_glue.group(1)
                if is_valid_two_d_list_format(glue):
                    pred_glue = ast.literal_eval(glue)
                    iou = compute_iou(pred_glue, item["solution"]["glue"])
            else:
                iou = 0.0
            ious.append(iou)
            
            item_res = {'video_path': video_path, 'prompt':prompt, 'gt':item["solution"], 'pred':ans, 'acc':acc, 'iou':iou }
            append_to_jsonl(log_path, item_res)
            
            pbar.set_postfix({"mIoU": sum(ious)/len(ious), 'accuracy': sum(accs)/len(accs)})
            
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
    
    print(f'=== {log_path} result ===')
    # if ious:
    print('mIoU:', sum(ious) / len(ious))
    print("Accuacy:", sum(accs)/len(accs))
                
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