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
from my_qwen_utils import process_vision_info
import random
from eval_prompts import GROUND_TEMPLATE_THINK as GROUND_TEMPLATE
from math import ceil

from petrel_client.client import Client
conf_path = '~/petreloss.conf'
client = Client(conf_path)
VIDEO_INFO_CACHE = {}

def get_args():
    parser = argparse.ArgumentParser(description='Evaluation for video temporal grounding')
    parser.add_argument('--dataset', default='charades', type=str, help='Specify the dataset, can be charades or anet')
    parser.add_argument('--split', default='default', type=str, help='Specify the split.')
    parser.add_argument("--model_base", type=str, default="your model")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--result_dir", type=str, default="your log", help="Directory to save results")
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


def create_work_items(data):
    work_items = []
    for vid, ann in data.items():
        for i in range(len(ann['sentences'])):
            work_items.append({
                'vid': vid,
                'ann': ann,
                'sentence_idx': i
            })
    random.shuffle(work_items)
    return work_items

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

def get_checkpoint_path(result_dir):
    os.makedirs(result_dir, exist_ok=True)
    return os.path.join(result_dir, "checkpoint.pkl")

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
    try:
        with open(file_path, 'a', encoding='utf-8') as f:
            json_line = json.dumps(data, ensure_ascii=False)  # 确保非 ASCII 字符正确编码
            f.write(json_line + '\n')  # 每行一个 JSON 对象
    except Exception as e:
        print(f"写入文件时发生错误: {e}")

def process_work_items(work_items, video_dir_path, model_base, device, result_dir, resume=False):
    ious = []
    thresh = np.array([0.3, 0.5, 0.7])
    recall = np.array([0, 0, 0])
    
    os.makedirs(f"./{result_dir}/{model_base.replace('/', '-')}_tg", exist_ok=True)
    log_path = f"{result_dir}/{model_base.replace('/', '-')}_tg/{device}.jsonl"

    processed_items = set()

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
        
        prompt = GROUND_TEMPLATE.replace('[EVENT]', ann['sentences'][sentence_idx])
        
        duration = ann['duration'] if 'duration' in ann else ann['video_duration']
        ext = 'mp4'
        video_path = os.path.join(video_dir_path, f"{vid}.{ext}")
        
        if video_path:
            try:
                # import pdb; pdb.set_trace()
                ans = inference(video_path, prompt, model, processor, device=device)
                # print('prompt', prompt)
                # print('ans', ans)
                sp, ep = parse_timestamp_output(ans)
                print(f"Parsed times: {sp}, {ep}")
                print(f"Ground truth: {ann['timestamps'][sentence_idx]}")
                print('-' * 50)
                
                if (sp is not None) and (ep is not None):
                    s, e = ann['timestamps'][sentence_idx]
                    iou_ = (min(e, ep) - max(s, sp)) / (max(e, ep) - min(s, sp))
                    ious.append(max(iou_, 0))
                    recall += (thresh <= iou_)
                else:
                    ious.append(0)
                
                processed_items.add(item_id)
                item_res = {'video_path': video_path, 'query':ann['sentences'][sentence_idx], 'answer':ans, 'timestamp':[sp, ep], 'iou':iou_, 'ans':ann['timestamps'][sentence_idx] }
                append_to_jsonl(log_path, item_res)
                # if (idx + 1) % 5 == 0 or idx == len(remaining_items) - 1:
                #     state = {
                #         'processed_items': processed_items,
                #         'ious': ious,
                #         'recall': recall
                #     }
                #     save_checkpoint(checkpoint_path, state)
                    
                miou = sum(ious) / len(ious) if ious else 0
                recall_str = str(recall / len(ious) if ious else [0, 0, 0])
                pbar.set_postfix({"mIoU": miou, 'recall': recall_str})
                
            except Exception as e:
                print(f"Error processing {vid}_{sentence_idx}: {e}")
    
    print(f'=== {log_path} result ===')
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
        f'{args.result_dir}_{slurm_procid}',
        args.resume
    )
    
    return ious, recall


def split_data(data, num_gpus):

    is_dict = isinstance(data, dict)

    if is_dict:
        data = list(data.items())
    elif not isinstance(data, list):
        data = list(data)

    data_size = len(data)
    chunk_size = ceil(data_size / num_gpus)  #
    chunks = [data[i * chunk_size:(i + 1) * chunk_size] for i in range(num_gpus)]

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

    slurm_procid = int(os.environ.get('SLURM_PROCID', 0))  
    num_gpus = args.num_gpus

    data_chunks = split_data(data, num_gpus)
    current_data_chunk = data_chunks[slurm_procid]
    evaluate(current_data_chunk, args, slurm_procid)