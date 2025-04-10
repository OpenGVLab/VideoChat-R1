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
from src.open_r1.my_qwen_utils import process_vision_info
import random


client = None

VIDEO_INFO_CACHE = {}

def get_args():
    parser = argparse.ArgumentParser(description='Evaluation for training-free video temporal grounding (Single GPU Version)')
    parser.add_argument('--dataset', default='charades', type=str, help='Specify the dataset.')
    parser.add_argument('--split', default='default', type=str, help='Specify the split.')
    parser.add_argument("--model_base", type=str, default="/path/to/qwen-model")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
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

GROUND_TEMPLATE = """To accurately pinpoint the event "[EVENT]" in the video, determine the precise time period of the event.

Output your thought process within the <think> </think> tags, including analysis with either specific timestamps (xx.xx) or time ranges (xx.xx to xx.xx) in <timestep> </timestep> tags.

Then, provide the start and end times (in seconds, precise to two decimal places) in the format "start time to end time" within the <answer> </answer> tags. For example: "12.54 to 17.83"."""

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
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_base,
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

def process_work_items(work_items, video_dir_path, model_base, device, checkpoint_dir, resume=False):
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
        
        prompt = GROUND_TEMPLATE.replace('[EVENT]', ann['sentences'][sentence_idx])
        
        # 确定视频路径
        duration = ann['duration'] if 'duration' in ann else ann['video_duration']
        ext = 'mp4'
        video_path = os.path.join(video_dir_path, f"{vid}.{ext}")
        # for ext in ['mp4', 'mkv', 'webm']:
        #     path = os.path.join(video_dir_path, f"{vid}.{ext}")
        #     if os.path.isfile(path):
        #         video_path = path
        #         break
                
        # 处理视频
        if video_path:
            try:
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
                
            except Exception as e:
                print(f"Error processing {vid}_{sentence_idx}: {e}")
    
    print('=== final result ===')
    # if ious:
    print('mIoU:', sum(ious) / len(ious))
    for th, r in zip(thresh, recall):
        print(f'R@{th}:', r / len(ious))
                
    return ious, recall

def evaluate(data, args):
    dataset = DATASETS[args.dataset]
    video_dir_path = dataset['video_root']
    
    work_items = create_work_items(data)
    
    ious, recall = process_work_items(
        work_items, 
        video_dir_path, 
        args.model_base, 
        args.device, 
        args.checkpoint_dir,
        args.resume
    )
    
    return ious, recall

if __name__=='__main__':
    args = get_args()
    assert args.dataset in DATASETS
    dataset = DATASETS[args.dataset]
    assert args.split in dataset['splits']
    
    print('evaluate', args.dataset, args.split)
    
    # load data
    with open(dataset['splits'][args.split]['annotation_file']) as f:
        data = json.load(f)

    evaluate(data, args)