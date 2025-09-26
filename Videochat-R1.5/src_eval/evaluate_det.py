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
from my_vision_process import process_vision_info
import random
import ast
from petrel_client.client import Client
import os
import json
from math import ceil
client = Client('~/petreloss.conf')
val_list = [
    {
        'anno_path': '/mnt/petrelfs/share_data/yanziang/video_agent_data/additional/REC_refcoco+_unc_val.json',
        'data_root':"pnorm2:s3://coco-caption/train2014/",
    },
    {
        'anno_path': '/mnt/petrelfs/share_data/yanziang/video_agent_data/additional/REC_refcoco_unc_val.json',
        'data_root':"pnorm2:s3://coco-caption/train2014/",
    },
    {
        'anno_path': '/mnt/petrelfs/share_data/yanziang/video_agent_data/additional/REC_refcocog_umd_test.json',
        'data_root':"pnorm2:s3://coco-caption/train2014/",
    },
    {
        'anno_path': '/mnt/petrelfs/share_data/yanziang/video_agent_data/additional/REC_refcoco_unc_testB.json',
        'data_root':"pnorm2:s3://coco-caption/train2014/",

    },
    {
        'anno_path': '/mnt/petrelfs/share_data/yanziang/video_agent_data/additional/REC_refcoco_unc_testA.json',
        'data_root':"pnorm2:s3://coco-caption/train2014/",

    },
    {
        'anno_path': '/mnt/petrelfs/share_data/yanziang/video_agent_data/additional/REC_refcoco+_unc_testB.json',
        'data_root':"pnorm2:s3://coco-caption/train2014/",
    },
    {
        'anno_path': '/mnt/petrelfs/share_data/yanziang/video_agent_data/additional/REC_refcoco+_unc_testA.json',
        'data_root':"pnorm2:s3://coco-caption/train2014/",

    },
    {
        'anno_path': '/mnt/petrelfs/share_data/yanziang/video_agent_data/additional/REC_refcocog_umd_val.json',
        'data_root':"pnorm2:s3://coco-caption/train2014/",
    }
]
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
def cached_process_vision_info(messages, return_video_kwargs=False):
    global VIDEO_INFO_CACHE

    video_path = None
    for msg in messages:
        for content in msg.get('content', []):
            if isinstance(content, dict) and 'image' in content:
                video_path = content['image']
                break

    cache_key = f"{video_path}_{return_video_kwargs}"
    if cache_key in VIDEO_INFO_CACHE:
        return VIDEO_INFO_CACHE[cache_key]
    # import pdb; pdb.set_trace()
    result = process_vision_info(messages, client=client, return_video_kwargs=return_video_kwargs)
    VIDEO_INFO_CACHE[cache_key] = result

    return result

def inference(image_path, prompt, model, processor, max_new_tokens=2048, device="cuda:0"):
    messages = [
        {"role": "user", "content": [
                {"type": "image", 
                "image": image_path, 
                },
                {"type": "text", "text": prompt},
            ]
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    image_inputs, video_inputs, video_kwargs = cached_process_vision_info(messages, return_video_kwargs=True)

    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)

    generated_ids = [output_ids[i][len(inputs.input_ids[i]):] for i in range(len(output_ids))]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]


# answer + glue + think prompt
DETECTION = """Detect the object "[QUESTION]" in the image within <glue> </glue> tag, and give me the caption of the image.

For example, <glue>...</glue> <sum>...</sum>"""

# DETECTION = """Detect the object "[QUESTION]" in the image. Outut the object in json format, and the """
# DETECTION = """Detect the object "[QUESTION]" in the image. 

# The position should be represented as a bounding box in the format [x_min, y_min, x_max, y_max].

# The bbox is: 
# """
def create_work_items(data, image_root):
    examples = []
    # import pdb; pdb.set_trace()
    with open(data['anno_path'], 'r') as f:
        now_data = json.load(f)
    for i, info in enumerate(now_data):
        
        video_path = os.path.join(image_root, info['img_path'])

        example = {
            "problem":info['expression'],
            "solution": info['bbox'],
            "image_path": video_path,
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


def append_to_jsonl(file_path, data):
    try:
        # 以追加模式打开文件
        with open(file_path, 'a', encoding='utf-8') as f:
            # 将数据序列化为 JSON 字符串并写入文件
            json_line = json.dumps(data, ensure_ascii=False)  # 确保非 ASCII 字符正确编码
            f.write(json_line + '\n')  # 每行一个 JSON 对象
    except Exception as e:
        print(f"写入文件时发生错误: {e}")



def calculate_iou(box1, box2):
    """
    计算两个边界框的交并比（IoU）。
    
    参数:
        box1 (list): 第一个边界框 [w0, h0, w1, h1]
        box2 (list): 第二个边界框 [w0, h0, w1, h1]
    
    返回:
        float: 两个边界框的 IoU 值（范围为 [0, 1]）
    """
    box1_w0, box1_h0, box1_w1, box1_h1 = box1
    box2_w0, box2_h0, box2_w1, box2_h1 = box2

    intersection_width = max(0, min(box1_w1, box2_w1) - max(box1_w0, box2_w0))
    intersection_height = max(0, min(box1_h1, box2_h1) - max(box1_h0, box2_h0))
    
    intersection_area = intersection_width * intersection_height
    
    box1_area = (box1_w1 - box1_w0) * (box1_h1 - box1_h0)
    box2_area = (box2_w1 - box2_w0) * (box2_h1 - box2_h0)
    
    union_area = box1_area + box2_area - intersection_area
    
    if union_area == 0:
        return 0.0
    
    iou = intersection_area / union_area
    return iou

def detection_iou(content, sol):
    reward = 0.0
    
    content_match = re.search(r'<answer>(.*?)</answer>', content)
    gred_answer = content_match.group(1).strip() if content_match else content.strip()
    try:
        gred_answer = ast.literal_eval(gred_answer)            
        
        reward = calculate_iou(gred_answer, sol)
    except Exception as e:
        print(e)
    
            

def process_work_items(work_items, model_base, device, checkpoint_dir, resume=False):
    model, processor = setup_model(model_base, device)

    log_path = f"{checkpoint_dir}_{device}.jsonl"
    print(log_path)
    pbar = tqdm(work_items)
    accs = []
    ious = []
    for idx, item in enumerate(pbar):
        video_path = item['image_path']

        example_prompt = DETECTION.replace("[QUESTION]", item["problem"])



        # try:
        ans = inference(video_path, example_prompt, model, processor, device=device)

        pattern_answer = r'<answer>(.*?)</answer>'
        match_answer = re.search(pattern_answer, ans, re.DOTALL)


        pattern_glue = r'<glue>(.*?)</glue>'
        match_glue = re.search(pattern_glue, ans, re.DOTALL)
        print(f'ann:{ans}')
        iou = 0
        try:
            if match_answer:
                glue = match_glue.group(1)
                # import pdb; pdb.set_trace()
                # if is_valid_two_d_list_format(glue):
                pred_glue = ast.literal_eval(glue)
                iou = detection_iou(pred_glue, item["solution"])
            else:
                pred_glue = ast.literal_eval(ans)
                iou = detection_iou(pred_glue, item["solution"])
                # iou = 0.0
        except Exception as e:
            iou = 0
    
        # ious.append(iou)

        item_res = {'video_path': video_path, 'prompt':example_prompt, 'gt':item["solution"], 'pred':ans,  'iou':iou }
        append_to_jsonl(log_path, item_res)

        # pbar.set_postfix({"mIoU": sum(ious)/len(ious), 'accuracy': sum(accs)/len(accs)})

        # except Exception as e:
        #     print(f"Error processing {video_path}: {e}")

    print('=== final result ===')
    # if ious:
    print('mIoU:', sum(ious) / len(ious))
    print("Accuacy:", sum(accs)/len(accs))

    return ious, accs

def evaluate(data, video_root, slurm_procid, args):
    work_items = create_work_items(data, image_root=video_root)

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

    slurm_procid = int(os.environ.get('SLURM_PROCID', 0))  # 当前进程的全局 ID
    print(f"slurm_procid: {slurm_procid}")
    num_gpus = 8  # 假设总共有 8 块 GPU

    # data_chunks = split_data(data, num_gpus)
    # val_list_data = []
    # for dd in val_list:

        
    current_data_chunk = val_list[slurm_procid]
    # current_data_chunk = data_chunks[slurm_procid]
    gpu_count = torch.cuda.device_count()
    print(f"可用的 GPU 数量: {gpu_count}")
    assert gpu_count == num_gpus, gpu_count
    # import pdb;pdb.set_trace()
    evaluate(current_data_chunk, "pnorm2:s3://coco-caption/train2014/", slurm_procid, args)