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

GQA_TEMPLATE = """Answer the question: "[QUESTION]" according to the content of the video. Select the answer from :[OPTION].

Output your thought process within the <think> </think> tags, including analysis with either specific timestamps (xx.xx) or time ranges (xx.xx to xx.xx) in <timestep> </timestep> tags.

Then, provide your answer within the <answer> </answer> tags, output the corresponding letter of the option. At the same time, in the <glue> </glue> tags, present the precise time period in seconds of the video clips on which you base your answer to this question in the format of [(s1, e1), (s2, e2), ...]. For example: <answer>A</answer><glue>[(5.2, 10.4)]</glue>.
"""

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

def is_valid_two_d_list_format(s):
    pattern = r'^\[(\(\d+(\.\d+)?,\s*\d+(\.\d+)?\)(,\s*\(\d+(\.\d+)?,\s*\d+(\.\d+)?\))*(,)?|)\]$'
    if not re.match(pattern, s):
        return False
    try:
        # 尝试将字符串转换为 Python 对象
        lst = ast.literal_eval(s)
        # 检查对象是否为列表
        if not isinstance(lst, list):
            return False
        # 检查列表中的每个元素是否为元组
        for item in lst:
            if not isinstance(item, tuple):
                return False
            # 检查元组是否包含两个元素
            if len(item) != 2:
                return False
            # 检查元组中的元素是否为数字
            for num in item:
                if not isinstance(num, (int, float)):
                    return False
            if item[0] > item[1]: # 保证符合时序区间
                return False
        return True
    except:
        return False

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
    
    os.makedirs(f"./eval_logs/{model_base.replace('/', '-')}_gqa", exist_ok=True)
    log_path = f"./eval_logs/{model_base.replace('/', '-')}_gqa/{device}.jsonl"
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
    
    print('=== final result ===')
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
        f'{args.checkpoint_dir}_{slurm_procid}',
        args.resume
    )
    
    return ious, accs

if __name__=='__main__':
    args = get_args()

    # load data
    with open("your_base_dir/NextGQA/nextgqa_test.json", "r") as f:
        data = json.load(f)

    slurm_procid = int(os.environ.get('SLURM_PROCID', 0))  # 当前进程的全局 ID
    print(f"slurm_procid: {slurm_procid}")
    num_gpus = 8  # 假设总共有 8 块 GPU

    data_chunks = split_data(data, num_gpus)
    current_data_chunk = data_chunks[slurm_procid]
    gpu_count = torch.cuda.device_count()
    print(f"可用的 GPU 数量: {gpu_count}")
    assert gpu_count == num_gpus, gpu_count
    
    evaluate(current_data_chunk, "p2:s3://nextqa", slurm_procid, args)