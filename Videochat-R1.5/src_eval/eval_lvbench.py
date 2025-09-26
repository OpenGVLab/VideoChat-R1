# from data_configs import DATASETS
import argparse
import numpy as np
import json
from tqdm import tqdm
import os
import re
import pickle
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
# from qwen_vl_utils import process_vision_info
from my_vision_process import process_vision_info
import random
import ast
from petrel_client.client import Client
import os
import json
from math import ceil
import torch.distributed as dist

def split_data(data, num_gpus):
    """
    将数据交错分割为 num_gpus 块。
    如果数据量不能被 num_gpus 整除，后面的块会包含较少元素。
    如果数据是字典，则返回的每个块也是字典。
    """
    # 记录原始数据类型
    is_dict = isinstance(data, dict)

    # 转换为可索引结构
    if is_dict:
        keys = list(data.keys())
        values = list(data.values())
    elif isinstance(data, list):
        keys = None
        values = data
    else:
        values = list(data)
        keys = None

    data_size = len(values)
    
    # 初始化空列表
    chunks = [[] for _ in range(num_gpus)]

    # 交错填充
    for i, value in enumerate(values):
        gpu_idx = i % num_gpus
        if keys is not None:  # 如果是字典，保存 key-value 对
            chunks[gpu_idx].append((keys[i], value))
        else:  # 否则只保存值
            chunks[gpu_idx].append(value)

    # 如果原始数据是字典，将每个块转换回字典
    if is_dict:
        chunks = [dict(chunk) for chunk in chunks]
    
    return chunks

conf_path = '~/petreloss.conf'
client = Client(conf_path)

VIDEO_INFO_CACHE = {}

def get_args():
    parser = argparse.ArgumentParser(description='Evaluation for training-free video temporal grounding (Single GPU Version)')
    

    parser.add_argument("--model_base", type=str, default="/path/to/model")
    parser.add_argument("--checkpoint_dir", type=str, default="/path/to/out_dir")
    parser.add_argument("--num_percptions", type=int, default=1)
    
    # num_percptions
    return parser.parse_args()

def inference(video_path, prompt, model, processor, max_new_tokens=2048, device="cuda:0", client = None, pred_glue=None):
    messages = [
        {"role": "user", "content": [
                {"type": "video", 
                "video": video_path,
                'key_time':pred_glue,
                "total_pixels": 128*6 * 28 * 28, 
                "min_pixels": 128 * 28 * 28,
                },
                {"type": "text", "text": prompt},
            ]
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True, client = client)
    fps_inputs = video_kwargs['fps']

    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, fps=fps_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)

    generated_ids = [output_ids[i][len(inputs.input_ids[i]):] for i in range(len(output_ids))]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    torch.cuda.empty_cache()
    return output_text[0]
QA_THINK_GLUE = """Answer the question: "[QUESTION]" according to the content of the video. Select the answer from :[OPTION].

Output your think process within the  <think> </think> tags.

Then, provide your answer within the <answer> </answer> tags, output the corresponding letter of the option. At the same time, in the <glue> </glue> tags, present the precise time period in seconds of the video clips on which you base your answer to this question in the format of [(s1, e1), (s2, e2), ...]. For example: <think>...</think><answer>A</answer><glue>[(5.2, 10.4)]</glue>.
"""

QA = """Answer the question: "[QUESTION]" according to the content of the video. Select the answer from :[OPTION].

Then, provide your answer within the <answer> </answer> tags, output the corresponding letter of the option. For example: <answer>A</answer>.
"""

def create_work_items(data, video_root):
    examples = []
    for i, info in enumerate(data):
        video_path = os.path.join(video_root, info['video'])

        media_dict = {'video_read_type': 'img'}
        example = {
            "problem": {"question":info['question'], "options":info['candidates']},
            "solution": {"answer":info['answer']},
            "video_path": [video_path, media_dict],
            "video_id": info['video']
        }

        examples.append(example)
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

def process_work_items(work_items, model_base, device, checkpoint_dir, num_percptions, client=None):
    model, processor = setup_model(model_base, device)

    log_path = f"{checkpoint_dir}_{device}.jsonl"
    # print(log_path)
    pbar = tqdm(work_items)
    
    for idx, item in enumerate(pbar):
        pred_glue = None
        answers = []
        accs = []
        for percption in range(num_percptions):    
            
            video_path = item['video_path']

            if percption == num_percptions - 1 :
                example_prompt = QA_THINK_GLUE.replace("[QUESTION]", item["problem"]["question"])
            else:
                example_prompt = QA.replace("[QUESTION]", item["problem"]["question"])
            
            option = ''
            for ii, op in enumerate(item["problem"]["options"]):
                option += op + '\n'
            
            prompt = example_prompt.replace("[OPTION]", option)


            # try:
            ans = inference(video_path, prompt, model, processor, device=device, client=client, pred_glue=pred_glue)

            pattern_answer = r'<answer>(.*?)</answer>'
            match_answer = re.search(pattern_answer, ans, re.DOTALL)

            acc = 0.0
            
            converted_answer = item['solution']["answer"]

            if match_answer:
                answer = match_answer.group(1)
                if extract_characters_regex(answer) == extract_characters_regex(converted_answer):
                    acc = 1.0

            accs.append(acc)

            # IoU

            pattern_glue = r'<glue>(.*?)</glue>'
            match_glue = re.search(pattern_glue, ans, re.DOTALL)
            # print(f'ann:{ans}')
            answers.append(ans)
            
            try:
                if match_glue:
                    glue = match_glue.group(1)
                    pred_glue = ast.literal_eval(glue)
                
                
            except Exception as e:
                # iou = 0
                pred_glue = None

        

        item_res = {'video_path': video_path, 'prompt':prompt, 'gt':item["solution"], 'pred':answers, 'acc':accs }
        append_to_jsonl(log_path, item_res)

    del model, processor
    return accs

def evaluate(data, video_root, slurm_procid, args):
    work_items = create_work_items(data, video_root=video_root)

    accs = process_work_items(
        work_items, 
        args.model_base, 
        f'cuda:{slurm_procid}', 
        f'{args.checkpoint_dir}_{slurm_procid}',
        num_percptions = args.num_percptions,
        client= client
    )

    return accs



if __name__=='__main__':
    args = get_args()


    slurm_procid = int(os.environ.get('SLURM_PROCID', 0)) 
    print(f"slurm_procid: {slurm_procid}")
    num_gpus = 8
    gpu_count = torch.cuda.device_count()
    print(f"可用的 GPU 数量: {gpu_count}")
    assert gpu_count == num_gpus, gpu_count
    
    paths = 'Videochat-R1.5/src_eval/jsons/LVBench'
    root = 'pnorm2:s3://LVBench/frames/'

    for dd in os.listdir(paths):
        path = os.path.join(paths, dd)
        with open(path, 'r') as f:
            data = json.load(f)

        data_chunks = split_data(data, num_gpus)
        current_data_chunk = data_chunks[slurm_procid]

        acc = evaluate(current_data_chunk, root, slurm_procid, args)
        print(acc)