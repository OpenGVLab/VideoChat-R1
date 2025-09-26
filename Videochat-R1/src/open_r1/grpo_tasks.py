# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from src.open_r1.trainer import Qwen2VLGRPOTrainer_Video_Tasks as Qwen2VLGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from src.open_r1.trainer.yza_vision_process import process_vision_info
from tqdm import tqdm
import torch
import json
import random
import ast
from torch.utils.data import ConcatDataset

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'iou', 'format'.
    """
    reward_funcs: list[str] = field(
        default_factory=lambda: ["gqa_iou", "gqa_format", "gqa_reward", "tg_iou", "tg_format", "tg_pad", "tracking_iou", "tracking_format", "tracking_pad"],
        metadata={"help": "List of reward functions. Possible values: 'iou', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )

    train_data_path_tg: str = field(
        default="/your_root/Annotations/Charades/charades_annotation/train.json",
        metadata={"help": "Path to the training data JSON file."},
    )
    
    train_data_path_gqa: str = field(
        default="/your_root/Annotations/NextGQA/nextgqa_val.json",
        metadata={"help": "Path to the training data JSON file."},
    )
    
    train_data_path_tracking: str = field(
        default="/your_root/Annotations/Got/got_train.json",
        metadata={"help": "Path to the training data JSON file."},
    )

    video_folder_tg: str = field(
        default="/Your_video_path/Charades_v1",  # Replace with your actual video folder path
        metadata={"help": "Path to the folder containing video files."},
    )
    
    
    video_folder_gqa: str = field(
        default="/Your_video_path/NextGQA",  # Replace with your actual video folder path
        metadata={"help": "Path to the folder containing video files."},
    )
    
    video_folder_tracking: str = field(
        default="/Your_video_path/GOT",  # Replace with your actual video folder path
        metadata={"help": "Path to the folder containing video files."},
    )



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
        

def iou_glue_reward(completions, solution, durations, **kwargs): # Modified reward function name and arguments
    """Reward function that calculates IoU between predicted and ground truth timestamps."""

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


    rewards = []
    # print(completions, solution, durations, **kwargs)
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol, duration in zip(completions, solution, durations): # Added video_durations
        reward = 0.0

        gt_glue = sol['glue']

        pattern_glue = r'<glue>(.*?)</glue>'
        match_glue = re.search(pattern_glue, content, re.DOTALL)

        if match_glue:
            glue = match_glue.group(1)
            if is_valid_two_d_list_format(glue):
                pred_glue = ast.literal_eval(glue)
                reward = compute_iou(pred_glue, gt_glue)
        else:
            reward = 0.0


        rewards.append(reward)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"Content: {content}\n")
                if reward > 0:
                    f.write(f"pred glue: {pred_glue}\n")
                f.write(f"gt glue: {gt_glue}\n")
                f.write(f"------------- {current_time} IoU reward: {reward} -------------\n") # Modified log message

    return rewards

def answer_reward(completions, solution, **kwargs): # Modified reward function name and arguments
    """Reward function that calculates IoU between predicted and ground truth timestamps."""

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
    
    rewards = []

    for content, sol in zip(completions, solution): 
        reward = 0.0
        
        pattern_answer = r'<answer>(.*?)</answer>'

        # 使用 search 方法查找首个匹配项
        match_answer = re.search(pattern_answer, content, re.DOTALL)

        if match_answer:
            # 获取捕获组中的内容
            answer = match_answer.group(1)
            if extract_characters_regex(answer) == extract_characters_regex(sol['answer']):
                reward = 1.0

        rewards.append(reward)

    return rewards


def format_reward_gqa(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = re.compile(r'<think>.*?</think>\s*<answer>.*?</answer>\s*<glue>.*?</glue>', re.DOTALL)
    matches = [re.fullmatch(pattern, content.strip()) for content in completions]
    # print('matches:', matches)
    print(completions[0])
    reward_list = []
    for i, match in enumerate(matches):
        if match:
            pattern_glue = r'<glue>(.*?)</glue>'

            # 使用 search 方法查找首个匹配项
            match_glue = re.search(pattern_glue, completions[i], re.DOTALL)

            if match_glue:
                # 获取捕获组中的内容
                glue = match_glue.group(1)
            else:
                raise ValueError(completions[i])

            if is_valid_two_d_list_format(glue):
                r = 1.0
            else:
                r = 0.0
        else:
            r = 0.0
        reward_list.append(r)
    return reward_list



def parse_timestamp_output(output_string):
    """Parses timestamp output, similar to the example code."""
    # 1. Find all <answer>...</answer> blocks.
    answer_matches = re.findall(r"<answer>(.*?)</answer>", output_string, re.DOTALL)

    if not answer_matches:
        return None  # No <answer> tags found.

    # 2. Use the content of the *last* <answer> block.
    last_answer_content = answer_matches[-1]
    print('last_answer_content:', last_answer_content)

    matches = re.findall(r"(\d+\.?\d*) (to|and) (\d+\.?\d*)", last_answer_content, re.IGNORECASE)
    if not matches:
        return None
    last_match = matches[-1]
    start_time = float(last_match[0])
    end_time = float(last_match[2])
    return start_time, end_time


def iou_timestamp_reward(completions, solution, durations, **kwargs): # Modified reward function name and arguments
    """Reward function that calculates IoU between predicted and ground truth timestamps."""
    # print(completions, solution, durations)
    # contents = [completion[0]["content"] for completion in completions]
    rewards = []
    # print(completions, solution, durations, **kwargs)
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol, duration in zip(completions, solution, durations): # Added video_durations
        reward = 0.0
        parsed_times = parse_timestamp_output(content)
        start_time, end_time = 0, 0
        gt_start, gt_end = sol
        # s, e = gt_start / duration, gt_end / duration
        s, e = gt_start, gt_end
        if parsed_times:
            start_time, end_time = parsed_times
            from_number = start_time
            to_number = end_time

            intersection = max(0, min(to_number, e) - max(from_number, s))
            union = max(to_number, e) - min(from_number, s)
            if union > 0:
                iou = intersection / union   # 0.1 0.3

            reward = iou

        print('gt second:', gt_start, gt_end)
        print('pred second:', start_time, end_time)
        print(f"------------- {current_time} IoU reward: {reward} -------------\n")

        rewards.append(reward)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"Content: {content}\n")
                f.write(f"pred second: {str(start_time)}, {str(end_time)}\n")
                f.write(f"gt second: {str(gt_start)}, {str(gt_end)}\n")
                f.write(f"------------- {current_time} IoU reward: {reward} -------------\n") # Modified log message

    return rewards


def format_reward_tg(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = re.compile(r'<think>.*?</think>\s*<answer>.*?</answer>', re.DOTALL)
    matches = [re.fullmatch(pattern, content.strip()) for content in completions]
    print(completions[0])
    print('matches:', matches)
    return [1.0 if match else 0.0 for match in matches]

def pad(completions, **kwargs):
    """Return 0"""
    return [0 for content in completions]


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


def tracking_iou_reward(completions, solution, **kwargs):
    
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
    
    rewards = []
    for content, sol in zip(completions, solution): # Added video_durations
        content = content.replace('\n', '')
        reward = 0.0

        gt_boxes = sol['answer']

        pattern_glue = r'<answer>(.*?)</answer>'
        match_glue = re.search(pattern_glue, content, re.DOTALL)
        
        if match_glue:
            try:
                glue = match_glue.group(1)
                if is_valid_list_of_lists(glue):
                    pred_glue = ast.literal_eval(glue)
                    reward = average_overlap(pred_glue, gt_boxes)
            except Exception as e:
                reward = 0
        else:
            reward = 0.0


        rewards.append(reward)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"Content: {content}\n")
                if reward > 0:
                    f.write(f"pred glue: {pred_glue}\n")
                f.write(f"gt glue: {gt_boxes}\n")
                f.write(f"------------- IoU reward: {reward} -------------\n") # Modified log message

    return rewards

def tracking_format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    # pattern = re.compile(r'<think>.*?</think>\s*<answer>.*?</answer>', re.DOTALL)
    pattern = re.compile(r'<think>.*?</think>\s*<answer>.*?</answer>', re.DOTALL)
    matches = [re.fullmatch(pattern, content.strip()) for content in completions]
    # print('matches:', matches)
    reward_list = []
    for i, match in enumerate(matches):
        if match:
            pattern_glue = r'<answer>(.*?)</answer>'

            # 使用 search 方法查找首个匹配项
            match_glue = re.search(pattern_glue, completions[i], re.DOTALL)

            if match_glue:
                # 获取捕获组中的内容
                glue = match_glue.group(1)
            else:
                raise ValueError(completions[i])

            
            if is_valid_list_of_lists(glue):
                r = 1.0
            else:
                r = 0.0
        else:
            r = 0.0
        reward_list.append(r)
    return reward_list




reward_funcs_registry = {
    "gqa_iou": iou_glue_reward, # Modified registry to use iou_timestamp_reward
    "gqa_reward": answer_reward,
    "gqa_format": format_reward_gqa,
    "tg_iou": iou_timestamp_reward, # Modified registry to use iou_timestamp_reward
    "tg_format": format_reward_tg,
    "tg_pad": pad,
    "tracking_iou": tracking_iou_reward, # Modified registry to use iou_timestamp_reward
    "tracking_format": tracking_format_reward,
    "tracking_pad": pad,
}



def load_json_dataset_gqa(train_data_path, video_folder):#, preprocessed_data_path=None): # Modified to accept preprocessed_data_path
    def create_dataset_from_json(file_path, split_name):
        with open(file_path, 'r', encoding="utf-8") as f:
            data = json.load(f)
        examples = []
        for info in data:
            video_path = os.path.join(video_folder, info['video'])

            example = {
                "problem": {"question":info['question'], "options":info['options']},
                "solution": {"answer":info['answer'], "glue":info['glue']},
                "video_path": video_path,
                "durations": info['duration'],
                "data_type": 'gqa'
            }

            examples.append(example)

        random.shuffle(examples)
        print(len(examples))
        print(examples[:5])
        dataset = Dataset.from_list(examples)
        # import pdb; pdb.set_trace()

        return dataset

    train_dataset = create_dataset_from_json(train_data_path, "train")
    # eval_dataset = create_dataset_from_json(eval_data_path, "eval")
    return train_dataset

def load_json_dataset_tg(train_data_path, video_folder): # Modified to accept preprocessed_data_path
    def create_dataset_from_json(file_path, split_name):
        with open(file_path, 'r') as f:
            data = json.load(f)
        examples = []
        for video_id, video_data in tqdm(data.items()):
            for sentence_id, (timestamps, sentence) in enumerate(zip(video_data['timestamps'], video_data['sentences'])):
                sentence = sentence.strip().lower()
                if sentence.endswith("."):
                    sentence = sentence[:-1]
                video_filename_base = video_id
                video_path = None
                # for ext in ['mp4', 'mkv', 'webm']:
                candidate_path = os.path.join(video_folder, f"{video_filename_base}.mp4")
                    # if os.path.isfile(candidate_path):
                video_path = candidate_path
                    #     break
                if video_path is None:
                    print(f"Warning: Video file not found for ID: {video_id}")
                    continue
                example = {
                    "problem": sentence,
                    "solution": (timestamps[0], timestamps[1]),
                    "video_path": video_path,
                    "durations": video_data['duration'],
                    "data_type": "tg" # Initialize video_path as None
                }
                examples.append(example)

        random.shuffle(examples)
        print(len(examples))
        print(examples[:5])
        dataset = Dataset.from_list(examples)

        example = dataset[[0]]
        return dataset
    train_dataset = create_dataset_from_json(train_data_path, "train")
    return train_dataset


def load_json_dataset_tracking(train_data_path, video_folder):#, preprocessed_data_path=None): # Modified to accept preprocessed_data_path
    def create_dataset_from_json(file_path, split_name):
        with open(file_path, 'r', encoding="utf-8") as f:
            data = json.load(f)
        examples = []


        for info in data:
            
            path = info['path']
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
            step = (nframes - 1) / 6  
            middle_indices = [int(i * step) for i in range(1, 6)]
            middle_elements = [sorted_files[i] for i in middle_indices]
            
            result = [first_element] + middle_elements + [last_element]
            # video_path = os.path.join(video_folder, result)
            # video_path = video_path + '/'
            example = {
                "problem": {"object":info['object'], "start":info['gt'][0]},
                "solution": {"answer":info['gt']},
                "video_path": result,
                "durations": 0,
                "data_type": "tracking" # Initialize video_path as None
                # "durations": info['duration'],
            }

            examples.append(example)

        random.shuffle(examples)
        print(len(examples))
        print(examples[:5])
        dataset = Dataset.from_list(examples)
        return dataset
    train_dataset = create_dataset_from_json(train_data_path, "train")
    # eval_dataset = create_dataset_from_json(eval_data_path, "eval")
    return train_dataset

def main(script_args, training_args, model_args):
    # Get reward functionsok
    # reward_funcs = 
    reward_funcs = {}
    for func in script_args.reward_funcs:
        task_type = func.split('_')[0]
        if task_type in reward_funcs:
            reward_funcs[task_type].append(reward_funcs_registry[func])
        else:
            reward_funcs[task_type] = [reward_funcs_registry[func]]
        
    # reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # # Load the dataset
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    # Load the dataset, now handles both raw and preprocessed data
    dataset_tg = load_json_dataset_tg(
        script_args.train_data_path_tg,
        None,
        script_args.video_folder_tg,
        # script_args.preprocessed_data_path # Pass preprocessed_data_path
    )
    dataset_gqa = load_json_dataset_gqa(
        script_args.train_data_path_gqa,
        None,
        script_args.video_folder_gqa,
        # script_args.preprocessed_data_path # Pass preprocessed_data_path
    )
    dataset_tracking = load_json_dataset_tracking(
        script_args.train_data_path_tracking,
        None,
        script_args.video_folder_tracking,
        # script_args.preprocessed_data_path # Pass preprocessed_data_path
    )
    # 创建 ConcatDataset
    dataset = ConcatDataset([dataset_tg, dataset_gqa, dataset_tracking])
    def __getitem__(self, idx): # Define getitem within the scope where dataset is available
        try:   
            example = dataset[idx]
            data_to_return = {k: v for k, v in example.items()} # Create a copy to avoid modifying original dataset
        
        except Exception as e:
            print(f"Warning: Error loading preprocessed data from {example['video_path'][0]}, falling back to video_path. Error: {e}")
            data_to_return["use_preprocessed"] = [False] # Fallback to video_path if loading fails
            print(idx)
            idx = idx + 1
            return self.__getitem__(idx)
        
        return data_to_return

    dataset.__getitem__ = __getitem__.__get__(dataset, Dataset) # Bind getitem to the dataset
    print(len(dataset))
    
    print(dataset.__getitem__(10).keys())
    
    if not training_args.use_vllm:
        trainer_cls = Qwen2VLGRPOTrainer
    else:
        raise NotImplementedError
    
    print("using: ", trainer_cls)


    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)