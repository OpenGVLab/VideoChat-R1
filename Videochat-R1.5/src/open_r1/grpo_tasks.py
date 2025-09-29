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
# from qwen_vl_utils import process_vision_info
from my_vision_process import process_vision_info
from tqdm import tqdm
import torch
import json
import random
from petrel_client.client import Client
import ast
from torch.utils.data import ConcatDataset
from src.open_r1.reward_funcs import iou_glue_reward, answer_reward, format_reward_gqa, format_reward_igqa, iou_timestamp_reward, format_reward, pad, tracking_iou_reward, tracking_format_reward, detection_iou, detection_glue

from src.open_r1.dist_utils import init_distributed_mode

from petrel_client.client import Client
client = Client('~/petreloss.conf')
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'iou', 'format'.
    """
    
    reward_funcs: list[str] = field(
        default_factory=lambda: ["gqa_iou", "gqa_format", "gqa_reward","igqa_iou", "igqa_format", "igqa_reward", "tg_iou", "tg_format", "tg_pad", "detection_iou", "detection_format", "detection_pad", "mc_answer", "mc_format", "mc_pad"],
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
    training_data: Optional[str] = field(
        default="/mnt/petrelfs/yanziang/videoo1/TimeZero/dataset/cotraining.json",
        metadata={"help": "file of training data"},
    )



reward_funcs_registry = {
    "gqa_iou": iou_glue_reward, # Modified registry to use iou_timestamp_reward
    "gqa_reward": answer_reward,
    "gqa_format": format_reward_gqa,
    "igqa_iou": detection_glue, # Modified registry to use iou_timestamp_reward
    "igqa_reward": answer_reward,
    "igqa_format": format_reward_igqa,
    "tg_iou": iou_timestamp_reward, # Modified registry to use iou_timestamp_reward
    "tg_format": format_reward,
    "tg_pad": pad,
    "detection_iou": detection_iou,
    "detection_format": format_reward,
    "detection_pad": pad,
    "mc_answer" : answer_reward,
    "mc_format": format_reward,
    "mc_pad" : pad
    # "tracking_iou": tracking_iou_reward, # Modified registry to use iou_timestamp_reward
    # "tracking_format": tracking_format_reward,
    # "tracking_pad": pad,
}

class TTSdata(Dataset):   
    
    def get_row(self, idx):
        """
        获取第 idx 行的内容，并以字典形式返回。
        :param idx: 行索引（从 0 开始）
        :return: 字典形式的行数据
        """
        if not (0 <= idx < len(self._data)):
            raise IndexError(f"Index {idx} out of range for table with {len(self._data)} rows.")
        
        # 获取单行数据
        row_table = self._data.slice(idx, 1)  # 提取第 idx 行
        row_dict = row_table.to_pydict()     # 转换为字典
        
        # 将每列的值从列表中解包为单个值
        unpacked_row = {key: value[0] for key, value in row_dict.items()}
        return unpacked_row

    def __getitem__(self, idx): # Define getitem within the scope where dataset is available
        try:   
            if isinstance(idx, list):
                idx = idx[0]
            
            example = self.get_row(idx)
            data_to_return = {k: v for k, v in example.items()} # Create a copy to avoid modifying original dataset

            if 'video_path' in example:
                messages = [{"role": "user", "content": [{"type": "video", "video": example["video_path"], "total_pixels": 8192 * 28 * 28}]}]
                image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True, client=client)
                fps_inputs = video_kwargs['fps']
                # # data_to_return["image_inputs"] = [torch.load(os.path.join(example["video_path"][0], "image_inputs.pt"))]
                data_to_return["video_inputs"] = [video_inputs]
                # with open(os.path.join(example["video_path"][0], "video_kwargs.json"), 'r') as f:
                data_to_return["video_kwargs"] = [video_kwargs]
            elif 'image_path' in example:
                messages = [{"role": "user", "content": [{"type": "image", "image": example["image_path"]}]}]
                image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True, client=client)
                data_to_return["image_inputs"] = [image_inputs]
            print(data_to_return.keys())
            return data_to_return
        # data_to_return["use_preprocessed"] = [True] # Flag to indicate preprocessed data is used
        except Exception as e:
            print(f"Warning: Error loading preprocessed data from {example['video_path'][0]}, falling back to video_path. Error: {e}")
            # data_to_return["use_preprocessed"] = [False] # Fallback to video_path if loading fails
            print(idx)
            # idx = idx + 1
            idx =  random.randint(0, len(self._data))
            return self.__getitem__(idx)
        
        return data_to_return

def load_json_datasets(training_file):
    with open(training_file, 'r') as f:
        all_annos = json.load(f)

    all_datasets = []
    
    for key in all_annos:
        media_type = all_annos[key]['media_type']
        task_type = all_annos[key]['task_type']
        visual_base = all_annos[key]['visual_base']
        anno_path = all_annos[key]['json_path']
        
        examples = []
        with open(anno_path, 'r') as f:
            anno_data = json.load(f)
            for info in anno_data:
                if media_type == 'video':
                    video_path = os.path.join(visual_base, info['video'])
                    example = {
                        "problem": {"question":info['question']},
                        "video_path": video_path,
                        
                        "data_type": task_type
                    }
                    if 'duration' in info:
                        example['durations'] = info['duration']
                        
                elif media_type == 'image':
                    # import pdb; pdb.set_trace()
                    image_path = os.path.join(visual_base, info['image'])
                    example = {
                        "problem": {"question":info['question']},
                        "image_path": image_path,
                        "data_type": task_type
                    }
                    
                if 'options' in info:
                    example['problem']['options'] = info['options']
                if 'answer' in info:
                    example['answer'] = info['answer']
                if 'glue' in info:
                    example['glue'] = info['glue']
                    
                examples.append(example)
            random.shuffle(examples)
            
            batch_size =  int(os.environ["WORLD_SIZE"])
            num_batchs = len(examples) // batch_size
            examples = examples[:num_batchs * batch_size]
            print(len(examples))
            # print(examples[:5])
            dataset = TTSdata.from_list(examples)
            all_datasets.append(dataset)
            # import pdb; pdb.set_trace()
    return ConcatDataset(all_datasets)

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
    
    dataset = load_json_datasets(script_args.training_data)
    
    
    
    print(len(dataset))
    # import pdb; pdb.set_trace()
    print('-'*200)
    # import pdb; pdb.set_trace()
    print(dataset.__getitem__(10).keys())
    
    
    if not training_args.use_vllm:
        trainer_cls = Qwen2VLGRPOTrainer
    else:
        raise NotImplementedError
    
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    # import pdb; pdb.set_trace()
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
    init_distributed_mode()
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)