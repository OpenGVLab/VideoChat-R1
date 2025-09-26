from collections import defaultdict
import os
import datetime
import json
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from pathlib import Path
import yaml
import sys, string
from typing import List, Dict, Optional, Union
import re
import PIL
import numpy as np
from loguru import logger as eval_logger

import io
from petrel_client.client import Client

client = Client('~/petreloss.conf')

DATA_LIST = {
    "action_sequence_think": "p2:s3://star/Charades_v1_480/",
    "action_prediction_think": "p2:s3://star/Charades_v1_480/",
    "action_antonym_think": "p2:s3://ssv2-video/",
    "fine_grained_action_think": "p2hdd:s3://Moments_in_Time_Raw/videos/",
    "unexpected_action_think": "p2:s3://funqa-test/test/",
    "object_existence_think": "p2:s3://clevrer/video_validation/",
    "object_interaction_think": "p2:s3://star/Charades_v1_480/",
    "object_shuffle_think": "p2:s3://perception/videos/",
    "moving_direction_think": "p2:s3://clevrer/video_validation/",
    "action_localization_think": "p2:s3://sta/sta_video/",
    "scene_transition_think": "p2:s3://scene-qa/video/",
    "action_count_think": "p2:s3://perception/videos/",
    "moving_count_think": "p2:s3://clevrer/video_validation/",
    "moving_attribute_think": "p2:s3://clevrer/video_validation/",
    "state_change_think": "p2:s3://perception/videos/",
    "fine_grained_pose_think": "p2:s3://nturgbd/",
    "character_order_think": "p2:s3://perception/videos/",
    "egocentric_navigation_think": "p2:s3://vlnqa/",
    "episodic_reasoning_think": "p2:s3://tvqa/frames_fps3_hq/",
    "counterfactual_inference_think": "p2:s3://clevrer/video_validation/",
}

# hf_home = os.getenv("HF_HOME", "./~/.cache/huggingface")
# base_cache_dir = os.path.expanduser(hf_home)

with open(Path(__file__).parent / "_default_template.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)


cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


def mvbench_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    # cache_dir = os.path.join(base_cache_dir, cache_name)
    cache_dir = ""
    dataset_folder = DATA_LIST[lmms_eval_specific_kwargs["sub_task"]]
    video_path = os.path.join(cache_dir, dataset_folder, doc["video"])
    if os.path.exists(video_path):
        video_path = video_path
    elif os.path.basename(dataset_folder) in ["clevrer", "star"]:
        alternative_video_path = os.path.join(cache_dir, "data0613", dataset_folder, doc["video"])
        if os.path.exists(alternative_video_path):
            video_path = alternative_video_path
        else:
            eval_logger.error(f"Video path: {video_path} does not exist, please check.")
    elif "s3://" not in video_path:
        eval_logger.error(f"Video path: {video_path} does not exist, please check.")

    if "start" in doc:
        start, end = doc['start'], doc['end']
        media_dict = {'start':start, 'end':end, 'video_read_type': 'decord'}
    else:
        media_dict = {'video_read_type': 'decord'}

    
    return [video_path, media_dict]


def mvbench_frames_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    # cache_dir = os.path.join(base_cache_dir, cache_name)
    cache_dir = ""
    dataset_folder = DATA_LIST[lmms_eval_specific_kwargs["sub_task"]]
    video_path = os.path.join(cache_dir, dataset_folder, doc["video"])
    if os.path.exists(video_path):
        video_path = video_path
    elif os.path.basename(dataset_folder) in ["clevrer", "star"]:
        alternative_video_path = os.path.join(cache_dir, "data0613", dataset_folder, doc["video"])
        if os.path.exists(alternative_video_path):
            video_path = alternative_video_path
        else:
            eval_logger.error(f"Video path: {video_path} does not exist, please check.")
    elif "s3://" not in video_path:
        eval_logger.error(f"Video path: {video_path} does not exist, please check.")

    # frame_image_list = read_frame(video_path)
    if "start" in doc:
        start, end = doc['start'], doc['end']
        media_dict = {'start':start, 'end':end, 'video_read_type': 'img'}
    else:
        media_dict = {'video_read_type': 'img'}

    return [video_path, media_dict]


def mvbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    option_prompt = ""
    option_list = doc["candidates"]
    option_letters = string.ascii_uppercase

    if lmms_eval_specific_kwargs.get("yinan_prompt", "") == "()":
        for char_index, option in enumerate(option_list):
            option_letter = option_letters[char_index]
            option_prompt += f"({option_letter}) {option}\n"

        full_text = "Question: " + doc["question"] + "\nOption:\n" + option_prompt + lmms_eval_specific_kwargs["post_prompt"]
        
    else:
        raise NotImplementedError(lmms_eval_specific_kwargs)
        for char_index, option in enumerate(option_list):
            option_letter = option_letters[char_index]
            option_prompt += f"{option_letter}. {option}\n"

        full_text = lmms_eval_specific_kwargs.get("pre_prompt", "") + doc["question"] + "\n" + option_prompt + lmms_eval_specific_kwargs["post_prompt"]

    return full_text

def mvbench_doc_to_text_think(doc, lmms_eval_specific_kwargs=None):
    option_prompt = ""
    option_list = doc["candidates"]
    option_letters = string.ascii_uppercase

    for char_index, option in enumerate(option_list):
        option_letter = option_letters[char_index]
        option_prompt += f"({option_letter}) {option}\n"


    full_prompt = f"""Answer the question: "{doc["question"]}" according to the content of the video. Select the answer from :{option_prompt}
    
Output your thought process within the <think> </think> tags, including analysis with either specific timestamps (xx.xx) or time ranges (xx.xx to xx.xx) in <timestep> </timestep> tags.

Then, provide your answer within the <answer> </answer> tags, output the corresponding letter of the option.
"""
    return full_prompt

def mvbench_doc_to_text_nothink(doc, lmms_eval_specific_kwargs=None):
    option_prompt = ""
    option_list = doc["candidates"]
    option_letters = string.ascii_uppercase

    for char_index, option in enumerate(option_list):
        option_letter = option_letters[char_index]
        option_prompt += f"({option_letter}) {option}\n"



    full_prompt = f"""Answer the question: "{doc["question"]}" according to the content of the video. Select the answer from :{option_prompt}
    
Provide your answer within the <answer> </answer> tags, output the corresponding letter of the option.
"""
    return full_prompt

def mcq_acc(answer, pred):
    periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
    commaStrip = re.compile("(\d)(\,)(\d)")
    punct = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]

    def processPunctuation(inText):
        outText = inText
        for p in punct:
            if (p + " " in inText or " " + p in inText) or (re.search(commaStrip, inText) != None):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = periodStrip.sub("", outText, re.UNICODE)
        return outText

    def process(answer):
        option_regex = re.compile(r"^([A-E])\.\s*(.+)$", re.IGNORECASE)
        match = option_regex.match(answer.strip())

        if match:
            # If matched, return the option letter in uppercase
            return match.group(1).upper()
        else:
            # If no match, process the answer as before
            answer = answer.replace("\n", " ")
            answer = answer.replace("\t", " ")
            answer = answer.strip()
            answer = processPunctuation(answer)
            answer = answer.strip("'")
            answer = answer.strip('"')
            answer = answer.strip(")")
            answer = answer.strip("(")
            answer = answer.strip().lower()

            # Try to find any single letter (A-E) in the processed answer
            letter_match = re.search(r"\b([A-E])\b", answer, re.IGNORECASE)
            if letter_match:
                return letter_match.group(1).upper()

            return answer

    pred = process(pred)
    answer = process(answer)

    if pred == answer:
        score = 1
    else:
        score = 0

    return score


def mvbench_process_results_answer(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case mvbench_perception_score), value: metric value
    """
    origin_pred = results[0]

    def find_solution(text):
        start_tag = '<answer>'
        end_tag = '</answer>'

        start_index = text.find(start_tag)
        if start_index != -1:
            start_index += len(start_tag)
            end_index = text.find(end_tag, start_index)
            if end_index != -1:
                result = text[start_index:end_index].strip()
            else:
                result = text[start_index:].strip().split('\n')[0].strip()
        else:
            result = text

        return result
    
    pred = find_solution(origin_pred)

    # Calculate the ground truth option letter
    option_letters = string.ascii_uppercase
    gt_option_letter = None
    for i, candidate in enumerate(doc["candidates"]):
        if candidate == doc["answer"]:
            gt_option_letter = option_letters[i]
            break

    if gt_option_letter is not None:
        # Calculate the score using mcq_acc function
        score = mcq_acc(gt_option_letter, pred)
    else:
        score = 0

    data_dict = {"pred_answer": origin_pred, "gt_answer": gt_option_letter, "score": score}

    return {"mvbench_accuracy": data_dict}


def mvbench_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    total_answered = 0
    total_correct = 0
    for result in results:
        if result["pred_answer"] != "":
            total_answered += 1
            total_correct += result["score"]

    return 100 * total_correct / total_answered if total_answered > 0 else 0
