from collections import defaultdict
import os
import datetime
import json
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
import io

from pathlib import Path
import yaml
import sys
from typing import List, Dict, Optional, Union
import re
import cv2
import numpy as np

from loguru import logger as eval_logger
from petrel_client.client import Client

client = Client(conf_path='~/petreloss.conf')


VIDEO_TYPE = ["short", "medium", "long"]
CATEGORIES = ["Knowledge", "Film & Television", "Sports Competition", "Artistic Performance", "Life Record", "Multilingual"]

SUB_CATEGORIES = [
    "Humanity & History",
    "Literature & Art",
    "Biology & Medicine",
    "Finance & Commerce",
    "Astronomy",
    "Geography",
    "Law",
    "Life Tip",
    "Technology",
    "Animation",
    "Movie & TV Show",
    "Documentary",
    "News Report",
    "Esports",
    "Basketball",
    "Football",
    "Athletics",
    "Other Sports",
    "Stage Play",
    "Magic Show",
    "Variety Show",
    "Acrobatics",
    "Handicraft",
    "Food",
    "Fashion",
    "Daily Life",
    "Travel",
    "Pet & Animal",
    "Exercise",
    "Multilingual",
]

TASK_CATEGORIES = [
    "Temporal Perception",
    "Spatial Perception",
    "Attribute Perception",
    "Action Recognition",
    "Object Recognition",
    "OCR Problems",
    "Counting Problem",
    "Temporal Reasoning",
    "Spatial Reasoning",
    "Action Reasoning",
    "Object Reasoning",
    "Information Synopsis",
]

replace_prompt = " Please answer yes or no."

# with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
#     raw_data = f.readlines()
#     safe_data = []
#     for i, line in enumerate(raw_data):
#         # remove function definition since yaml load cannot handle it
#         if "!function" not in line:
#             safe_data.append(line)

#     config = yaml.safe_load("".join(safe_data))

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
# cache_dir = os.path.join(hf_home, cache_dir)
# base_cache_dir = config["dataset_kwargs"]["cache_dir"]
# base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "videomme.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


def parse_subtitle_time(time_str):
    h, m, s_ms = time_str.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def load_subtitles(subtitle_path):
    subtitles = {}
    with open(subtitle_path, "r", encoding="utf-8") as file:
        content = file.read().split("\n\n")
        for section in content:
            if section.strip():
                lines = section.split("\n")
                if len(lines) >= 3:
                    time_range = lines[1].split(" --> ")
                    start_time = parse_subtitle_time(time_range[0])
                    end_time = parse_subtitle_time(time_range[1])
                    text = " ".join(line for line in lines[2:])
                    subtitles[(start_time, end_time)] = text
    return subtitles


def convert_time_to_frame(time_in_seconds, fps):
    return int(time_in_seconds * fps)


def extract_subtitles(video_path, subtitle_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    subtitles = load_subtitles(subtitle_path)

    subtitle_frames = []
    for (start_time, end_time), text in subtitles.items():
        start_frame = convert_time_to_frame(start_time, fps)
        end_frame = convert_time_to_frame(end_time, fps)
        subtitle_frames.append((start_frame, end_frame, text))

    return subtitle_frames, total_frame


def parse_subtitle_time(time_str):
    h, m, s_ms = time_str.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def load_subtitles(subtitle_path):
    subtitles = {}
    with open(subtitle_path, "r", encoding="utf-8") as file:
        content = file.read().split("\n\n")
        for section in content:
            if section.strip():
                lines = section.split("\n")
                if len(lines) >= 3:
                    time_range = lines[1].split(" --> ")
                    start_time = parse_subtitle_time(time_range[0])
                    end_time = parse_subtitle_time(time_range[1])
                    text = " ".join(line for line in lines[2:])
                    subtitles[(start_time, end_time)] = text
    return subtitles


def convert_time_to_frame(time_in_seconds, fps):
    return int(time_in_seconds * fps)


def extract_subtitles(video_path, subtitle_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    subtitles = load_subtitles(subtitle_path)

    subtitle_frames = []
    for (start_time, end_time), text in subtitles.items():
        start_frame = convert_time_to_frame(start_time, fps)
        end_frame = convert_time_to_frame(end_time, fps)
        subtitle_frames.append((start_frame, end_frame, text))

    return subtitle_frames, total_frame


def videomme_doc_to_visual(doc):
    # cache_dir = os.path.join(base_cache_dir, cache_name)
    cache_dir = cache_name
    video_path = doc["videoID"] + ".mp4"
    video_path = os.path.join(cache_dir, "videos", video_path)
    if os.path.exists(video_path):
        video_path = video_path
    elif os.path.exists(video_path.replace("mp4", "MP4")):
        video_path = video_path.replace("mp4", "MP4")
    elif os.path.exists(video_path.replace("mp4", "mkv")):
        video_path = video_path.replace("mp4", "mkv")
    elif 's3://' not in video_path:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


def videomme_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    option_prompt = "Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option."
    question = doc["question"]
    option = str(doc["options"])
    question = question + "\n" + option
    # pre_prompt = lmms_eval_specific_kwargs["pre_prompt"] if "pre_prompt" in lmms_eval_specific_kwargs else ""
    post_prompt = lmms_eval_specific_kwargs["post_prompt"] if "post_prompt" in lmms_eval_specific_kwargs else "The best answer is:"
    # full_prompt = pre_prompt + option_prompt + "\n" + question + "\n" + post_prompt
    full_prompt = option_prompt + "\n" + question + "\n" + post_prompt
    return full_prompt


def videomme_doc_to_text_internvl(doc, lmms_eval_specific_kwargs=None):
    """
    VLMEvalKit style
    """
    option_prompt = "These are the frames of a video. Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option."
    question = doc["question"]
    # option = str(doc["options"])
    option = "\n".join(doc['options'])
    question = "Question: " + question + "\n" + option
    post_prompt = lmms_eval_specific_kwargs["post_prompt"] if "post_prompt" in lmms_eval_specific_kwargs else "The best answer is:"
    full_prompt = option_prompt + "\n" + question  + post_prompt
    return full_prompt

def videomme_doc_to_text_vc(doc, lmms_eval_specific_kwargs=None):
    """
    VLMEvalKit style
    """
    option_prompt = "These are the frames of a video. Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option."
    question = doc["question"]
    # option = str(doc["options"])
    option = "\n".join(doc['options'])
    question = "Question: " + question + "\nOption:\n" + option
    post_prompt = lmms_eval_specific_kwargs["post_prompt"] if "post_prompt" in lmms_eval_specific_kwargs else "The best answer is:"
    full_prompt = option_prompt + "\n" + question  + post_prompt
    return full_prompt

def videomme_doc_to_text_think(doc, lmms_eval_specific_kwargs=None):

    question = doc["question"]
    # option = str(doc["options"])
    option = "\n".join(doc['options'])

    full_prompt = f"""Answer the question: "{question}" according to the content of the video. Select the answer from :{option}

Output your thought process within the <think> </think> tags, including analysis with either specific timestamps (xx.xx) or time ranges (xx.xx to xx.xx) in <timestep> </timestep> tags.

Then, provide your answer within the <answer> </answer> tags, output the corresponding letter of the option.
"""
    return full_prompt

def videomme_doc_to_text_think_glue(doc, lmms_eval_specific_kwargs=None):

    question = doc["question"]
    # option = str(doc["options"])
    option = "\n".join(doc['options'])

    full_prompt = f"""Answer the question: "{question}" according to the content of the video. Select the answer from :{option}

Output your thought process within the <think> </think> tags, including analysis with either specific timestamps (xx.xx) or time ranges (xx.xx to xx.xx) in <timestep> </timestep> tags.

Then, provide your answer within the <answer> </answer> tags, output the corresponding letter of the option. At the same time, in the <glue> </glue> tags, present the precise time period in seconds of the video clips on which you base your answer to this question in the format of [(s1, e1), (s2, e2), ...]. For example: <answer>A</answer><glue>[(5.2, 10.4)]</glue>.
"""
    return full_prompt


def videomme_doc_to_text_nothink_glue(doc, lmms_eval_specific_kwargs=None):

    question = doc["question"]
    # option = str(doc["options"])
    option = "\n".join(doc['options'])

    full_prompt = f"""Answer the question: "{question}" according to the content of the video. Select the answer from :{option}

Provide your answer within the <answer> </answer> tags, output the corresponding letter of the option. At the same time, in the <glue> </glue> tags, present the precise time period in seconds of the video clips on which you base your answer to this question in the format of [(s1, e1), (s2, e2), ...]. For example: <answer>A</answer><glue>[(5.2, 10.4)]</glue>.
"""
    return full_prompt

def videomme_doc_to_text_nothink(doc, lmms_eval_specific_kwargs=None):

    question = doc["question"]
    # option = str(doc["options"])
    option = "\n".join(doc['options'])

    full_prompt = f"""Answer the question: "{question}" according to the content of the video. Select the answer from :{option}

Provide your answer within the <answer> </answer> tags, output the corresponding letter of the option.
"""
    return full_prompt

def videomme_doc_to_text_vc2(doc, lmms_eval_specific_kwargs=None):
    """
    VLMEvalKit style
    """

    question = doc["question"]
    # option = str(doc["options"])
    option = "\n".join(doc['options'])
    question = "Question: " + question + "\nOption:\n" + option
    post_prompt = lmms_eval_specific_kwargs["post_prompt"] if "post_prompt" in lmms_eval_specific_kwargs else "The best answer is:"
    full_prompt = question + post_prompt
    return full_prompt


# Frames + Subs
# This video's subtitles are listed below:
# 【subtitles】

# Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option.
# 【question】
# The best answer is:
# Frames / Frames + Audio
# Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.
# 【question】
# The best answer is:

def videomme_doc_to_text_subtitle_vc(doc, lmms_eval_specific_kwargs=None):
    # cache_dir = os.path.join(base_cache_dir, cache_name)
    cache_dir = cache_name
    video_path = doc["videoID"] + ".mp4"
    subtitle_path = os.path.join(cache_dir, "subtitle", doc["videoID"] + ".srt")
    video_path = os.path.join(cache_dir, "videos", video_path)

    subtitle = ""
    subtitles_prompt = ""
    if os.path.exists(subtitle_path):  # Denote have subtitle
        subtitle = open(subtitle_path).readlines()
        subtitles_prompt = "This video's subtitles are listed below: \n"
    elif 's3://' in subtitle_path:
        _byte = client.get(subtitle_path)
        if _byte is not None and len(_byte) > 0:
            with io.BytesIO(client.get(subtitle_path)) as f:
                subtitle = f.readlines()
                subtitle = [sub.decode('utf-8') for sub in subtitle]
            subtitles_prompt = "This video's subtitles are listed below: \n"
    else:
        print(f"Can't find subtitle for: {subtitle_path}")
    if subtitle != "":
        if "gemini_api_flag" in lmms_eval_specific_kwargs:  # specific for gemini_api
            if lmms_eval_specific_kwargs["gemini_api_flag"] == "full subtitle":
                textlist = []
                for ele in subtitle:
                    pattern = r'<font color="white" size=".72c">(.*?)</font>'
                    matches = re.findall(pattern, ele)
                    if matches:
                        textlist.append(matches[0])
                subtitle_text = "\n".join(textlist)
        else:
            raise NotImplementedError("用全部subtitles即可")
            if "frame_num" in lmms_eval_specific_kwargs:
                frame_num = lmms_eval_specific_kwargs["frame_num"]
                subtitle_by_frame, total_frame = extract_subtitles(video_path, subtitle_path)
                uniform_sampled_frames = np.linspace(0, total_frame - 1, frame_num, dtype=int).tolist()

                subtitle_by_frame_idx = []
                for frame_idx in uniform_sampled_frames:
                    for idx, title in enumerate(subtitle_by_frame):
                        if frame_idx < title[1] and frame_idx >= title[0]:
                            subtitle_by_frame_idx.append(idx)
                subtitle_by_frame_idx = list(set(subtitle_by_frame_idx))

                textlist = []
                for idx in subtitle_by_frame_idx:
                    pattern = r'<font color="white" size=".72c">(.*?)</font>'
                    raw_text = re.findall(pattern, subtitle_by_frame[idx][2])
                    try:
                        textlist.append(raw_text[0])
                    except:
                        continue
                subtitle_text = "\n".join(textlist)
        subtitle = subtitle_text

    if subtitle != "":
        option_prompt = "These are the frames and subtitles of a video. Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option."
    else:
        option_prompt = "These are the frames of a video. Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option."
    question = doc["question"]
    # option = str(doc["options"])
    option = "\n".join(doc['options'])
    question = "Question: " + question + "\nOption:\n" + option
    full_prompt = subtitles_prompt + subtitle + "\n" + option_prompt + "\n" + question + "\n" + "The best answer is:"
    return full_prompt

def videomme_doc_to_text_subtitle_new(doc, lmms_eval_specific_kwargs=None):
    # cache_dir = os.path.join(base_cache_dir, cache_name)
    cache_dir = cache_name
    video_path = doc["videoID"] + ".mp4"
    subtitle_path = os.path.join(cache_dir, "subtitle", doc["videoID"] + ".srt")
    video_path = os.path.join(cache_dir, "videos", video_path)

    subtitle = ""
    subtitles_prompt = ""
    if os.path.exists(subtitle_path):  # Denote have subtitle
        subtitle = open(subtitle_path).readlines()
        subtitles_prompt = "This video's subtitles are listed below: \n"
    elif 's3://' in subtitle_path:
        _byte = client.get(subtitle_path)
        if _byte is not None and len(_byte) > 0:
            with io.BytesIO(client.get(subtitle_path)) as f:
                subtitle = f.readlines()
                subtitle = [sub.decode('utf-8') for sub in subtitle]
            subtitles_prompt = "This video's subtitles are listed below: \n"
    else:
        print(f"Can't find subtitle for: {subtitle_path}")
    if subtitle != "":
        if "gemini_api_flag" in lmms_eval_specific_kwargs:  # specific for gemini_api
            if lmms_eval_specific_kwargs["gemini_api_flag"] == "full subtitle":
                textlist = []
                for ele in subtitle:
                    pattern = r'<font color="white" size=".72c">(.*?)</font>'
                    matches = re.findall(pattern, ele)
                    if matches:
                        textlist.append(matches[0])
                subtitle_text = "\n".join(textlist)
        else:
            raise NotImplementedError("用全部subtitles即可")
            if "frame_num" in lmms_eval_specific_kwargs:
                frame_num = lmms_eval_specific_kwargs["frame_num"]
                subtitle_by_frame, total_frame = extract_subtitles(video_path, subtitle_path)
                uniform_sampled_frames = np.linspace(0, total_frame - 1, frame_num, dtype=int).tolist()

                subtitle_by_frame_idx = []
                for frame_idx in uniform_sampled_frames:
                    for idx, title in enumerate(subtitle_by_frame):
                        if frame_idx < title[1] and frame_idx >= title[0]:
                            subtitle_by_frame_idx.append(idx)
                subtitle_by_frame_idx = list(set(subtitle_by_frame_idx))

                textlist = []
                for idx in subtitle_by_frame_idx:
                    pattern = r'<font color="white" size=".72c">(.*?)</font>'
                    raw_text = re.findall(pattern, subtitle_by_frame[idx][2])
                    try:
                        textlist.append(raw_text[0])
                    except:
                        continue
                subtitle_text = "\n".join(textlist)
        subtitle = subtitle_text

    if subtitle != "":
        option_prompt = "These are the frames of a video. Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option."
    else:
        option_prompt = "These are the frames of a video. Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option."
    question = doc["question"]
    # option = str(doc["options"])
    option = "\n".join(doc['options'])
    question = "Question: " + question + "\n" + option
    full_prompt = subtitles_prompt + subtitle + "\n" + option_prompt + "\n" + question + "\n" + "The best answer is:"
    return full_prompt


def videomme_doc_to_text_subtitle(doc, lmms_eval_specific_kwargs=None):
    # cache_dir = os.path.join(base_cache_dir, cache_name)
    cache_dir = cache_name
    video_path = doc["videoID"] + ".mp4"
    subtitle_path = os.path.join(cache_dir, "subtitle", doc["videoID"] + ".srt")
    video_path = os.path.join(cache_dir, "videos", video_path)

    subtitle = ""
    subtitles_prompt = ""
    if os.path.exists(subtitle_path):  # Denote have subtitle
        subtitle = open(subtitle_path).readlines()
        subtitles_prompt = "This video's subtitles are listed below: \n"
    elif 's3://' in subtitle_path:
        _byte = client.get(subtitle_path)
        if _byte is not None and len(_byte) > 0:
            with io.BytesIO(client.get(subtitle_path)) as f:
                subtitle = f.readlines()
                subtitle = [sub.decode('utf-8') for sub in subtitle]
            subtitles_prompt = "This video's subtitles are listed below: \n"
    else:
        print(f"Can't find subtitle for: {subtitle_path}")
    if subtitle != "":
        if "gemini_api_flag" in lmms_eval_specific_kwargs:  # specific for gemini_api
            if lmms_eval_specific_kwargs["gemini_api_flag"] == "full subtitle":
                textlist = []
                for ele in subtitle:
                    pattern = r'<font color="white" size=".72c">(.*?)</font>'
                    matches = re.findall(pattern, ele)
                    if matches:
                        textlist.append(matches[0])
                subtitle_text = "\n".join(textlist)
        else:
            raise NotImplementedError("用全部subtitles即可")
            if "frame_num" in lmms_eval_specific_kwargs:
                frame_num = lmms_eval_specific_kwargs["frame_num"]
                subtitle_by_frame, total_frame = extract_subtitles(video_path, subtitle_path)
                uniform_sampled_frames = np.linspace(0, total_frame - 1, frame_num, dtype=int).tolist()

                subtitle_by_frame_idx = []
                for frame_idx in uniform_sampled_frames:
                    for idx, title in enumerate(subtitle_by_frame):
                        if frame_idx < title[1] and frame_idx >= title[0]:
                            subtitle_by_frame_idx.append(idx)
                subtitle_by_frame_idx = list(set(subtitle_by_frame_idx))

                textlist = []
                for idx in subtitle_by_frame_idx:
                    pattern = r'<font color="white" size=".72c">(.*?)</font>'
                    raw_text = re.findall(pattern, subtitle_by_frame[idx][2])
                    try:
                        textlist.append(raw_text[0])
                    except:
                        continue
                subtitle_text = "\n".join(textlist)
        subtitle = subtitle_text

    if subtitle != "":
        option_prompt = "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option."
    else:
        option_prompt = "Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option."
    question = doc["question"]
    option = str(doc["options"])
    question = question + "\n" + option
    full_prompt = subtitles_prompt + subtitle + "\n" + option_prompt + "\n" + question + "\n" + "The best answer is:"
    return full_prompt


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

    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""

    matches = re.search(r"[ABCD]", s)
    if matches is None:
        return ""
    return matches[0]


matrices = []

for i in VIDEO_TYPE:
    for j in CATEGORIES:
        for k in SUB_CATEGORIES:
            for l in TASK_CATEGORIES:
                matrices.append(f"{i}_{j}_{k}_{l}")


def videomme_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case videomme score), value: metric value
    """
    pred = results[0]
    pred_ans = extract_characters_regex(pred)
    # gt_ans = doc["answer"].lower().strip().replace(".", "")

    category = doc["domain"]
    sub_category = doc["sub_category"]
    task_category = doc["task_type"]
    data_dict = {"question_id": doc["question_id"], "duration": doc["duration"], "category": category, "sub_category": sub_category, "task_category": task_category, "pred_answer": pred_ans, "answer": doc["answer"]}

    # return {f"videomme_percetion_score": data_dict for metric in matrices}
    return {f"videomme_percetion_score": data_dict}


def videomme_process_results_reason(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case videomme score), value: metric value
    """
    pred = results[0]

    def find_solution(text):
        start_tag = '<|begin_of_solution|>'
        end_tag = '<|end_of_solution|>'

        start_index = text.find(start_tag)
        if start_index != -1:
            start_index += len(start_tag)
            end_index = text.find(end_tag, start_index)
            if end_index != -1:
                result = text[start_index:end_index].strip()
            else:
                result = text[start_index:].strip().split('\n')[0].strip()
        else:
            result = text.strip().split('\n')[-1].strip()
        return result
    
    pred_ans = find_solution(pred)
    pred_ans = extract_characters_regex(pred_ans)
    # gt_ans = doc["answer"].lower().strip().replace(".", "")

    category = doc["domain"]
    sub_category = doc["sub_category"]
    task_category = doc["task_type"]
    data_dict = {"question_id": doc["question_id"], "duration": doc["duration"], "category": category, "sub_category": sub_category, "task_category": task_category, "pred_answer": pred_ans, "answer": doc["answer"]}

    # return {f"videomme_percetion_score": data_dict for metric in matrices}
    return {f"videomme_percetion_score": data_dict}


def videomme_process_results_think(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case videomme score), value: metric value
    """
    pred = results[0]

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
            result = text.strip().split('\n')[-1].strip()
        return result
    
    pred_ans = find_solution(pred)
    pred_ans = extract_characters_regex(pred_ans)
    # gt_ans = doc["answer"].lower().strip().replace(".", "")

    category = doc["domain"]
    sub_category = doc["sub_category"]
    task_category = doc["task_type"]
    data_dict = {"question_id": doc["question_id"], "duration": doc["duration"], "category": category, "sub_category": sub_category, "task_category": task_category, "pred_answer": pred_ans, "answer": doc["answer"]}

    # return {f"videomme_percetion_score": data_dict for metric in matrices}
    return {f"videomme_percetion_score": data_dict}


def videomme_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    category2score = {}

    for video_type in VIDEO_TYPE:
        for category in CATEGORIES:
            for sub_category in SUB_CATEGORIES:
                for task_category in TASK_CATEGORIES:
                    key = f"{video_type}_{category}_{sub_category}_{task_category}"
                    category2score[key] = {"correct": 0, "answered": 0}

    for result in results:
        video_type = result["duration"]
        category = result["category"]
        sub_category = result["sub_category"]
        task_category = result["task_category"]
        key = f"{video_type}_{category}_{sub_category}_{task_category}"
        category2score[key]["answered"] += 1
        category2score[key]["correct"] += result["pred_answer"] == result["answer"]

    for video_type in VIDEO_TYPE:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if video_type in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        eval_logger.info(f"Evaluation on video Type: {video_type}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    for category in CATEGORIES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if category in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        eval_logger.info(f"Evaluation on Categories: {category}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    for sub_cate in SUB_CATEGORIES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if sub_cate in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        eval_logger.info(f"Evaluation on Video Sub Categories: {sub_cate}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    for task_cate in TASK_CATEGORIES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if task_cate in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        eval_logger.info(f"Evaluation on Task Categories: {task_cate}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")
        print((f"Evaluation on Task Categories: {task_cate}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%"))

    total_correct = 0
    total_answered = 0
    for k, v in category2score.items():
        total_correct += v["correct"]
        total_answered += v["answered"]
    eval_logger.info(f"Overall Performance: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")
    print(f"Overall Performance: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")
    return 100 * total_correct / total_answered if total_answered > 0 else 0
