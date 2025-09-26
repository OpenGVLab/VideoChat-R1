from decord import VideoReader, cpu
import numpy as np
import os
import sys
import datetime
import lmms_eval.tasks._task_utils.file_utils as file_utils
import json

import yaml
import random
from pathlib import Path

with open(Path(__file__).parent / "_default_template.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))


from loguru import logger as eval_logger



# Pass in video path here
# Can only work correctly with video llm
def video_caption_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    video_path = doc["video"]
    data_root = config["dataset_kwargs"]["cache_dir"]
    video_path = os.path.join(data_root, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    elif "s3://" not in video_path:
        sys.exit(f"video path:{video_path} does not exist, please check")

    return [video_path]


# This is the place where you format your question
def video_caption_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    # if "pre_prompt" in lmms_eval_specific_kwargs:
    #     pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if "post_prompt" in lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs["post_prompt"]

    # question = doc["caption"]
    
    return f"{post_prompt}"


def video_caption_doc_to_answer(doc):
    return doc["caption"]


# Process result for mcq answer generation
def video_caption_process_results_generation(doc, result):
    pred = result[0]
    return {"submission": {f'{doc["video"]}': {"pred":pred, "gt":doc["caption"]}}}


def video_caption_aggregate(results, args):
    video_caption_aggregate_submissions(results, args, "charades")
    
def video_caption_aggregate_submissions(results, args, task):
    now_date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    submission_file_name = f"inference_results_video_caption_{task}_{now_date_time}.json"
    path = file_utils.generate_submission_file(submission_file_name, args)

    # results is a list of 5031 dict,
    # need to convert results into a single dict with 5031 key-value pairs
    combined_submission = {}

    for submission_dict in results:
        combined_submission.update(submission_dict)

    with open(path, "w") as f:
        json.dump(combined_submission, f, indent=4)

    eval_logger.info(f"Submission file saved to {path}")






