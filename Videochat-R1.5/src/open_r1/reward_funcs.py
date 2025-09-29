
import re
import ast
from datetime import datetime
import os
import numpy as np
import logging
logger = logging.getLogger(__name__)

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
        if len(lst) > 3:
            return False
        return True
    except:
        return False
        

def iou_glue_reward(completions, glue, durations, **kwargs): # Modified reward function name and arguments
    """Reward function that calculates IoU between predicted and ground truth timestamps."""
    solution = glue
    # import pdb; pdb.set_trace()
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
        # if len(sorted_intervals) > 3:
        #     merged = [[0, 0]]
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

    # # 示例用法
    # list_a = [[0, 3], [2, 4], [22, 25]]
    # list_b = [[1, 5], [2, 2], [2, 4]]
    # iou = compute_iou(list_a, list_b)

    rewards = []
    # print(completions, solution, durations, **kwargs)
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol, duration in zip(completions, solution, durations): # Added video_durations
        reward = 0.0

        gt_glue = sol

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

def answer_reward(completions, answer, **kwargs): # Modified reward function name and arguments
    """Reward function that calculates IoU between predicted and ground truth timestamps."""
    solution = answer
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
            if extract_characters_regex(answer) == extract_characters_regex(sol):
                reward = 1.0
            # print(f'answer:{answer}, {sol}')

        rewards.append(reward)
    print(f'content:{content}, sol:{sol}, reward:{reward}, rewards:{rewards}')
    return rewards


def format_reward_gqa(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    # pattern = re.compile(r'<sum>(.{11,}?)</sum>\s*<answer>.*?</answer>\s*<glue>.*?</glue>', re.DOTALL)
    
    pattern = re.compile(r'<answer>.*?</answer>\s*<glue>.*?</glue>', re.DOTALL)
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

def is_valid_bbox(s):
    """
    判断字符串是否符合 [x0, y0, x1, y1] 格式。
    
    支持整数、浮点数、负数、空格。
    返回布尔值。
    """
    try:
        if isinstance(eval(s), list) and (len(eval(s)) == 4):
            return 1
        return 0
    except Exception as e:
        return 0
def format_reward_igqa(completions, **kwargs):
    # import pdb; pdb.set_trace()
    """Reward function that checks if the completion has a specific format."""
    # pattern = re.compile(r'<sum>(.{11,}?)</sum>\s*<answer>.*?</answer>\s*<glue>.*?</glue>', re.DOTALL)
    
    pattern = re.compile(r'<answer>.*?</answer>\s*<glue>.*?</glue>', re.DOTALL)
    matches = [re.fullmatch(pattern, content.strip()) for content in completions]
    # print('matches:', matches)
    print(completions[0])
    reward_list = []
    for i, match in enumerate(matches):
        if match:
            pattern_glue = r'<glue>(.*?)</glue>'

            match_glue = re.search(pattern_glue, completions[i], re.DOTALL)

            if match_glue:
                # 获取捕获组中的内容
                glue = match_glue.group(1)
                if is_valid_bbox(glue):
                    r = 1.0
                else:
                    r = 0.0
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
    # print('last_answer_content:', last_answer_content)

    matches = re.findall(r"(\d+\.?\d*) (to|and) (\d+\.?\d*)", last_answer_content, re.IGNORECASE)
    if not matches:
        return None
    last_match = matches[-1]
    start_time = float(last_match[0])
    end_time = float(last_match[2])
    return start_time, end_time


def iou_timestamp_reward(completions, glue, durations, **kwargs): # Modified reward function name and arguments
    """Reward function that calculates IoU between predicted and ground truth timestamps."""
    # import pdb; pdb.set_trace()
    solution = glue
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

    # # 示例用法
    # list_a = [[0, 3], [2, 4], [22, 25]]
    # list_b = [[1, 5], [2, 2], [2, 4]]
    # iou = compute_iou(list_a, list_b)

    rewards = []
    # print(completions, solution, durations, **kwargs)
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol, duration in zip(completions, solution, durations): # Added video_durations
        reward = 0.0

        gt_glue = sol

        pattern_glue = r'<answer>(.*?)</answer>'
        match_glue = re.search(pattern_glue, content, re.DOTALL)
        pred_glue = None
        if match_glue:
            glue = match_glue.group(1)
            if is_valid_two_d_list_format(glue):
                pred_glue = ast.literal_eval(glue)
                reward = compute_iou(pred_glue, [gt_glue])
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
    print(f'pred_glue:{pred_glue}, gt_glue:{gt_glue}, reward:{reward}, rewards:{rewards}')
    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    # pattern = re.compile(r'<sum>(.{11,}?)</sum>\s*<answer>.*?</answer>', re.DOTALL)
    pattern = re.compile(r'<answer>.*?</answer>', re.DOTALL)
    matches = [re.fullmatch(pattern, content.strip()) for content in completions]
    # print(completions[0])
    # print('matches:', matches)
    return [1.0 if match else 0.0 for match in matches]

def pad(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    # pattern = re.compile(r'<sum>.*?</sum>\s*<answer>.*?</answer>', re.DOTALL)
    # matches = [re.fullmatch(pattern, content.strip()) for content in completions]
    # # print('matches:', matches)
    # print(completions)
    # return [1.0 if match else 0.0 for match in matches]
    return [0 for content in completions]


def is_valid_list_of_lists(s):
    try:
        # 尝试将字符串解析为 Python 对象
        s = s.replace('\n', '')
        data = ast.literal_eval(s)
        
        # 检查解析后的对象是否是一个列表
        if not isinstance(data, list):
            return False
        
        # 检查列表的长度是否为 8
        if len(data) != 8:
            return False
        
        # 检查列表中的每个元素是否是长度为 4 的列表
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
    # pattern = re.compile(r'<sum>.*?</sum>\s*<answer>.*?</answer>', re.DOTALL)
    pattern = re.compile(r'<answer>.*?</answer>', re.DOTALL)
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


def detection_iou(completions, glue, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    solution = glue
    rewards = []
    # print(completions, solution, durations, **kwargs)
    # current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(completions, solution): # Added video_durations
        reward = 0.0
        
        content_match = re.search(r'<answer>(.*?)</answer>', content)
        gred_answer = content_match.group(1).strip() if content_match else content.strip()
        try:
            gred_answer = ast.literal_eval(gred_answer)            
            
            reward = calculate_iou(gred_answer, sol)
            
            rewards.append(reward)
        except Exception as e:
            rewards.append(0)
    # import pdb; pdb.set_trace()
    return rewards

def detection_glue(completions, glue, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    # import pdb; pdb.set_trace()
    solution = glue
    rewards = []
    # print(completions, solution, durations, **kwargs)
    # current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(completions, solution): # Added video_durations
        reward = 0.0
        
        content_match = re.search(r'<glue>(.*?)</glue>', content)
        gred_answer = content_match.group(1).strip() if content_match else content.strip()
        try:
            gred_answer = ast.literal_eval(gred_answer)            
            
            reward = calculate_iou(gred_answer, sol)
            
            rewards.append(0.2 * reward)
        except Exception as e:
            rewards.append(0)
    return rewards
