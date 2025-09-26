from __future__ import annotations

from torchvision import transforms
import base64
import logging
import math
import os
import sys
import time
import warnings
from functools import lru_cache
from io import BytesIO

import requests
import torch
import torchvision
from packaging import version
from PIL import Image
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode
from typing import Optional
import random
import os
import io
import av
import cv2
import decord
import imageio
from decord import VideoReader
import torch
import numpy as np
import math
import gc
import torchaudio
from torchvision.transforms.functional import pil_to_tensor
import re
# logger = logging.getLogger(__name__)

# from models.backbones.beats.BEATs import BEATs, BEATsConfig
from petrel_client.client import Client
client = Client('~/petreloss.conf')


def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets

def get_frame_indices(num_frames, vlen, sample='middle', fix_start=None, input_fps=1, min_num_frames=1, max_num_frames=-1, local_num_frames=8):

    if min_num_frames > vlen:
        if sample == 'dynamic_fps1':
            min_num_frames = (vlen // local_num_frames) * local_num_frames
        else:
            min_num_frames = vlen


    if sample == 'dynamic_fps1':

        duration = float(vlen) / input_fps
        num_segments = int(duration // local_num_frames)
        if num_segments == 0:
            num_frames = local_num_frames
        else:
            num_frames = local_num_frames * num_segments

        if max_num_frames > 0:
            num_frames = min(num_frames, max_num_frames)
        sample = "middle" # NOTE

        # logger.info(f"? is OK (img), duation={duration} frames={num_frames}!!!!")

    num_frames = max(min_num_frames, num_frames)

    # print(f"\033[0;31m vlen={vlen}, input_fps={input_fps} num_frames={num_frames} \033[0m")
        
    if sample in ["rand", "middle"]: # uniform sampling
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == 'middle':
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif "fps" in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        raise NotImplementedError
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
            # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    else:
        raise ValueError(f"Not support sample type: {sample}")
    
    
    return frame_indices



logger = logging.getLogger(__name__)

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 768 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 512

# Set the maximum number of video token inputs.
# Here, 128K represents the maximum number of input tokens for the VLLM model.
# Remember to adjust it according to your own configuration.
VIDEO_TOTAL_PIXELS = int(float(os.environ.get('VIDEO_MAX_PIXELS', 128000 * 28 * 28 * 0.9)))
logger.info(f"set VIDEO_TOTAL_PIXELS: {VIDEO_TOTAL_PIXELS}")


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def to_rgb(pil_image: Image.Image) -> Image.Image:
      if pil_image.mode == 'RGBA':
          white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
          white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
          return white_background
      else:
          return pil_image.convert("RGB")


def fetch_image(ele: dict[str, str | Image.Image], client= None, size_factor: int = IMAGE_FACTOR) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif isinstance(image[0], Image.Image):
        image_obj = image[0]
    elif image.startswith("http://") or image.startswith("https://"):
        response = requests.get(image, stream=True)
        image_obj = Image.open(BytesIO(response.content))
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(BytesIO(data))
    elif 's3' in image:
        file_content = client.get(image)
        image_obj = Image.open(io.BytesIO(file_content))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = to_rgb(image_obj)
    ## resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels", MIN_PIXELS)
        max_pixels = ele.get("max_pixels", MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height))

    return image


def smart_nframes(
    ele: dict,
    total_frames: int,
    video_fps: int | float,
) -> int:
    """calculate the number of frames for video used for model inputs.

    Args:
        ele (dict): a dict contains the configuration of video.
            support either `fps` or `nframes`:
                - nframes: the number of frames to extract for model inputs.
                - fps: the fps to extract frames for model inputs.
                    - min_frames: the minimum number of frames of the video, only used when fps is provided.
                    - max_frames: the maximum number of frames of the video, only used when fps is provided.
        total_frames (int): the original total number of frames of the video.
        video_fps (int | float): the original fps of the video.

    Raises:
        ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

    Returns:
        int: the number of frames for video used for model inputs.
    """
    assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        fps = ele.get("fps", FPS)
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR)
        nframes = total_frames / video_fps * fps
        if nframes > total_frames:
            logger.warning(f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]")
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = floor_by_factor(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
    return nframes


def _read_video_torchvision(
    ele: dict,
) -> (torch.Tensor, float):
    raise NotADirectoryError
    """read video using torchvision.io.read_video

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    video_path = ele["video"]
    if version.parse(torchvision.__version__) < version.parse("0.19.0"):
        if "http://" in video_path or "https://" in video_path:
            warnings.warn("torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0.")
        if "file://" in video_path:
            video_path = video_path[7:]
    st = time.time()
    video, audio, info = io.read_video(
        video_path,
        start_pts=ele.get("video_start", 0.0),
        end_pts=ele.get("video_end", None),
        pts_unit="sec",
        output_format="TCHW",
    )
    total_frames, video_fps = video.size(0), info["video_fps"]
    logger.info(f"torchvision:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long()
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    video = video[idx]
    return video, sample_fps


def is_decord_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("decord") is not None

def pts_to_secs(pts: int, time_base: float, start_pts: int) -> float:
    """
    Converts a present time with the given time base and start_pts offset to seconds.

    Returns:
        time_in_seconds (float): The corresponding time in seconds.

    https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/data/utils.py#L54-L64
    """
    if pts == math.inf:
        return math.inf

    return int(pts - start_pts) * time_base

def get_pyav_video_duration(video_reader):
    video_stream = video_reader.streams.video[0]
    video_duration = pts_to_secs(
        video_stream.duration,
        video_stream.time_base,
        video_stream.start_time
    )
    return float(video_duration)



def _read_video_img(
    ele: dict,
    client
) -> (torch.Tensor, float):
    """read video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    import re

    def extract_frame_number(filename):
        # Extract the numeric part from the filename using regular expressions
        if filename.endswith('.jpg'):
            match = re.search(r'_(\d+).jpg$', filename)
        elif filename.endswith('.jpeg'):
            match = re.search(r'_(\d+).jpeg$', filename)
        elif filename.endswith('.png'):
            match = re.search(r'_(\d+).png$', filename)
        else:
            raise NotImplementedError(f"Wrong filename: {filename}")

        return int(match.group(1)) if match else -1


    def sort_frames(frame_paths):
        # Extract filenames from each path and sort by their numeric part
        return sorted(frame_paths, key=lambda x: extract_frame_number(os.path.basename(x)))
    video_path = ele["video"]
    # import pdb; pdb.set_trace()
    if 'media_dict' in ele:
        media_dict = ele["media_dict"]
    else:
        media_dict = {}
    if "s3://" in video_path:
        img_list = sort_frames(client.list(video_path))
    else:
        img_list = sort_frames(list(os.listdir(video_path)))
    
    
    
    if "start" in media_dict.keys():
        clip = [media_dict['start'], media_dict['end']]
    else:
        clip = None

    if 'tvqa' in video_path.lower():
        fps = 3.0 # tvqa是3fps的
    else:
        fps = 1.0 # NOTE 未知数据直接当1fps处理

    if clip is not None:
        start = float(clip[0])
        end = float(clip[1])
        start = max(0, start)
        end = min(len(img_list) / fps, end) # 防止end超过视频末尾 
        vlen = (end - start) * fps
    else:
        vlen = len(img_list)
    
    duration = vlen / fps

    num_frames = smart_nframes(ele, total_frames=vlen, video_fps=fps)
    sample = 'middle'
    if clip is not None:
        def _get_index_by_time(start_sec, end_sec, num_segments=8, fps=1., max_frame=9999):
            start_idx = max(1, round(start_sec * fps))
            end_idx = min(round(end_sec * fps), max_frame)
            seg_size = float(end_idx - start_idx) / (num_segments - 1)
            offsets = np.array([start_idx + int(np.round(seg_size * idx)) for idx in range(num_segments)])
            return offsets

        frame_indices = _get_index_by_time(float(clip[0]), float(clip[1]), num_segments=num_frames, fps=fps, max_frame=len(img_list)-1)
    else:
        frame_indices = get_frame_indices(
            num_frames, vlen, sample=sample, local_num_frames=1, input_fps=fps
        )

    imgs = []
    for idx in frame_indices:
        frame_fname = os.path.join(video_path, img_list[idx])
        if "s3://" in video_path:
            img_bytes = client.get(frame_fname)
        else:
            with open(frame_fname, 'rb') as f:
                img_bytes = f.read()
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        imgs.append(img)

    # print(f"\033[0;31m img_list={len(img_list)} video_path={video_path}, len(imgs)={len(imgs)}, frame_indices={frame_indices} num_frames={num_frames} \033[0m")
    frames = np.array(imgs, dtype=np.uint8)

    frames = torch.tensor(np.array(imgs), dtype=torch.uint8).permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
    sample_fps = num_frames / max(vlen, 1e-6) * fps

    return frames, sample_fps

def _read_video_av(
    ele: dict,
    client
) -> (torch.Tensor, float):
    """read video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    video_path = ele["video"]
    if 'media_dict' in ele:
        media_dict = ele["media_dict"]
    else:
        media_dict = {}
    if "start" in media_dict.keys():
        clip = [media_dict['start'], media_dict['end']]
    else:
        clip = None

    if clip is not None:
        raise NotImplementedError("av don't support clip!!!")
    if 's3://' in video_path:
        video_bytes = client.get(video_path)
        byteio = io.BytesIO(video_bytes)
        byteio.seek(0)
        reader = av.open(byteio)
    else:
        byteio = None
        reader = av.open(video_path)
    frames = [f.to_rgb().to_ndarray() for f in reader.decode(video=0)]
    vlen = len(frames)
    sample = 'middle'
    duration = get_pyav_video_duration(reader)
    fps = vlen / float(duration)
    num_frames = smart_nframes(ele, total_frames=vlen, video_fps=fps)
    frame_indices = get_frame_indices(
        num_frames, vlen, sample=sample,
        input_fps=fps,  local_num_frames=1
    )
    frames = np.stack([frames[idx] for idx in frame_indices])  # (T, H, W, C), torch.uint8
    frames = torch.tensor(frames).permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
    if byteio != None:
        byteio.close()
        
    reader.close()
    sample_fps = num_frames / max(vlen, 1e-6) * fps

    return frames, sample_fps


import numpy as np


def sample_frames(key_timestamps, total_frames, num_frames, key_ratio, fps):
    """
    从total_frames中取num_frames个帧，从key_timestamps中采样的帧密度是其它部分的key_ratio倍
    
    参数：
        key_timestamps (list of tuple): 每个元素是一个包含起始和结束时间的元组，单位为秒。
        total_frames (int): 总帧数。
        num_frames (int): 需要采样的帧数。
        key_ratio (float): 从key_timestamps中采样的帧密度是其它部分的key_ratio倍。
        fps (int): 视频帧率（每秒帧数）。
        
    返回：
        list: 包含采样帧索引的列表。
    """

    # Step 1: 将关键时间段转换为帧区间
    key_frame_ranges = []
    for start_sec, end_sec in key_timestamps:
        start_frame = int(start_sec * fps)
        end_frame = min(int(end_sec * fps), total_frames - 1)
        if start_frame <= end_frame:
            key_frame_ranges.append((start_frame, end_frame))



    # Step 2: 计算关键区域和非关键区域的帧数
    key_frames_count = sum(end - start + 1 for start, end in key_frame_ranges)

    non_key_frames_count = total_frames - key_frames_count

    # Step 3: 根据 key_ratio 分配采样帧数
    if non_key_frames_count == 0 and key_ratio == 0:
        raise ValueError("No frames available for sampling.")

    # 设定权重：关键区域权重为 key_ratio，非关键区域为 1
    key_weight = key_ratio
    non_key_weight = 1.0

    total_weighted_frames = key_frames_count * key_weight + non_key_frames_count * non_key_weight

    key_sample_count = round(num_frames * (key_frames_count * key_weight) / total_weighted_frames)
    non_key_sample_count = num_frames - key_sample_count

    # Step 4: 从关键区域采样
    key_samples = []
    for start, end in key_frame_ranges:
        num = max(1, round(key_sample_count * (end - start + 1) / key_frames_count))
        samples = np.linspace(start, end, num=num, dtype=int)
        key_samples.extend(samples)

    # Step 5: 从非关键区域采样
    non_key_samples = []

    # 构建非关键区域帧集合
    key_set = set()
    for start, end in key_frame_ranges:
        key_set.update(range(start, end + 1))

    non_key_frames = [i for i in range(total_frames) if i not in key_set]

    # 均匀采样非关键帧
    if non_key_sample_count > 0 and len(non_key_frames) > 0:
        indices = np.linspace(0, len(non_key_frames) - 1, num=non_key_sample_count, dtype=int)
        non_key_samples = [non_key_frames[i] for i in indices]

    # 合并结果并排序
    all_samples = sorted(set(key_samples + non_key_samples))

    return all_samples

def _read_video_decord(
    ele: dict,
    client = None,
) -> (torch.Tensor, float):
    """read video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    import decord
    video_path = ele["video"]
    
    if video_path.endswith('.avi'):
        return _read_video_av(ele, client)
    st = time.time()

    if 's3://' in video_path:
        video_bytes = client.get(video_path)
        if video_bytes is None or len(video_bytes) == 0:
            raise ValueError(f"Can't read byte from {video_path}!")
        byteio = BytesIO(video_bytes)
        vr = decord.VideoReader(byteio, num_threads=1)
    else:
        byteio = None
        vr = decord.VideoReader(video_path)

    # TODO: support start_pts and end_pts
    if 'video_start' in ele or 'video_end' in ele:
        raise NotImplementedError("not support start_pts and end_pts in decord for now.")
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    logger.info(f"decord:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    # import pdb; pdb.set_trace()
    if 'key_time' in ele and ele['key_time'] is not None:
        try:
            idx = sample_frames(ele['key_time'], total_frames-1, nframes, 1.5, vr.get_avg_fps())
        except Exception as e:
            idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    else:
        idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    video = vr.get_batch(idx).asnumpy()
    video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    return video, sample_fps


VIDEO_READER_BACKENDS = {
    "decord": _read_video_decord,
    'img': _read_video_img,
    'frame': _read_video_img,
    'av': _read_video_av,
    'torchvision':_read_video_torchvision
}

FORCE_QWENVL_VIDEO_READER = os.getenv("FORCE_QWENVL_VIDEO_READER", None)


@lru_cache(maxsize=1)
def get_video_reader_backend() -> str:
    if FORCE_QWENVL_VIDEO_READER is not None:
        video_reader_backend = FORCE_QWENVL_VIDEO_READER
    elif is_decord_available():
        video_reader_backend = "decord"
    else:
        video_reader_backend = "torchvision"
    print(f"qwen-vl-utils using {video_reader_backend} to read video.", file=sys.stderr)
    return video_reader_backend

def fetch_video(ele: dict, client = None, image_factor: int = IMAGE_FACTOR, return_video_sample_fps: bool = False) -> torch.Tensor | list[Image.Image]:
    video_reader_backend = get_video_reader_backend()
    # import pdb; pdb.set_trace()
    if isinstance(ele["video"], list):
        if isinstance(ele["video"][1], dict) and 'video_read_type' in ele['video'][1]:
            
            video_reader_backend = ele['video'][1]['video_read_type']
            ele['video'] = ele['video'][0]
            
    if isinstance(ele["video"], str):
        
        # print(f'video_reader_backend:{video_reader_backend}')
        # import pdb; pdb.set_trace()
        try:
            video, sample_fps = VIDEO_READER_BACKENDS[video_reader_backend](ele, client=client)
        except Exception as e:
            video, sample_fps = VIDEO_READER_BACKENDS[video_reader_backend](ele, client=client)
            logger.warning(f"video_reader_backend {video_reader_backend} error, use torchvision as default, msg: {e}")
            video, sample_fps = VIDEO_READER_BACKENDS["torchvision"](ele)

        nframes, _, height, width = video.shape
        
        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
        max_pixels_supposed = ele.get("max_pixels", max_pixels)
        if max_pixels_supposed > max_pixels:
            logger.warning(f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}].")
        max_pixels = min(max_pixels_supposed, max_pixels)

        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=image_factor,
            )
        else:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()
        if return_video_sample_fps:
            return video, sample_fps
        return video
    else:
        assert isinstance(ele["video"], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        images = [
            fetch_image({"image": video_element, **process_info}, size_factor=image_factor)
            for video_element in ele["video"]
        ]
        nframes = ceil_by_factor(len(images), FRAME_FACTOR)
        if len(images) < nframes:
            images.extend([images[-1]] * (nframes - len(images)))
        if return_video_sample_fps:
            return images, process_info.pop("fps", 2.0)
        return images

def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                        "image" in ele
                        or "images" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or ele["type"] in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    return vision_infos


def process_vision_info(
    conversations: list[dict] | list[list[dict]],
    return_video_kwargs: bool = False,
    client = None
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None, Optional[dict]]:

    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info,client=client))
        elif "video" in vision_info:
            video_input, video_sample_fps = fetch_video(vision_info, return_video_sample_fps=True, client=client)
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
        # elif "images" in video_info:
        #     # video_sample_fps = 2
        #     # for dd in vision_info['images']:
        #     video_input = fetch_images(vision_info)
        #     video_inputs.append(video_input)
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    if return_video_kwargs:
        return image_inputs, video_inputs, {'fps': video_sample_fps_list}
    return image_inputs, video_inputs
