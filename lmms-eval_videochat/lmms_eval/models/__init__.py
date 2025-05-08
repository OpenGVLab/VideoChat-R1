import importlib
import os
import hf_transfer
from loguru import logger
import sys
import hf_transfer

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

logger.remove()
logger.add(sys.stdout, level="WARNING")

AVAILABLE_MODELS = {
    # "batch_gpt4": "BatchGPT4",
    # "claude": "Claude",
    # "from_log": "FromLog",
    # "fuyu": "Fuyu",
    # "gemini_api": "GeminiAPI",
    # "gpt4v": "GPT4V",
    # "idefics2": "Idefics2",
    # "instructblip": "InstructBLIP",
    # "internvl": "InternVLChat",
    # "internvl2": "InternVL2",
    "internvl2_video": "InternVL2_video",
    "internvl2_video_new": "InternVL2_video_new",
    # "llama_vid": "LLaMAVid",
    # "llava": "Llava",
    # "llava_hf": "LlavaHf",
    "llava_onevision": "Llava_OneVision",
    # "llava_sglang": "LlavaSglang",
    # "llava_vid": "LlavaVid",
    # "longva": "LongVA",
    # "mantis": "Mantis",
    # "minicpm_v": "MiniCPM_V",
    # "mplug_owl_video": "mplug_Owl",
    # "phi3v": "Phi3v",
    # "qwen_vl": "Qwen_VL",
    # "qwen_vl_api": "Qwen_VL_API",
    # "reka": "Reka",
    # "srt_api": "SRT_API",
    # "tinyllava": "TinyLlava",
    # "videoChatGPT": "VideoChatGPT",
    # "video_llava": "VideoLLaVA",
    # "vila": "VILA",
    # "xcomposer2_4KHD": "XComposer2_4KHD",
    # "xcomposer2d5": "XComposer2D5",
    "videochat_next": "VideoChat_NeXT",
    "videochat_next_image": "VideoChat_NeXT_image",
    "videochat_next_dynamic": "VideoChat_NeXT_dynamic",
    "videochat_next_pdrop": "VideoChat_NeXT_Pdrop",
    "videochat_next_fastv": "VideoChat_NeXT_FastV",
    "videochat_pdrop": "VideoChat_Pdrop",
    "videochat": "VideoChat",
    "videochat_next_old": "VideoChat_NeXT_old",
    "videochat_next_dynamic_pdrop":"VideoChat_NeXT_dynamic_pdrop",
    "videochat_next_dynamic_newprompt":"VideoChat_NeXT_dynamic_newprompt",
    "videochat_next_dynamic_pdrop_newprompt":"VideoChat_NeXT_dynamic_pdrop_newprompt",
    "videochat_flash": "VideoChat_Flash",
    "videochat_flash2": "VideoChat_Flash2",
    "qwen2_5_vl_lxh": "Qwen2_5_VL"
}

for model_name, model_class in AVAILABLE_MODELS.items():
    try:
        exec(f"from .{model_name} import {model_class}")
    except Exception as e:
        logger.debug(f"Failed to import {model_class} from {model_name}: {e}")

if os.environ.get("LMMS_EVAL_PLUGINS", None):
    # Allow specifying other packages to import models from
    for plugin in os.environ["LMMS_EVAL_PLUGINS"].split(","):
        m = importlib.import_module(f"{plugin}.models")
        for model_name, model_class in getattr(m, "AVAILABLE_MODELS").items():
            try:
                exec(f"from {plugin}.models.{model_name} import {model_class}")
            except ImportError as e:
                logger.debug(f"Failed to import {model_class} from {model_name}: {e}")
