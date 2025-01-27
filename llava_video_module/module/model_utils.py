from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
import copy
import torch
import sys
import warnings
from decord import VideoReader, cpu
import numpy as np
import torch
import os
import json
import torch
import whisper
import torchaudio
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from transformers import AutoModel, AutoTokenizer
from scenedetect import open_video, SceneManager
from module.audio_processing import transcribe_audio
from moviepy.video.io.VideoFileClip import VideoFileClip
import re
import cv2


# 모델 초기화하는 함수
def initialize_model(pretrained, model_name, device="cuda",device_map="auto"):
    print(f"Initializing model: {model_name} on device: {device}")
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map,attn_implementation="sdpa")  # Add any other thing you want to pass in llava_model_args
    model.eval()
    model = model.half()
    return tokenizer, model, image_processor, max_length



# 질문을 생성하는 함수
def create_question(prompt,video_time, frame_count, frame_time, conv_template="qwen_1_5"):
    time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {frame_count} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
    question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruciton}\n{prompt}"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()
