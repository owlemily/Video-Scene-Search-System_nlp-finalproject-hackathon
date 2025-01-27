"""
scene_caption_modules.py

함수 목록:
1. initialize_model
2. scene_caption_InternVideo2

추후 구현 예정:
3. scene_caption
"""

import json
import os

import decord
import torch
import yaml
from googletrans import Translator
from scene_utils import (
    read_timestamps_from_txt,
    save_all_video_scenes_by_timestamps,
    save_timestamps_to_txt,
)
from specific_model_utils.InternVideo2_utils import load_video
from transformers import AutoModel, AutoTokenizer

decord.bridge.set_bridge("torch")


def initialize_model(model_path="OpenGVLab/InternVideo2-Chat-8B"):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )
    model = AutoModel.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).cuda()

    return model, tokenizer


def scene_caption_InternVideo2(
    video_dir,
    output_directory,
    timestamp_file,
    folder_name,
    model_path,
    generation_config,
    prompt,
    final_output_path,
    use_audio,
    output_json_path,
):
    model, tokenizer = initialize_model(model_path)

    translator = Translator()

    # 최종 JSON 데이터 구조 생성
    final_json_data = []

    scene_timestamps = read_timestamps_from_txt(timestamp_file)

    for video_id, timestamps in scene_timestamps.items():
        for i, (start, end) in enumerate(timestamps):
            clip_name = f"{video_id}_{start:.3f}_{end:.3f}_{i + 1:03d}.mp4"
            clip_path = os.path.join(output_directory, clip_name)
            video_tensor = load_video(clip_path, num_segments=8, return_msg=False)
            video_tensor = video_tensor.to(model.device)

            if use_audio:
                with open(output_json_path, "r") as f:
                    scripts = json.load(f)
                audio_text = scripts[video_id][i]["audio_text"]

                # 비디오 클립 로드 및 모델 처리

            prompt = (
                prompt["clip_prompt_template"] + f"\n[script]: {audio_text}"
                if use_audio
                else prompt["clip_prompt_template"]
            )

            chat_history = []
            response, chat_history = model.chat(
                tokenizer,
                "",
                prompt,
                media_type="video",
                media_tensor=video_tensor,
                chat_history=chat_history,
                return_history=True,
                generation_config=generation_config,
            )

            translated_description = translator.translate(
                response, src="en", dest="ko"
            ).text

            final_json_data.append(
                {
                    "video_id": video_id,
                    "start_time": start,
                    "end_time": end,
                    "clip_id": f"{video_id}_{start}_{end}_{i + 1:03d}",
                    "caption": response,
                    "caption_ko": translated_description,
                }
            )

    with open(final_output_path, "w", encoding="utf-8") as json_file:
        json.dump(final_json_data, json_file, ensure_ascii=False, indent=4)

    print(f"All outputs have been saved to {final_output_path}.")


if __name__ == "__main__":
    # Load configuration from YAML file
    with open("config/config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    video_dir = config["video_dir"]
    output_directory = config["clip_output_directory"]  # full_clip저장할곳
    timestamp_file = config["timestamp_file"]
    folder_name = config["output_folder"]  # script
    model_path = config["model_path"]
    generation_config = config["generation_config"]
    prompt = config["prompt"]
    final_output_path = config["final_output"]

    # Output folder setup
    os.makedirs(folder_name, exist_ok=True)

    # Get video files
    video_files = [file for file in os.listdir(video_dir) if file.endswith(".mp4")]
    print(video_files)
    save_timestamps_to_txt(video_dir, timestamp_file, threshold=30.0, min_scene_len=2)
    # 동영상 클립 분할 실행
    save_all_video_scenes_by_timestamps(video_dir, output_directory, timestamp_file)
    # 5개만 우선 테스트해봅니다.
    # video_files = video_files[:2]
