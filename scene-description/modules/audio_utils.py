"""
audio_utils.py

함수 목록:
1. reduce_repeated_characters
2. convert_to_mono
3. save_all_mono_audio_from_scene_folder
4. transcribe_audio
5. transcribe_and_save_scene_information_into_json
"""

import json
import os
import re

import torch
import torchaudio
import whisper
from tqdm import tqdm

from .scene_utils import read_timestamps_from_txt


def reduce_repeated_characters(text, max_repeats=5):
    """
    반복되는 문자를 최대 max_repeats로 줄이는 함수
    예시: "Hellooooo" -> "Hellooo" (max_repeats=3)

    Args:
        text (str): 입력 텍스트.
        max_repeats (int): 최대 반복 횟수.

    Returns:
        str: 반복되는 문자를 최대 max_repeats로 줄인 텍스트.
    """
    return re.sub(r"(.)\1{" + str(max_repeats) + r",}", r"\1" * max_repeats, text)


def convert_to_mono(input_wav_path, output_mono_path):
    """
    오디오 파일을 모노 채널로 변환하여 저장하는 함수
    Whisper STT 모델이 입력으로 모노 오디오를 필요로 함

    Args:
        input_wav_path (str): 입력 오디오 파일 경로.
        output_mono_path (str): 모노 오디오 파일을 저장할 경로.

    Returns:
        None
    """
    waveform, sample_rate = torchaudio.load(input_wav_path)
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    torchaudio.save(output_mono_path, waveform, sample_rate)


def save_all_mono_audio_from_scene_folder(scene_folder, output_mono_audio_folder):
    """
    Scene 폴더에서 모든 Scene의 오디오를 모노로 변환하여 저장하는 함수

    Args:
        scene_folder (str): Scene 비디오 파일이 저장된 폴더 경로.
        output_mono_audio_folder (str): 모노 오디오 파일을 저장할 폴더 경로.

    Returns:
        None
    """
    os.makedirs(output_mono_audio_folder, exist_ok=True)

    for scene_file in tqdm(os.listdir(scene_folder), desc="Saving mono audio"):
        if not scene_file.endswith(".mp4"):
            continue

        scene_path = os.path.join(scene_folder, scene_file)
        scene_filename = os.path.splitext(scene_file)[0]

        temp_audio_path = os.path.join(
            output_mono_audio_folder, f"{scene_filename}_temp.wav"
        )
        mono_audio_path = os.path.join(
            output_mono_audio_folder, f"{scene_filename}.wav"
        )

        _, start, end, _ = scene_filename.rsplit("_", 3)
        duaration = float(end) - float(start)
        # mp4 파일(Scene)에서 오디오 추출하는 ffmpeg 명령어 (temp_audio_path에 저장)
        ffmpeg_command = f'ffmpeg -i "{scene_path}" -to {duaration} -vn -acodec pcm_s16le "{temp_audio_path}" -loglevel error'
        os.system(ffmpeg_command)

        # 모노로 변환하여 mono_audio_path에 저장
        convert_to_mono(temp_audio_path, mono_audio_path)
        # print(f"Audio saved: {mono_audio_path}") # 너무 출력이 많아서 주석처리함

        os.remove(temp_audio_path)


def transcribe_audio(mono_audio_path, model):
    """
    모노 오디오 파일을 받아 SST 모델(ex. whisper)을 사용하여 텍스트로 반환하는 함수

    Args:
        mono_audio_path (str): 모노 오디오 파일 경로.
        model: SST 모델.

    Returns:
        str: 변환된 텍스트.
    """
    result = model.transcribe(mono_audio_path, language="en")
    text = result.get("text", "").strip()

    return text


def transcribe_and_save_scene_information_into_json(
    mono_audio_folder, output_json_path, timestamp_txt
):
    """
    현재 모든 Scene의 모노 오디오를 이용하여 SST 모델을 사용하여 자막으로 변환하고, JSON 파일로 Scene 정보(video_id, audio_text 등)를 저장하는 함수

    Args:
        mono_audio_folder (str): 모노 오디오 파일이 저장된 폴더 경로.
        output_json_path (str): JSON 파일로 저장할 경로.
        timestamp_txt (str): Scene 타임스탬프가 저장된 txt 파일 경로.

    Returns:
        None
    """
    print("Loading SST model...")

    # STT 모델인 Whisper를 불러옴
    whisper_model = whisper.load_model("large-v3")

    timestamp_dict = read_timestamps_from_txt(timestamp_txt)

    results = {}

    for video_id, timestamps in tqdm(timestamp_dict.items()):
        # 만약 video_id가 '-'로 시작하면, '_'로 변경 (scene 변환시 오류 방지)
        if video_id.startswith("-"):
            video_id = "_" + video_id.lstrip("-")

        results[video_id] = []
        for i, (start, end) in enumerate(timestamps):
            mono_audio_path = os.path.join(
                mono_audio_folder, f"{video_id}_{start}_{end}_{i + 1:03d}.wav"
            )
            audio_text = transcribe_audio(mono_audio_path, whisper_model)
            audio_text = reduce_repeated_characters(audio_text)
            results[video_id].append(
                {
                    "clip_id": i + 1,
                    "start": start,
                    "end": end,
                    "audio_text": audio_text,
                }
            )

    with open(output_json_path, "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=2)

    print(f"Results saved to {output_json_path}")


if __name__ == "__main__":
    # # save_all_mono_audio_from_scene_folder 함수 테스트
    # save_all_mono_audio_from_scene_folder("../scenes", "../mono_audio")

    # # transcribe_audio 함수 테스트
    # mono_audio_path = "../mono_audio/5qlG1ODkRWw_1.084_3.921_001.wav"
    # whisper_model = whisper.load_model("large-v3")
    # print(transcribe_audio(mono_audio_path, whisper_model))

    # transcribe_and_save_scene_information_into_json 함수 테스트
    transcribe_and_save_scene_information_into_json(
        "../mono_audio", "../scene_info.json", "../timestamps.txt"
    )
