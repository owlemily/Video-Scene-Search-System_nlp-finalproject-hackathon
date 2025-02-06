"""
평가자가 VTT 서비스를 평가할 때 사용하는 유틸리티 함수들을 정의한 파일입니다.

함수 목록:
1. save_frame_at_time
2. trim_video_segment_and_save
3. save_audio_from_scenes
4. convert_to_mono
5. transcribe_audio
6. translate_caption
"""

import os
import subprocess

import cv2
import yaml

import deepl
import torch
import torchaudio
from googletrans import Translator

def load_config(config_path):
    """
    YAML 설정 파일을 로드하는 함수
    Args:
        config_path (str): 설정 파일 경로 (ex. "../config/config.yaml")

    Returns:
        config (dict): 로드된 설정 파일
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"[ERROR] config_path가 존재하지 않습니다: {config_path}"
        )
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"[ERROR] 유효하지 않은 config 파일입니다: {config_path}\n{e}")

def save_frame_at_time(video_path, time_sec, output_path):
    """
    비디오 파일에서 특정 시간의 프레임을 저장하는 함수. (Video_id, timestamp를 입력받으면 해당 프레임을 jpg로 저장)

    Args:
        video_path (str): 비디오 파일 경로 (ex. "video_input_folder/video_id.mp4")
        time_sec (float): 저장할 프레임의 시간 (초)
        output_path (str): 저장할 프레임의 경로 (ex. "temp_save_folder/output.jpg")

    Returns:
        None
    """
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("에러: 비디오를 열 수 없습니다.")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(fps * time_sec)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # 프레임 읽기
    success, frame = cap.read()

    if not success:
        cap.release()
        raise Exception("해당 시간은 비디오의 범위를 벗어납니다.")

    # 이미지 저장
    cv2.imwrite(output_path, frame)
    print(f"Frame saved at {output_path}")

    cap.release()


def trim_video_segment_and_save(video_path, start, end, output_scene_folder):
    """
    원본 비디오에서 Start, End에 해당하는 동영상 부분을 잘라서 저장하는 함수.

    Args:
        video_path (str): 비디오 파일 경로.
        start (float): Scene 시작 시간.
        end (float): Scene 종료 시간.
        output_scene_folder (str): Scene이 저장될 폴더 경로.

    Returns:
        None
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"입력 비디오 파일이 존재하지 않습니다: {video_path}")

    os.makedirs(output_scene_folder, exist_ok=True)

    # video_path에서 video_id를 가져옵니다. (파일명)
    video_id = os.path.splitext(os.path.basename(video_path))[0]

    # video_id가 -로 시작하는 경우, FFmpeg에서 동영상을 잘라 저장할때 오류가 발생합니다. (예: -video_id_0_10.mp4)
    # 따라서, -로 시작하는 경우 _로 변경하여 저장하고 후에 다시 -로 파일명을 바꿔줍니다.
    starts_with_hyphen = video_id.startswith("-")
    if starts_with_hyphen:
        video_id = "_" + video_id.lstrip("-")

    output_scene_path = os.path.join(
        output_scene_folder, f"{video_id}_{start}_{end}.mp4"
    )

    # FFmpeg 명령어 (시작 시간부터 종료 시간까지의 동영상을 잘라서 저장)
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",  # 기존 파일 덮어쓰기
        "-loglevel",
        "error",  # 오류 메시지만 출력
        "-i",
        video_path,  # 입력 파일
        "-ss",
        str(start),  # 시작 시간
        "-to",
        str(end),  # 종료 시간
        "-c:v",
        "libx264",
        "-crf",
        "23",
        "-preset",
        "fast",  # 무손실 인코딩
        "-c:a",
        "aac",  # 오디오 인코딩
        "-b:a",
        "192k",  # 오디오 비트레이트 설정
        "-force_key_frames",
        f"expr:gte(t,{start})",  # 정확한 키 프레임 컷팅
        output_scene_path,
    ]

    try:
        subprocess.run(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg 실행 중 오류 발생:\n{e.stderr}")  # FFmpeg 오류 출력
        raise

    # starts_with_hyphen가 True이면, 저장한 output_scene_path의 앞 _를 -로 rename합니다.
    if starts_with_hyphen:
        output_scene_path_hyphen = os.path.join(
            output_scene_folder, f"-{video_id.lstrip('_')}_{start}_{end}.mp4"
        )
        os.rename(output_scene_path, output_scene_path_hyphen)


def save_audio_from_scenes(scene_path, output_mono_audio_folder):
    """
    입력된 Scene 비디오 파일에서 오디오를 추출하여 모노로 변환하여 wav 확장자로 저장하는 함수

    Args:
        scene_path (str): Scene 비디오 파일 경로.
        output_mono_audio_folder (str): 모노 오디오 파일을 저장할 폴더 경로.

    Returns:
        None
    """
    scene_file = os.path.basename(scene_path)  # scene_path에서 파일명만 추출
    scene_filename = os.path.splitext(scene_file)[0]  # .mp4 제거

    temp_audio_path = os.path.join(
        output_mono_audio_folder, f"{scene_filename}_temp.wav"
    )
    mono_audio_path = os.path.join(output_mono_audio_folder, f"{scene_filename}.wav")

    # video_id, start, end를 _로 구분하여 파일명으로 저장하므로 반대로 추출
    _, start, end = scene_filename.rsplit("_", 2)
    duaration = float(end) - float(start)

    # mp4 파일(Scene)에서 오디오 추출하는 ffmpeg 명령어 (temp_audio_path에 저장)
    ffmpeg_command = f'ffmpeg -i "{scene_path}" -to {duaration} -vn -acodec pcm_s16le "{temp_audio_path}" -loglevel error'
    os.system(ffmpeg_command)

    # 모노로 변환하여 mono_audio_path에 저장
    convert_to_mono(temp_audio_path, mono_audio_path)
    # print(f"Audio saved: {mono_audio_path}") # 너무 출력이 많아서 주석처리함

    os.remove(temp_audio_path)


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


def translate_caption(caption, translator, target_lang="ko"):
    """
    caption을 입력으로 받아 번역된 caption_ko를 반환하는 함수

    Args:
        caption (str): 번역할 캡션
        translator (googletrans.Translator or deepl.Translator): 번역기 객체
        target_lang (str): 번역할 언어 코드 (default: "ko")

    Returns:
        str: 번역된 캡션
    """
    try:
        if isinstance(translator, Translator):  # googletrans 사용
            return translator.translate(caption, dest=target_lang).text
        elif isinstance(translator, deepl.Translator):  # DeepL 사용
            return translator.translate_text(caption, target_lang=target_lang).text
        else:
            raise ValueError("지원되지 않는 번역기 객체입니다.")
    except Exception as e:
        print(f"번역 실패: {caption}. 오류: {e}")
        return ""


if __name__ == "__main__":
    video_path = "./video_input_folder/ncFDuKdgNE.mp4"
    start = 10.0
    end = 20.0
    output_scene_folder = "./output_scene_folder"

    trim_video_segment_and_save(video_path, start, end, output_scene_folder)
