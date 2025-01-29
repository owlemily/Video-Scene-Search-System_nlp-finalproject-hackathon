"""
scene_caption_pipeline.py

코드 구조:
1. config 파일 로드
2. 비디오 폴더로부터 타임스탬프 추출하여 txt 파일로 저장
3. 타임스탬프 파일로부터 비디오 Scene들을 추출하여 Scene 폴더에 저장
4. Scene 폴더로부터 모든 Scene의 오디오를 저장 (if use_audio)
5. Scene 오디오를 텍스트로 변환하여 json 파일로 저장 (if use_audio)
6. Scene Caption 생성하여 json 파일로 저장
"""

import os

from modules.audio_utils import (
    save_all_mono_audio_from_scene_folder,
    transcribe_and_save_scene_information_into_json,
)
from modules.scene_caption_modules import scene_caption
from modules.scene_utils import (
    save_all_video_scenes_by_timestamps,
    save_timestamps_to_txt,
)
from modules.utils import load_config

if __name__ == "__main__":
    # config 파일 로드
    config = load_config("./config/scene_config.yaml")

    video_folder = config["extract_frames"]["video_folder"]  # 비디오 폴더
    scene_folder = config["extract_scenes"]["scene_folder"]  # Scene 폴더
    timestamp_file = config["extract_scenes"]["timestamp_file"]  # 타임스탬프 파일
    mono_audio_folder = config["audio"]["mono_audio_folder"]  # 모노 오디오 폴더
    model_path = config["model"]["model_name"]
    generation_config = config["extract_scenes"]["generation_config"]
    prompt = config["generation"]["prompt"]
    scene_info_json_file = config["audio"][
        "scene_info_json_file"
    ]  # 오디오 스크립트 포함된 Scene 정보 JSON 파일
    use_audio = config["audio"]["use_audio"]
    scene_output_filename = config["data"]["scene_output_filename"]

    # 비디오 폴더의 모든 비디오에 대해 타임스탬프 추출하여 txt 파일로 저장
    save_timestamps_to_txt(
        video_folder, timestamp_file, threshold=30.0, min_scene_len=2
    )

    # 타임스탬프 txt 파일로부터 비디오 Scene(mp4) 추출하여 scene_folder에 저장
    save_all_video_scenes_by_timestamps(video_folder, scene_folder, timestamp_file)

    if use_audio:
        # Scene 폴더로부터 모든 Scene들의 모노 오디오를 mono_audio_folder에 저장
        save_all_mono_audio_from_scene_folder(scene_folder, mono_audio_folder)

        # mono_audio_folder에 저장된 모든 Scene의 오디오를 텍스트로 변환하여 Scene 정보 JSON 파일로 저장
        transcribe_and_save_scene_information_into_json(
            mono_audio_folder, scene_info_json_file, timestamp_file
        )

    # Scene Caption 생성하여 json 파일로 저장
    os.makedirs("output", exist_ok=True)
    scene_caption_output_path = os.path.join("output", scene_output_filename)

    scene_caption(
        model_path,
        scene_folder,
        prompt,
        generation_config,
        use_audio,
        mono_audio_folder,
        scene_info_json_file,
        scene_caption_output_path,
    )
