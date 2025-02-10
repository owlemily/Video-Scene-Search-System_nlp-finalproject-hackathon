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
    config = load_config("./config/base_config.yaml")

    video_folder = config["general"]["video_folder"]
    translator_name = config["general"]["translator_name"]

    PySceneDetect_threshold = config["scene_caption"]["PySceneDetect_threshold"]
    PySceneDetect_min_scene_len = config["scene_caption"]["PySceneDetect_min_scene_len"]

    timestamp_file = config["scene_caption"]["timestamp_file"]
    scene_folder = config["scene_caption"]["scene_folder"]
    output_folder = config["scene_caption"]["output_folder"]
    scene_output_filename = config["scene_caption"]["scene_output_filename"]

    model_path = config["scene_caption"]["model"]

    prompt = config["scene_caption"]["prompt"]
    max_new_tokens = config["scene_caption"]["max_new_tokens"]
    max_num_frames = config["scene_caption"]["max_num_frames"]

    use_audio = config["scene_caption"]["audio"]["use_audio"]
    mono_audio_folder = config["scene_caption"]["audio"]["mono_audio_folder"]

    scene_info_with_audio_scripts_file = config["scene_caption"]["audio"][
        "scene_info_with_audio_scripts_file"
    ]

    # 비디오 폴더의 모든 비디오에 대해 타임스탬프 추출하여 txt 파일로 저장
    save_timestamps_to_txt(
        video_folder,
        timestamp_file,
        PySceneDetect_threshold,
        PySceneDetect_min_scene_len,
    )

    # 타임스탬프 txt 파일로부터 비디오 Scene(mp4) 추출하여 scene_folder에 저장
    save_all_video_scenes_by_timestamps(video_folder, scene_folder, timestamp_file)

    if use_audio:
        # Scene 폴더로부터 모든 Scene들의 모노 오디오를 mono_audio_folder에 저장
        save_all_mono_audio_from_scene_folder(scene_folder, mono_audio_folder)

        if scene_info_with_audio_scripts_file is not None:
            # mono_audio_folder에 저장된 모든 Scene의 오디오를 텍스트로 변환하여 Scene 정보 JSON 파일로 저장
            transcribe_and_save_scene_information_into_json(
                mono_audio_folder, scene_info_with_audio_scripts_file, timestamp_file
            )

    # Scene Caption 생성하여 json 파일로 저장
    os.makedirs(output_folder, exist_ok=True)
    scene_caption_output_path = os.path.join(output_folder, scene_output_filename)

    scene_caption(
        model_path,
        scene_folder,
        prompt,
        max_new_tokens,
        max_num_frames,
        use_audio,
        mono_audio_folder,
        scene_info_with_audio_scripts_file,
        scene_caption_output_path,
        translator_name,
    )
