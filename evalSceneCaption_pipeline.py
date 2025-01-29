"""
evalSceneCaption_pipeline.py

코드 구조:
1. config 파일 로드
2. Scene Caption 생성하여 json 파일로 저장
"""

import os
from code.scene_caption_modules import scene_caption
from code.utils import load_config

if __name__ == "__main__":
    # config 파일 로드
    config = load_config("./config/scene_config_InternVideo2_5_chat.yaml")

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
