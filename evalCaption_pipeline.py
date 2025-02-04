"""
evalCaption_pipeline.py

코드 구조:
1. config 파일 로드
2. frames 폴더들의 이미지들로 데이터셋 생성 후 저장 (datasets 폴더에 저장)
3. 데이터셋으로부터 프레임 캡션 생성 후 Json 파일로 저장
"""

from modules.frame_caption_modules import frame_caption
from modules.frame_utils import create_and_save_dataset
from modules.utils import load_config

if __name__ == "__main__":
    # config 파일 로드
    config = load_config("./config/evalCaption.yaml")

    device = config["general"]["device"]
    video_folder = config["general"]["video_folder"]
    translator_name = config["general"]["translator_name"]

    frames_folder = config["frame_caption"]["frames_folder"]
    frame_rate = config["frame_caption"]["frame_rate"]

    output_folder = config["frame_caption"]["output_folder"]
    frame_output_filename = config["frame_caption"]["frame_output_filename"]

    model_name = config["frame_caption"]["model"]

    use_datasets = config["frame_caption"]["use_datasets"]
    datasets_folder = config["frame_caption"]["datasets_folder"]
    datasets_name = config["frame_caption"]["datasets_name"]

    caption_prompt = config["frame_caption"]["prompt"]
    max_new_tokens = config["frame_caption"]["max_new_tokens"]
    batch_size = config["frame_caption"]["batch_size"]

    # frames 폴더들의 이미지들로 데이터셋 생성 후 저장 (datasets 폴더에 저장)
    create_and_save_dataset(frames_folder, datasets_folder, datasets_name)

    # 데이터셋으로부터 프레임 캡션 생성 후 Json 파일로 저장
    frame_caption(
        device,
        frames_folder,
        output_folder,
        datasets_folder,
        datasets_name,
        model_name,
        caption_prompt,
        max_new_tokens,
        batch_size,
        use_datasets,
        frame_output_filename,
        translator_name,
    )
