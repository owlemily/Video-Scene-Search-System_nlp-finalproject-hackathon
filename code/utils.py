"""
utils.py

함수 목록:
1. load_config
2. extract_frames_from_single_video
3. extract_frames_from_folder
4. get_video_id_and_timestamp
5. create_and_save_dataset
"""

import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import yaml
from datasets import Dataset, Features, Image, Value
from tqdm import tqdm


def load_config(config_path):
    """
    YAML 설정 파일을 로드하는 함수
    Args:
        config_path (str): 설정 파일 경로 (ex. "../config/config.yaml")
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def extract_frames_from_single_video(video_file_path, frames_output_folder, frame_rate):
    """
    1개의 비디오 파일을 입력으로 받아, 프레임(이미지 파일)들을 지정한 폴더에 저장하는 함수

    Args:
        video_file_path (str): 비디오 파일 경로. (ex. "../video/5qlG1ODkRWw.mp4")
        frames_output_folder (str): 추출된 프레임을 저장할 디렉토리 경로. (ex. "../frames")
        frame_rate (float): 초당 저장할 프레임 수.
    """
    if not os.path.exists(frames_output_folder):
        os.makedirs(frames_output_folder)

    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        print(f"비디오 파일을 열 수 없습니다: {video_file_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps / frame_rate) if fps > 0 else 1
    frame_count = 0
    saved_count = 0

    video_id = os.path.splitext(os.path.basename(video_file_path))[0]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval == 0:
            timestamp = round(frame_count / fps / (1 / frame_rate)) * (1 / frame_rate)
            frame_filename = f"{video_id}_{timestamp:.3f}.jpg"
            frame_path = os.path.join(frames_output_folder, frame_filename)
            cv2.imwrite(frame_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"{video_id}: 추출된 프레임 수 = {saved_count}")


def extract_frames_from_folder(video_folder, frames_output_folder, frame_rate):
    """
    비디오 폴더 내의 모든 비디오 파일로부터 프레임을 추출하여 지정된 폴더에 저장하는 함수

    Args:
        video_folder (str): 비디오 파일들이 있는 폴더 경로. (ex. "../video")
        frames_output_folder (str): 추출된 프레임을 저장할 디렉토리 경로. (ex. "../frames")
        frame_rate (float): 초당 저장할 프레임 수.
    """
    video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]
    print(f"총 {len(video_files)}개의 비디오 파일을 발견했습니다.")

    # 프레임 추출 - ThreadPoolExecutor를 사용하여 병렬 처리
    def process_video(video_file):
        video_path = os.path.join(video_folder, video_file)
        extract_frames_from_single_video(video_path, frames_output_folder, frame_rate)

    with ThreadPoolExecutor() as executor:
        # tqdm을 사용해 진행률 표시
        list(
            tqdm(
                executor.map(process_video, video_files),
                desc="비디오 프레임 추출 중",
                total=len(video_files),
            )
        )


def get_video_id_and_timestamp(file_name):
    """
    파일 이름에서 video_id와 timestamp를 반환하는 함수

    Args:
        file_name (str): 파일 이름 (ex. "5qlG1ODkRWw_0.000.jpg")
    """
    try:
        file_base = os.path.splitext(file_name)[0]
        video_id, timestamp_str = file_base.rsplit("_", 1)
        timestamp = float(timestamp_str)
        return video_id, timestamp

    except (IndexError, ValueError):
        return "", float("inf")


def create_and_save_dataset(
    frames_directory, output_dataset_directory, output_dataset_name
):
    """
    프레임 폴더로부터 데이터셋을 생성하고 저장하는 함수

    Args:
        frames_directory (str): 프레임 이미지 파일들이 있는 폴더 경로 (ex. "../frames")
        output_dataset_directory (str): 생성된 데이터셋을 저장할 디렉토리 경로 (ex. "../datasets")
        output_dataset_name (str): 생성된 데이터셋 파일 이름 (ex. "my_dataset")
    """
    image_file_paths = sorted(
        [
            os.path.join(frames_directory, f)
            for f in os.listdir(frames_directory)
            if f.endswith(".jpg")
        ]
    )
    image_file_names = sorted(
        [f for f in os.listdir(frames_directory) if f.endswith(".jpg")]
    )

    # 데이터를 저장할 딕셔너리 초기화
    data = {"image": [], "video_id": [], "timestamp": [], "frame_image_path": []}

    for file_path, frame_name in zip(image_file_paths, image_file_names):
        video_id, timestamp = get_video_id_and_timestamp(frame_name)
        data["image"].append(file_path)
        data["video_id"].append(video_id)
        data["timestamp"].append(timestamp)
        data["frame_image_path"].append(file_path)

    # Dataset 생성
    features = Features(
        {
            "image": Image(),
            "video_id": Value("string"),
            "timestamp": Value("string"),
            "frame_image_path": Value("string"),
        }
    )
    dataset = Dataset.from_dict(data, features=features)

    # 데이터셋 정보 출력
    print("생성된 데이터셋:")
    print(dataset)
    print("데이터셋 첫 번째 항목 예시:")
    print(dataset[0])

    # Datasets 저장
    if not os.path.exists(output_dataset_directory):
        os.makedirs(output_dataset_directory)
    dataset_name = os.path.join(output_dataset_directory, output_dataset_name)
    dataset.save_to_disk(dataset_name)


if __name__ == "__main__":
    # load_config 함수 테스트
    config = load_config("../config/fcs_config.yaml")
    print(config)

    # extract_frames 함수 테스트
    video_file_path = "../video/5qlG1ODkRWw.mp4"
    frames_output_folder = "../frames_test_5qlG1ODkRWw"
    frame_rate = 1
    extract_frames_from_single_video(video_file_path, frames_output_folder, frame_rate)

    # extract_frames_from_folder 함수 테스트
    extract_frames_from_folder("../video", "../frames", 1.0)

    # get_video_id_and_timestamp 함수 테스트
    print(get_video_id_and_timestamp("5qlG1ODkRWw_0.000.jpg"))

    # create_and_save_dataset 함수 테스트
    create_and_save_dataset("../frames", "../datasets", "my_test_dataset")
