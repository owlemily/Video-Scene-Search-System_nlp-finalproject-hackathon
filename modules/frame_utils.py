"""
frame_utils.py

함수 목록:
1. extract_frames_from_single_video
2. extract_frames_from_folder
3. get_video_id_and_timestamp
4. create_and_save_dataset
5. extract_key_frames
"""

import os
from concurrent.futures import ThreadPoolExecutor

import cv2
from datasets import Dataset, Features, Image, Value
from tqdm import tqdm


def extract_frames_from_single_video(video_file_path, frames_output_folder, frame_rate):
    """
    1개의 비디오 파일을 입력으로 받아, 프레임(이미지 파일)들을 지정한 폴더에 저장하는 함수

    Args:
        video_file_path (str): 비디오 파일 경로. (ex. "../video/5qlG1ODkRWw.mp4")
        frames_output_folder (str): 추출된 프레임을 저장할 디렉토리 경로. (ex. "../frames")
        frame_rate (float): 초당 저장할 프레임 수.

    Returns:
        None
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

    Returns:
        None
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
    파일 이름(ex. "5qlG1ODkRWw_0.000.jpg")에서 video_id와 timestamp를 반환하는 함수

    Args:
        file_name (str): 파일 이름 (ex. "5qlG1ODkRWw_0.000.jpg")

    Returns:
        video_id (str): 비디오 ID
        timestamp (float): 타임스탬프
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

    Returns:
        None
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


def extract_key_frames(
    video_path,
    key_frames_output_folder,
    processor,
    clip_model,
    similarity_threshold=0.85,
    stddev_threshold=10,
):
    """
    1개의 비디오 파일에서 키 프레임들을 추출하여 key_frames_output_folder에 저장하는 함수

    Args:
        video_path (str): 비디오 파일 경로
        key_frames_output_folder (str): 키 프레임 저장 폴더 경로
        processor (CLIPProcessor): CLIP Processor
        clip_model (CLIP): CLIP 모델
        similarity_threshold (float): 이전 프레임과의 유사도 임계값
        stddev_threshold (int): 단색 프레임 판단 임계값 (default: 10)

    Returns:
        None
    """

    def is_solid_color(frame, stddev_threshold=10):
        """프레임이 단색인지 판단 (표준 편차 기준)"""
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, stddev = cv2.meanStdDev(gray_frame)
        return stddev[0][0] < stddev_threshold

    def generate_clip_embedding(frame):
        """프레임에 대한 CLIP 임베딩 생성"""
        inputs = processor(images=frame, return_tensors="pt").to("cuda")
        with torch.no_grad():
            embedding = clip_model.get_image_features(**inputs)
        return embedding

    os.makedirs(key_frames_output_folder, exist_ok=True)

    # 비디오 열기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 비디오를 열 수 없습니다: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print(f"[ERROR] 유효하지 않은 FPS입니다 (FPS <= 0): {video_path}")
        cap.release()
        return

    frame_count = 0
    saved_count = 0
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    prev_embedding = None
    buffer_frame = None
    buffer_timestamp = None

    while True:
        ret, frame = cap.read()
        if not ret:
            # 마지막 버퍼 프레임을 키 프레임으로 저장
            if buffer_frame is not None and not is_solid_color(
                buffer_frame, stddev_threshold
            ):
                key_frame_filename = f"{video_id}_{buffer_timestamp}.jpg"
                key_frame_path = os.path.join(
                    key_frames_output_folder, key_frame_filename
                )
                cv2.imwrite(key_frame_path, buffer_frame)
                saved_count += 1
            break

        # 단색 프레임 건너뛰기
        if is_solid_color(frame, stddev_threshold):
            frame_count += 1
            continue

        # 현재 프레임의 CLIP 임베딩 생성
        current_embedding = generate_clip_embedding(frame)

        if prev_embedding is None:
            # 첫번째 프레임
            buffer_frame = frame
            buffer_timestamp = f"{frame_count / fps:.3f}"
            prev_embedding = current_embedding
        else:
            # 이전 프레임과 비교
            similarity = F.cosine_similarity(
                prev_embedding, current_embedding, dim=1
            ).item()
            if similarity >= similarity_threshold:
                # 유사하면 버퍼를 현재 프레임으로 업데이트
                buffer_frame = frame
                buffer_timestamp = f"{frame_count / fps:.3f}"
                prev_embedding = current_embedding
            else:
                # 다르면 버퍼를 키 프레임으로 저장 후, 버퍼를 현재 프레임으로 업데이트
                if buffer_frame is not None:
                    key_frame_filename = f"{video_id}_{buffer_timestamp}.jpg"
                    key_frame_path = os.path.join(
                        key_frames_output_folder, key_frame_filename
                    )
                    cv2.imwrite(key_frame_path, buffer_frame)
                    saved_count += 1

                buffer_frame = frame
                buffer_timestamp = f"{frame_count / fps:.3f}"
                prev_embedding = current_embedding

        frame_count += 1

    cap.release()
    print(f"[INFO] {video_id}: Extracted {saved_count} key frames.")


if __name__ == "__main__":
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
