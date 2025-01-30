"""
scene_utils.py

함수 목록:
1. get_filled_scene_timestamps
2. save_timestamps_to_txt
3. read_timestamps_from_txt
4. save_video_scenes_by_timestamps
5. process_video
6. save_all_video_scenes_by_timestamps
"""

import os
import subprocess
from multiprocessing import Pool, cpu_count

import cv2
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector
from tqdm import tqdm


def get_filled_scene_timestamps(video_path, threshold=30.0, min_scene_len=2):
    """
    1개의 비디오에 대해 PySceneDetect를 사용하여 Scene들의 타임스탬프(start, end) 리스트를 반환하는 함수
    Args:
        video_path (str): 비디오 파일 경로. (ex. "../video/5qlG1ODkRWw.mp4")
        threshold (float): ContentDetector의 threshold 값 (높을수록 Scene이 적어짐).
        min_scene_len (int): Scene의 최소 길이(초 단위).

    Returns:
        list: Scene의 타임스탬프 리스트. (ex. [(0.0, 10.0), (15.0, 20.0), ...])
    """
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()
    timestamps = []

    for scene in scene_list:
        start = scene[0].get_seconds()
        end = scene[1].get_seconds()
        if end - start >= min_scene_len:
            timestamps.append((start, end))

    # 비어있는 구간 채우기
    filled_timestamps = []
    previous_end = 0.0

    for start, end in timestamps:
        if start > previous_end:
            filled_timestamps.append((previous_end, start))
        filled_timestamps.append((start, end))
        previous_end = end

    real_end = float(
        cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT)
    ) / cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)
    real_end = real_end

    if previous_end < real_end:
        filled_timestamps.append((previous_end, real_end))

    return filled_timestamps


def save_timestamps_to_txt(
    video_folder, output_txt_path, threshold=30.0, min_scene_len=2
):
    """
    비디오 폴더 내의 모든 비디오에 대해 Scene 타임스탬프를 추출하여 txt 파일에 저장하는 함수

    Args:
        video_folder (str): 비디오 파일들이 있는 폴더 경로. (ex. "../video")
        output_txt_path (str): Scene 타임스탬프를 저장할 txt 파일 경로. (ex. "../timestamps.txt")
        threshold (float): ContentDetector의 threshold 값.
        min_scene_len (int): Scene의 최소 길이(초 단위).

    Returns:
        None
    """
    with open(output_txt_path, "w") as txt_file:
        video_files = [
            video_name
            for video_name in os.listdir(video_folder)
            if video_name.endswith(".mp4")
        ]
        for video_name in tqdm(
            video_files, desc="타임스탬프를 추출하여 txt 파일에 저장하는 중"
        ):
            video_path = os.path.join(video_folder, video_name)
            video_id = os.path.splitext(video_name)[0]

            try:
                timestamps = get_filled_scene_timestamps(
                    video_path, threshold, min_scene_len
                )

                # Write to txt file
                for start, end in timestamps:
                    txt_file.write(f"{video_id}\t{start}\t{end}\n")

            except Exception as e:
                print(f"Error processing {video_name}: {e}")


def read_timestamps_from_txt(timestamp_txt_path):
    """
    txt 파일로부터 비디오별 Scene 타임스탬프를 읽어오는 함수

    Args:
        timestamp_txt_path (str): Scene 타임스탬프가 저장된 txt 파일 경로.

    Returns:
        dict: 비디오별 Scene 타임스탬프를 저장한 딕셔너리. (ex. {"5qlG1ODkRWw": [(0.0, 10.0), (15.0, 20.0), ...], ...})
    """
    timestamps_dict = {}

    with open(timestamp_txt_path, "r") as txt_file:
        for line in txt_file:
            video_id, start, end = line.strip().split("\t")
            start = float(start)
            end = float(end)

            if video_id not in timestamps_dict:
                timestamps_dict[video_id] = []

            timestamps_dict[video_id].append((start, end))

    return timestamps_dict


def save_video_scenes_by_timestamps(video_path, timestamps, output_scene_folder):
    """
    정확한 타임스탬프 기반으로 비디오를 프레임 손실 없이 자르는 함수.

    Args:
        video_path (str): 비디오 파일 경로.
        timestamps (list): [(start_time, end_time)] 형식의 리스트.
        output_scene_folder (str): Scene을 저장할 폴더.
        reencode (bool): True면 H.264로 재인코딩하여 정확한 컷팅 수행.

    Returns:
        None
    """
    os.makedirs(output_scene_folder, exist_ok=True)

    video_id = os.path.splitext(os.path.basename(video_path))[0]

    # 만약 video_id가 '-'로 시작하면, '_'로 변경 (scene 변환시 오류 방지)
    if video_id.startswith("-"):
        video_id = "_" + video_id.lstrip("-")

    for i, (start, end) in enumerate(timestamps):
        output_scene_path = os.path.join(
            output_scene_folder, f"{video_id}_{start}_{end}_{i + 1:03d}.mp4"
        )

        # 정확한 프레임 컷팅을 위한 H.264 재인코딩 방식
        ffmpeg_cmd = [
            "ffmpeg",
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
            "aac",  # 오디오는 그대로 복사
            "-b:a",
            "192k",  # 오디오 비트레이트 설정 (품질 유지)
            # "-threads",
            # "7",  # 멀티코어 CPU 활용
            "-force_key_frames",
            "expr:gte(t,n_forced*2)",  # 정확한 키 프레임 컷팅
            output_scene_path,
        ]

        # FFmpeg 실행
        subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # print(f"Scene {i + 1} saved: {output_scene_path}") # 너무 출력이 많아서 주석처리함


def process_video(args):
    """병렬 처리를 위한 헬퍼 함수"""
    video_path, timestamps, output_scene_folder = args
    save_video_scenes_by_timestamps(video_path, timestamps, output_scene_folder)


def save_all_video_scenes_by_timestamps(
    video_folder, output_scene_folder, timestamp_txt_path
):
    """
    타임스탬프 txt를 읽어와서 비디오 폴더 내의 모든 비디오를 병렬로 Scene(.mp4)를 추출하여 저장하는 함수.

    Args:
        video_folder (str): 비디오 파일들이 있는 폴더 경로. (ex. "../video")
        output_scene_folder (str): Scene을 저장할 폴더 경로. (ex. "../scenes")
        timestamp_txt_path (str): Scene 타임스탬프가 저장된 txt 파일 경로.

    Returns:
        None
    """

    # 타임스탬프 로드 (파일에서 읽어옴)
    timestamp_dict = read_timestamps_from_txt(timestamp_txt_path)

    # 처리할 비디오 리스트 생성
    video_tasks = []
    for video_id, timestamps in timestamp_dict.items():
        video_path = os.path.join(video_folder, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            print(f"Video file {video_path} not found. Skipping...")
            continue
        video_tasks.append((video_path, timestamps, output_scene_folder))

    # 병렬 처리: CPU 코어 개수만큼 워커(worker) 생성
    num_workers = min(len(video_tasks), cpu_count())  # 코어 수만큼 병렬 실행
    with Pool(num_workers) as pool:
        list(
            tqdm(
                pool.imap_unordered(process_video, video_tasks),
                total=len(video_tasks),
                desc="Processing Videos",
            )
        )


if __name__ == "__main__":
    video_folder = "../video"
    output_txt_path = "../timestamps.txt"
    output_scene_folder = "../scenes"

    # 타임스탬프 추출 후 Scene 폴더에 Scene(.mp4) 저장
    save_timestamps_to_txt(video_folder, output_txt_path)
    save_all_video_scenes_by_timestamps(
        video_folder, output_scene_folder, output_txt_path
    )


# =============================================================================
"""
Legacy code

아래 코드는 이전에 사용했던 코드로, 현재 사용하지 않습니다.

1. save_all_video_scenes_by_timestamps
- 병렬 처리를 사용하지 않고, For문으로 Scene path를 추출하여 저장하는 함수입니다.
"""
# def save_all_video_scenes_by_timestamps(
#     video_folder, output_scene_folder, timestamp_txt_path
# ):
#     """
#     타임스탬프 txt를 읽어와서 비디오 폴더 내의 모든 비디오에 대해 Scene(.mp4)를 추출하여 저장하는 함수

#     Args:
#         video_folder (str): 비디오 파일들이 있는 폴더 경로. (ex. "../video")
#         output_scene_folder (str): Scene을 저장할 폴더 경로. (ex. "../scenes")
#         timestamp_txt_path (str): Scene 타임스탬프가 저장된 txt 파일 경로.

#     Returns:
#         None
#     """
#     timestamp_dict = read_timestamps_from_txt(timestamp_txt_path)

#     for video_id, timestamps in tqdm(timestamp_dict.items(), desc="Scene 저장 중"):
#         video_path = os.path.join(video_folder, f"{video_id}.mp4")
#         if not os.path.exists(video_path):
#             print(f"Video file {video_path} not found. Skipping...")
#             continue

#         save_video_scenes_by_timestamps(video_path, timestamps, output_scene_folder)
