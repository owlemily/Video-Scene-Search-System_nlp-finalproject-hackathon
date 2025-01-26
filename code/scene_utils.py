import json
import os
import re

import cv2
import whisper
from modules.audio_processing import transcribe_audio
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector


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
        start = round(scene[0].get_seconds(), 3)
        end = round(scene[1].get_seconds(), 3)
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

    if previous_end < real_end:
        filled_timestamps.append((previous_end, real_end))

    return timestamps


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
        for video_name in video_files:
            video_path = os.path.join(video_folder, video_name)
            video_id = os.path.splitext(video_name)[0]

            try:
                timestamps = get_filled_scene_timestamps(
                    video_path, threshold, min_scene_len
                )

                # Write to txt file
                for start, end in timestamps:
                    txt_file.write(f"{video_id}\t{start:.3f}\t{end:.3f}\n")

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


def save_all_video_scenes_by_timestamps(
    video_folder, output_scene_folder, timestamp_txt_path
):
    os.makedirs(output_scene_folder, exist_ok=True)

    timestamp_dict = read_timestamps_from_txt(timestamp_txt_path)

    for video_id, timestamps in timestamp_dict.items():
        video_path = os.path.join(video_folder, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            print(f"Video file {video_path} not found. Skipping...")
            continue

        for start, end in timestamps:
            output_scene_path = os.path.join(
                output_scene_folder, f"{video_id}_{start:.3f}_{end:.3f}.mp4"
            )

            try:
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(output_scene_path, fourcc, fps, (width, height))

                start_frame = int(start * fps)
                end_frame = int(end * fps)

                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                for _ in range(start_frame, end_frame):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)

                cap.release()
                out.release()
                print(f"Saved clip: {output_scene_path}")

            except Exception as e:
                print(f"Error processing clip {output_scene_path}: {e}")


def reduce_repeated_characters(text, max_repeats=5):
    # 정규식을 사용하여 반복되는 문자를 최대 max_repeats로 줄임
    return re.sub(r"(.)\1{" + str(max_repeats) + r",}", r"\1" * max_repeats, text)


def process_video(video_path, output_json_path, timestamp_txt):
    # Load Whisper model
    whisper_model = whisper.load_model("large-v3")

    # Extract scene timestamps

    timestamps = read_timestamps_from_txt(timestamp_txt)

    video_id = os.path.splitext(os.path.basename(video_path))[0]

    # Prepare results
    results = []
    clip_id = 1  # Initialize clip ID

    for start, end in timestamps[video_id]:
        print(f"Processing scene from {start:.2f}s to {end:.2f}s...")
        text = transcribe_audio(video_path, start, end, whisper_model)
        text = reduce_repeated_characters(text)
        results.append(
            {
                "clip": clip_id,
                "start": round(start, 3),
                "end": round(end, 3),
                "text": text,
            }
        )
        clip_id += 1  # Increment clip ID

    # Save results to JSON file
    with open(output_json_path, "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=2)

    print(f"Results saved to {output_json_path}")
