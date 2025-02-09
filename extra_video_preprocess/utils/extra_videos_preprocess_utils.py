import os
from concurrent.futures import ThreadPoolExecutor

import cv2
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
            frame_filename = f"{video_id}_{timestamp:.1f}.jpg"
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


from multiprocessing import Pool, cpu_count

from scenedetect import ContentDetector, SceneManager, open_video


def get_filled_scene_timestamps(video_path, threshold=30.0, min_scene_len=2):
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

    # Fill empty segments
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

    return video_path, filled_timestamps


def process_video(args):
    video_path, threshold, min_scene_len = args
    try:
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        _, timestamps = get_filled_scene_timestamps(
            video_path, threshold, min_scene_len
        )
        return [(video_id, start, end) for start, end in timestamps]
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return []


def save_timestamps_to_txt(
    video_folder, output_txt_path, threshold=30.0, min_scene_len=2
):
    video_files = [
        os.path.join(video_folder, video_name)
        for video_name in os.listdir(video_folder)
        if video_name.endswith(".mp4")
    ]

    num_workers = max(1, cpu_count() - 1)  # Use available CPU cores minus one

    with Pool(num_workers) as pool, open(output_txt_path, "w") as txt_file:
        tasks = [(video_path, threshold, min_scene_len) for video_path in video_files]

        for result in tqdm(
            pool.imap_unordered(process_video, tasks),
            total=len(video_files),
            desc="Processing Videos",
        ):
            for video_id, start, end in result:
                txt_file.write(f"{video_id}\t{start}\t{end}\n")

    print("All timestamps have been saved successfully.")


# timestamps.txt 파일 읽기
def read_timestamps(file_path):
    scenes = []
    scene_count = {}

    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue  # 유효하지 않은 라인 건너뛰기

            video_id, start_time, end_time = parts

            # 같은 video_id에 대한 scene 번호 증가
            if video_id not in scene_count:
                scene_count[video_id] = 1
            else:
                scene_count[video_id] += 1

            scene_id = f"{video_id}_{start_time}_{end_time}_{scene_count[video_id]:03d}"
            scene_path = f"scenes/{scene_id}.mp4"

            scenes.append(
                {
                    "video_id": video_id,
                    "start_time": start_time,
                    "end_time": end_time,
                    "scene_id": scene_id,
                    "scene_path": scene_path,
                    "caption": "",
                    "caption_ko": "",
                }
            )

    return scenes


# JSON 파일 생성
def create_json(timestamp_file, output_file):
    scenes = read_timestamps(timestamp_file)

    # 고정 값 설정
    model_path = "lmms-lab/LLaVA-Video-7B-Qwen2"
    prompt = (
        "Analyze the provided video, and describe in detail the movements and actions of objects and backgrounds.\n\n"
        "** Instructions **  \n"
        "1. Describe the movement and behaviour of all objects in the video. It clearly distinguishes who does what among the many characters. \n"
        "2. **Describe the movement of other objects** in scenes such as vehicles, animals, and inanimate objects. \n"
        "3. When describing actions, refer to the **camera's point of view** (e.g. ‘move to the left with the knife’). \n"
        "4. If the subject is expressing emotions (e.g., fear, excitement, aggression), describe the subject's facial expressions and body language. **If no emotions are detected, focus on the details of the movements.**\n"
        "5. If the subjects interact with each other (e.g., fighting, talking, helping), clearly describe the subjects' actions and the nature of the interaction. \n"
        "6. Do not speculate or guess about content not covered in the video (avoid hallucinations). \n\n"
        "**Example:**\n"
        "A woman with white hair and a purple dress is talking to a snowman in front of a bonfire. She is looking at the snowman melting because of the fire. The snowman is trying to pull up his face, which is dripping down, so that it doesn't collapse.\n"
    )
    max_new_tokens = 512
    max_num_frames = 48

    data = {
        "model_path": model_path,
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "max_num_frames": max_num_frames,
        "scenes": scenes,
    }

    return data

    # with open(output_file, "w", encoding="utf-8") as f:
    #     json.dump(data, f, indent=4, ensure_ascii=False)


# 사용 예시
if __name__ == "__main__":
    frames_output_folder = "./frames_fps10"
    os.makedirs(frames_output_folder, exist_ok=True)
    extract_frames_from_folder(
        video_folder="./video",
        frames_output_folder=frames_output_folder,
        frame_rate=10,
    )

    save_timestamps_to_txt(
        "./video_all", "./timestamps.txt", threshold=30.0, min_scene_len=2
    )

    # 실행
    timestamp_file = "timestamps.txt"
    output_file = "output.json"
    create_json(timestamp_file, output_file)
    print(f"JSON 파일이 생성되었습니다: {output_file}")
