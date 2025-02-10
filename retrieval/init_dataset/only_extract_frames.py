import os
from concurrent.futures import ThreadPoolExecutor

import cv2
from tqdm import tqdm


# 프레임 추출 함수
def extract_frames(video_path, output_dir, frame_rate):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"비디오 파일을 열 수 없습니다: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps / frame_rate) if fps > 0 else 1
    frame_count = 0
    saved_count = 0

    video_id = os.path.splitext(os.path.basename(video_path))[0]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval == 0:
            timestamp = round(frame_count / fps / (1 / frame_rate)) * (1 / frame_rate)
            frame_filename = f"{video_id}_{timestamp:.3f}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"{video_id}: 추출된 프레임 수 = {saved_count}")


# main 함수
def main():
    # 설정 로드
    video_folder = "video"  # 원본 비디오 파일 저장 경로 (code 기준 상위 디렉토리)
    if not os.path.exists(video_folder):
        print("비디오 폴더가 존재하지 않습니다. 먼저 download_video.sh를 실행하세요.")
        return
    frames_folder = "../datasets/frames_22"  # 프레임 이미지 저장 경로
    if not os.path.exists(frames_folder):
        os.makedirs(frames_folder)
    frame_rate = 4

    video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]

    print(f"총 {len(video_files)}개의 비디오 파일을 발견했습니다.")

    # 프레임 추출 - ThreadPoolExecutor를 사용하여 병렬 처리
    def process_video(video_file):
        video_path = os.path.join(video_folder, video_file)
        extract_frames(video_path, frames_folder, frame_rate)

    with ThreadPoolExecutor() as executor:
        # tqdm을 사용해 진행률 표시
        list(
            tqdm(
                executor.map(process_video, video_files),
                desc="비디오 프레임 추출 중",
                total=len(video_files),
            )
        )


if __name__ == "__main__":
    main()
