"""
extra_utils.py

함수 목록:
1. get_video_info (밖에서 쓰이는 함수)
2. extract_fps
3. extract_audio
4. change_audio_speed
5. change_fps (밖에서 쓰이는 함수)
7. count_key_frames
8. get_total_duration
9. print_total_durations (밖에서 쓰이는 함수)
10. extract_key_frames (밖에서 쓰이는 함수)
"""

import os
import subprocess

import cv2
import librosa
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from pydub import AudioSegment


def get_video_info(video_folder):
    """
    비디오 폴더의 모든 비디오 파일에 대한 정보를 출력합니다. (파일명, 해상도, FPS, 프레임 수, 시간)

    Args:
        video_folder (str): 비디오 폴더 경로

    Returns:
        None
    """
    if not os.path.exists(video_folder):
        print(f"[ERROR] 폴더가 존재하지 않습니다: {video_folder}")
        return

    video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]
    video_files.sort()
    if not video_files:
        print(f"[INFO] 폴더에 비디오 파일이 존재하지 않습니다: {video_folder}")
        return

    print(
        f"{'File Name':<30} {'Resolution':<20} {'FPS':<10} {'Frames':<15} {'Duration (s)':<15}"
    )
    print("-" * 90)

    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)

        # 비디오 파일 열기
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] 비디오 파일을 열 수 없습니다: {video_file}")
            continue

        # 비디오 속성 추출
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        # 비디오 정보 출력
        print(
            f"{video_file:<30} {width}x{height:<15} {fps:<11.3f} {frame_count:<15} {duration:<15.3f}"
        )

        cap.release()


def extract_fps(video_path):
    """
    OpenCV를 이용해 비디오의 FPS 추출

    Args:
        video_path (str): 비디오 파일 경로

    Returns:
        float: FPS 값
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def extract_audio(video_path, output_audio_path):
    """
    비디오 파일로부터 오디오 추출하여 output_audio_path에 저장

    Args:
        video_path (str): 입력 비디오 파일 경로
        output_audio_path (str): 추출될 오디오 파일 경로

    Returns:
        bool: 오디오 추출 성공 여부
    """
    try:
        audio = AudioSegment.from_file(video_path, format="mp4")
        audio.export(output_audio_path, format="wav")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to extract audio: {e}")
        return False


def change_audio_speed(input_audio_path, output_audio_path, speed_factor):
    """
    Librosa를 이용해 오디오 파일의 속도를 조정하여 저장

    Args:
        input_audio_path (str): 입력 오디오 파일 경로
        output_audio_path (str): 조정된 오디오 파일 경로
        speed_factor (float): 속도 조정 비율

    Returns:
        None
    """
    y, sr = librosa.load(input_audio_path, sr=None)  # 원본 오디오 로드
    y_fast = librosa.effects.time_stretch(y, rate=speed_factor)  # 스피드 조정
    sf.write(output_audio_path, y_fast, sr)  # 조정된 오디오 저장


def change_fps(input_video_dir, output_video_dir):
    """
    input_video_dir의 모든 비디오 파일의 FPS를 변경하여 output_video_dir에 저장하는 함수

    과정:
    1. OpenCV를 활용하여 비디오 파일을 배속하여 저장
    2. 비디오 파일에서 오디오 추출 후, 속도 조정하여 저장 (Librosa 사용)
    3. 조정된 비디오와 오디오를 합쳐서 저장 (FFmpeg 사용)

    Args:
        input_video_dir (str): 입력 비디오 폴더 경로
        output_video_dir (str): 출력 비디오 폴더 경로

    Returns:
        None
    """
    if not os.path.exists(input_video_dir):
        print(f"[ERROR] 입력 폴더가 존재하지 않습니다: {input_video_dir}")
        return

    os.makedirs(output_video_dir, exist_ok=True)

    video_files = [f for f in os.listdir(input_video_dir) if f.endswith(".mp4")]
    if not video_files:
        print(f"[INFO] 폴더에 비디오 파일이 존재하지 않습니다: {input_video_dir}")
        return

    for video_name in video_files:
        input_video_path = os.path.join(input_video_dir, video_name)
        temp_output_video_path = os.path.join(
            output_video_dir, f"processed_{video_name}"
        )
        temp_audio_path = os.path.join(output_video_dir, "temp_audio.wav")
        temp_audio_fast_path = os.path.join(output_video_dir, "temp_audio_fast.wav")

        # 원본 FPS 추출
        original_fps = extract_fps(input_video_path)
        if original_fps is None:
            print(f"[ERROR] Could not determine FPS for {video_name}. Skipping...")
            continue

        # 조정할 타겟 FPS 설정
        if 23 <= original_fps <= 25:
            target_fps = 25
        elif 29 <= original_fps <= 31:
            target_fps = 32
        elif 59 <= original_fps <= 61:
            target_fps = 64
        else:
            target_fps = (
                original_fps  # 만약 FPS가 23~25, 29~31, 59~61 사이가 아니면 그대로 유지
            )

        # 배속 비율 설정
        speed_factor = target_fps / original_fps

        # OpenCV를 이용해 비디오 열기
        cap = cv2.VideoCapture(input_video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 타겟 FPS으로 VideoWriter 설정
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MPEG-4 Codec
        out = cv2.VideoWriter(
            temp_output_video_path, fourcc, target_fps, (frame_width, frame_height)
        )

        # 모든 프레임을 유지하되 다른 FPS로 저장
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # 비디오 파일에서 오디오 추출 후, 속도 조정하여 저장
        if extract_audio(input_video_path, temp_audio_path):
            change_audio_speed(temp_audio_path, temp_audio_fast_path, speed_factor)

            # 조정된 비디오와 오디오를 합쳐서 저장
            final_output_video_path = os.path.join(output_video_dir, video_name)
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    temp_output_video_path,
                    "-i",
                    temp_audio_fast_path,
                    "-c:v",
                    "libx264",
                    "-crf",
                    "23",
                    "-preset",
                    "fast",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "128k",
                    final_output_video_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # 임시 파일 삭제
            os.remove(temp_audio_path)
            os.remove(temp_audio_fast_path)
            os.remove(temp_output_video_path)

            print(
                f"[INFO] Processed and saved: {final_output_video_path} with FPS {target_fps}"
            )


def count_key_frames(key_frame_directory):
    """
    Key Frame 폴더의 모든 키프레임 파일 수를 세는 함수
    """
    key_frame_files = [f for f in os.listdir(key_frame_directory) if f.endswith(".jpg")]
    return len(key_frame_files)


def get_total_duration(directory, file_extension=".mp4"):
    """
    입력 폴더의 모든 파일의 총 시간을 계산하는 함수

    Args:
        directory (str): 입력 폴더 경로
        file_extension (str): 파일 확장자 (".mp4", ".wav")

    Returns:
        tuple: (시간, 분, 초, 밀리초)
    """
    total_duration_ms = 0
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path) and file.endswith(file_extension):
            try:
                if file_extension == ".mp4":
                    video = cv2.VideoCapture(file_path)
                    if video.isOpened():
                        fps = video.get(cv2.CAP_PROP_FPS)
                        frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
                        if fps > 0 and frame_count > 0:
                            duration_ms = (frame_count / fps) * 1000
                            total_duration_ms += duration_ms
                    video.release()
                elif file_extension == ".wav":
                    waveform, sample_rate = torchaudio.load(file_path)
                    duration_ms = (waveform.size(1) / sample_rate) * 1000
                    total_duration_ms += duration_ms
            except Exception as e:
                print(f"[ERROR] Could not process {file_path}: {e}")
    total_seconds = total_duration_ms / 1000
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int(total_duration_ms % 1000)
    return hours, minutes, seconds, milliseconds


def print_total_durations(video_folder, scene_folder, audio_folder, key_frames_folder):
    """
    각 폴더의 총 시간과 키프레임 개수를 출력하는 함수

    Args:
        video_folder (str): 비디오 폴더 경로
        scene_folder (str): Scene 폴더 경로
        audio_folder (str): 오디오 폴더 경로
        key_frames_folder (str): 키프레임 폴더 경로

    Returns:
        None
    """

    # 각 폴더의 총 시간 계산
    video_hours, video_minutes, video_seconds, video_milliseconds = get_total_duration(
        video_folder, ".mp4"
    )
    scene_hours, scene_minutes, scene_seconds, scene_milliseconds = get_total_duration(
        scene_folder, ".mp4"
    )
    audio_hours, audio_minutes, audio_seconds, audio_milliseconds = get_total_duration(
        audio_folder, ".wav"
    )

    # 키 프레임 개수 계산
    total_key_frames = count_key_frames(key_frames_folder)

    # 결과 출력
    print(
        f"[INFO] Total video duration: {video_hours}h {video_minutes}m {video_seconds}s {video_milliseconds}ms"
    )
    print(
        f"[INFO] Total scene duration: {scene_hours}h {scene_minutes}m {scene_seconds}s {scene_milliseconds}ms"
    )
    print(
        f"[INFO] Total audio duration: {audio_hours}h {audio_minutes}m {audio_seconds}s {audio_milliseconds}ms"
    )
    print(f"[INFO] Total key frames: {total_key_frames} frames")


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
