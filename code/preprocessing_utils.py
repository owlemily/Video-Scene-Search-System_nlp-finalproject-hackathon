# Import necessary libraries
import os
import subprocess

import cv2
import torch
import torch.nn.functional as F
import torchaudio
import yaml
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector


# Common utility functions
def load_config(config_path):
    """Load configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[ERROR] Configuration file not found: {config_path}")
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"[ERROR] Invalid YAML file: {config_path}\n{e}")


def count_key_frames(key_frame_directory):
    """Count the total number of key frame files in the specified directory."""
    key_frame_files = [f for f in os.listdir(key_frame_directory) if f.endswith(".jpg")]
    return len(key_frame_files)


def get_total_duration(directory, file_extension=".mp4"):
    """Calculate the total duration of all files with the specified extension in a directory."""
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


def extract_scene_timestamps(video_path, threshold=30.0, min_scene_len=2):
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

    return timestamps


def save_timestamps_to_txt(input_dir, output_txt, threshold=30.0, min_scene_len=2):
    with open(output_txt, "w") as txt_file:
        for video_name in os.listdir(input_dir):
            if not video_name.endswith(".mp4"):
                continue

            video_path = os.path.join(input_dir, video_name)
            video_id = os.path.splitext(video_name)[0]

            try:
                timestamps = extract_scene_timestamps(
                    video_path, threshold, min_scene_len
                )

                # Merge consecutive timestamps to ensure no gaps
                merged_timestamps = []
                previous_end = 0.0

                for start, end in timestamps:
                    if start > previous_end:
                        merged_timestamps.append((previous_end, start))
                    merged_timestamps.append((start, end))
                    previous_end = end

                # Add final segment if there is a gap at the end
                total_duration = float(
                    cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT)
                ) / cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)

                if previous_end < total_duration:
                    merged_timestamps.append((previous_end, total_duration))

                # Write to txt file
                for start, end in merged_timestamps:
                    txt_file.write(f"{video_id}\t{start:.3f}\t{end:.3f}\n")

            except Exception as e:
                print(f"Error processing {video_name}: {e}")


def split_audio_from_txt(input_dir, output_dir, timestamp_txt):
    """Split audio from video files based on timestamps in a text file."""
    os.makedirs(output_dir, exist_ok=True)
    try:
        with open(timestamp_txt, "r") as txt_file:
            lines = txt_file.readlines()
    except FileNotFoundError:
        print(f"[ERROR] Timestamp file not found: {timestamp_txt}")
        return

    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            print(f"[WARNING] Skipping empty line at index {idx}")
            continue

        try:
            video_id, start, end = line.split("\t")
            start = float(start)
            end = float(end)
        except ValueError:
            print(f"[ERROR] Invalid line format at index {idx}: {line}")
            continue

        video_path = os.path.join(input_dir, f"{video_id}.mp4")
        output_audio_path = os.path.join(
            output_dir, f"{video_id}_scene_{start:.3f}-{end:.3f}.wav"
        )

        if not os.path.exists(video_path):
            print(f"[WARNING] Video file {video_path} not found. Skipping...")
            continue

        try:
            command = [
                "ffmpeg",
                "-i",
                video_path,
                "-ss",
                f"{start:.3f}",
                "-to",
                f"{end:.3f}",
                "-vn",
                "-acodec",
                "pcm_s16le",
                output_audio_path,
            ]
            subprocess.run(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
            )
            print(f"[INFO] Saved audio: {output_audio_path}")
        except subprocess.CalledProcessError as e:
            print(
                f"[ERROR] Error processing audio {output_audio_path}: {e.stderr.decode('utf-8')}"
            )


def split_scenes_from_txt(input_dir, output_dir, timestamp_txt):
    """Split video scenes based on timestamps specified in a text file."""
    os.makedirs(output_dir, exist_ok=True)

    with open(timestamp_txt, "r") as txt_file:
        for line in txt_file:
            video_id, start, end = line.strip().split("\t")
            start = float(start)
            end = float(end)

            video_path = os.path.join(input_dir, f"{video_id}.mp4")
            output_scene_path = os.path.join(
                output_dir, f"{video_id}_scene_{start:.3f}-{end:.3f}.mp4"
            )

            if not os.path.exists(video_path):
                print(f"[WARNING] Video file {video_path} not found. Skipping...")
                continue

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
                for frame_idx in range(start_frame, end_frame):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)

                cap.release()
                out.release()
                print(f"[INFO] Saved scene: {output_scene_path}")
            except Exception as e:
                print(f"[ERROR] Error processing scene {output_scene_path}: {e}")


def extract_key_frames(
    video_path,
    output_dir,
    processor,
    clip_model,
    similarity_threshold=0.85,
    stddev_threshold=10,
):
    """
    Extract key frames from a video using CLIP embeddings and save them to the output directory.

    Parameters:
        video_path (str): Path to the video file.
        output_dir (str): Directory to save the extracted key frames.
        processor: CLIPProcessor instance for preprocessing frames.
        clip_model: CLIPModel instance for generating embeddings.
        similarity_threshold (float): Threshold for similarity between consecutive frames (default=0.85).
        stddev_threshold (int): Threshold for determining solid color frames (default=10).
    """

    def is_solid_color(frame, stddev_threshold=10):
        """Check if the frame is a solid color based on standard deviation."""
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, stddev = cv2.meanStdDev(gray_frame)
        return stddev[0][0] < stddev_threshold

    def generate_clip_embedding(frame):
        """Generate CLIP embedding for the given frame."""
        inputs = processor(images=frame, return_tensors="pt").to("cuda")
        with torch.no_grad():
            embedding = clip_model.get_image_features(**inputs)
        return embedding

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open video file: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print(f"[ERROR] Invalid FPS for video file: {video_path}")
        cap.release()
        return

    frame_count = 0
    saved_count = 0
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    prev_embedding = None
    buffer_frame = None
    buffer_timestamp = None

    # Process video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            # Save the last buffered frame
            if buffer_frame is not None and not is_solid_color(
                buffer_frame, stddev_threshold
            ):
                frame_filename = f"{video_id}_{buffer_timestamp}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, buffer_frame)
                saved_count += 1
            break

        # Skip solid color frames
        if is_solid_color(frame, stddev_threshold):
            frame_count += 1
            continue

        # Generate CLIP embedding for the current frame
        current_embedding = generate_clip_embedding(frame)

        if prev_embedding is None:
            # First frame
            buffer_frame = frame
            buffer_timestamp = f"{frame_count / fps:.3f}"
            prev_embedding = current_embedding
        else:
            # Compare with the previous frame
            similarity = F.cosine_similarity(
                prev_embedding, current_embedding, dim=1
            ).item()
            if similarity >= similarity_threshold:
                # Update buffer if frames are similar
                buffer_frame = frame
                buffer_timestamp = f"{frame_count / fps:.3f}"
            else:
                # Save the buffered frame if frames are not similar
                if buffer_frame is not None:
                    frame_filename = f"{video_id}_{buffer_timestamp}.jpg"
                    frame_path = os.path.join(output_dir, frame_filename)
                    cv2.imwrite(frame_path, buffer_frame)
                    saved_count += 1

                # Update buffer with the current frame
                buffer_frame = frame
                buffer_timestamp = f"{frame_count / fps:.3f}"
                prev_embedding = current_embedding

        frame_count += 1

    cap.release()
    print(f"[INFO] {video_id}: Extracted {saved_count} key frames.")


def print_total_durations(config_path):
    """Print total durations of videos, scenes, and audio, and count total key frames."""
    config = load_config(config_path)

    # Directories
    input_directory = config["general"]["video_folder"]
    output_directory = config["scene_detection"]["full_scenes_folder"]
    audio_directory = config["scene_detection"]["audio_folder"]
    key_frames_directory = config["general"]["key_frames_folder"]

    # Calculate durations
    video_hours, video_minutes, video_seconds, video_milliseconds = get_total_duration(
        input_directory, ".mp4"
    )
    scene_hours, scene_minutes, scene_seconds, scene_milliseconds = get_total_duration(
        output_directory, ".mp4"
    )
    audio_hours, audio_minutes, audio_seconds, audio_milliseconds = get_total_duration(
        audio_directory, ".wav"
    )

    # Count key frames
    total_key_frames = count_key_frames(key_frames_directory)

    # Print results
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
