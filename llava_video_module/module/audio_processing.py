import os
import json
import torch
import whisper
import torchaudio
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from transformers import AutoModel, AutoTokenizer

from moviepy.video.io.VideoFileClip import VideoFileClip

def convert_to_mono(wav_path, output_path):
    """
    Converts an audio file to mono channel.

    :param wav_path: Path to the input audio file.
    :param output_path: Path to save the mono audio file.
    :return: Path to the mono audio file.
    """
    waveform, sample_rate = torchaudio.load(wav_path)
    if waveform.size(0) > 1:  # Check if multi-channel
        waveform = torch.mean(waveform, dim=0, keepdim=True)  # Convert to mono
    torchaudio.save(output_path, waveform, sample_rate)
    return output_path

def transcribe_audio(video_path, start_time, end_time, model):
    """
    Extract and transcribe audio for a specific time range using Whisper.

    :param video_path: Path to the input video file.
    :param start_time: Start time of the segment in seconds.
    :param end_time: End time of the segment in seconds.
    :param model: Whisper model instance.
    :return: Transcribed text.
    """
    temp_audio_path = "temp_audio.wav"
    temp_mono_audio_path = "temp_audio_mono.wav"

    # Extract audio segment using ffmpeg
    ffmpeg_command = f"ffmpeg -i \"{video_path}\" -ar 16000 -ac 2 -ss {start_time} -to {end_time} -y {temp_audio_path}"
    os.system(ffmpeg_command)

    # Convert to mono
    convert_to_mono(temp_audio_path, temp_mono_audio_path)

    # Transcribe audio with Whisper
    result = model.transcribe(temp_mono_audio_path, language='en')
    text = result.get("text", "").strip()

    # If no text is detected, return an empty string
    if not text:
        text = ""

    # Clean up temporary files
    os.remove(temp_audio_path)
    os.remove(temp_mono_audio_path)

    return text