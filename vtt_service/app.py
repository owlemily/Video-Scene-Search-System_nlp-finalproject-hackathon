import gc
import os
import torch

import deepl
import streamlit as st
from googletrans import Translator
from vtt_service_utils import load_config, save_frame_at_time, convert_to_mono, trim_video_segment_and_save
from captioning import initialize_llava_video_model, single_scene_caption_LlavaVideo, load_qwen2_5_VL_model, single_frame_caption_Qwen2_5_VL

video_input_folder = "./video_input_folder"
temp_save_folder = "./temp_save_folder"

config = load_config("./config/base_config.yaml")

translator_name = config["general"]["translator_name"]

prompt = config["scene_caption"]["prompt"]
max_new_tokens = config["scene_caption"]["max_new_tokens"]
max_num_frames = config["scene_caption"]["max_num_frames"]
use_audio = config["scene_caption"]["audio"]["use_audio"]
mono_audio_folder = config["scene_caption"]["audio"]["mono_audio_folder"]

frame_prompt = config['frame_caption']['prompt']
frame_max_new_tokens = config['frame_caption']['max_new_tokens']

# 번역기 설정
if translator_name == "googletrans":
    translator = Translator()
elif translator_name == "deepl":
    auth_key = os.environ.get("DEEPL_API_KEY")
    translator = deepl.Translator(auth_key)


def clear_gpu_memory():
    """GPU 메모리 해제 함수"""
    if "scene_model" in st.session_state:
        del st.session_state["scene_model"]
    if "scene_tokenizer" in st.session_state:
        del st.session_state["scene_tokenizer"]
    if "scene_processor" in st.session_state:
        del st.session_state["scene_processor"]
    if "frame_model" in st.session_state:
        del st.session_state["frame_model"]
    if "frame_processor" in st.session_state:
        del st.session_state["frame_processor"]
    
    gc.collect()
    torch.cuda.empty_cache()


def load_scene_model():
    """Scene Captioning 모델 로드 (필요할 때만)"""
    if "scene_model" not in st.session_state:
        st.session_state["scene_tokenizer"], st.session_state["scene_model"], st.session_state["scene_processor"] = initialize_llava_video_model()


def load_frame_model():
    """Frame Captioning 모델 로드 (필요할 때만)"""
    if "frame_model" not in st.session_state:
        st.session_state["frame_model"], st.session_state["frame_processor"] = load_qwen2_5_VL_model()


# Streamlit UI
if "previous_page" not in st.session_state:
    st.session_state["previous_page"] = None

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Scene Captioning", "Frame Captioning"])

# 페이지 변경 감지 후 GPU 캐시 해제
if st.session_state["previous_page"] != page:
    clear_gpu_memory()
    st.session_state["previous_page"] = page

if page == "Scene Captioning":
    st.title("Scene Captioning")

    video_id = st.text_input("Video ID", "")
    start = st.text_input("Start Time", "")
    end = st.text_input("End Time", "")

    if st.button("Generate Caption"):
        load_scene_model()  # 필요할 때만 모델 로드
        video_path = os.path.join(video_input_folder, f"{video_id}.mp4")

        # 동영상 구간 저장
        trim_video_segment_and_save(video_path, start, end, temp_save_folder)

        temp_video_filename = f"{video_id}_{start}_{end}.mp4"
        temp_video_path = os.path.join(temp_save_folder, temp_video_filename)
        st.video(temp_video_path)

        # 오디오 변환
        temp_audio_path = os.path.join(temp_save_folder, f"{temp_video_filename[:-4]}_temp.wav")
        temp_mono_audio_path = os.path.join(temp_save_folder, f"{temp_video_filename[:-4]}.wav")

        duration = float(end) - float(start)
        ffmpeg_command = f'ffmpeg -i "{temp_video_path}" -to {duration} -vn -acodec pcm_s16le "{temp_audio_path}" -loglevel error'
        os.system(ffmpeg_command)
        convert_to_mono(temp_audio_path, temp_mono_audio_path)
        os.remove(temp_audio_path)

        caption = single_scene_caption_LlavaVideo(
            model=st.session_state["scene_model"],
            tokenizer=st.session_state["scene_tokenizer"],
            image_processor=st.session_state["scene_processor"],
            scene_path=temp_video_path,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            max_num_frames=max_num_frames,
            enable_audio_text=False,
            whisper_model=None,
            mono_audio_folder=mono_audio_folder,
            translator=translator,
        )
        st.subheader("Generated Caption:")
        st.write(caption)

        # 임시 파일 삭제
        os.remove(temp_video_path)
        os.remove(temp_mono_audio_path)

elif page == "Frame Captioning":
    st.title("Frame Captioning")

    video_id = st.text_input("Video ID", "xqsDUwDwdUM")
    timestamp = float(st.text_input("Timestamp", "0"))

    if st.button("Generate Frame Caption"):
        load_frame_model()  # 필요할 때만 모델 로드
        video_path = os.path.join(video_input_folder, f"{video_id}.mp4")
        image_filename = f"{video_id}_{timestamp}.jpg"
        temp_img_path = os.path.join(temp_save_folder, image_filename)

        save_frame_at_time(video_path, timestamp, temp_img_path)
        st.image(temp_img_path, caption=f"Frame at {timestamp}")

        caption = single_frame_caption_Qwen2_5_VL(
            model=st.session_state["frame_model"],
            processor=st.session_state["frame_processor"],
            frame_path=temp_img_path,
            prompt=frame_prompt,
            max_new_tokens=frame_max_new_tokens,
            translator=translator,
        )
        st.subheader("Generated Frame Caption:")
        st.write(caption)

        # 임시 이미지 삭제
        os.remove(temp_img_path)
