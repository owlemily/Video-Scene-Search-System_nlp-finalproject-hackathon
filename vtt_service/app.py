import gc
import os

import deepl
import streamlit as st
from googletrans import Translator
from modules.frame_utils import save_frame_at_time
from modules.scene_caption_modules import initialize_model, single_scene_caption
from modules.scene_utils import save_video_scenes_by_timestamps
from moudles.audio_utils import convert_to_mono

video_input_folder = "./video_input_folder"
temp_save_folder = "temp_save_folder"

config = load_config("./config/evalSceneCaption.yaml")

video_folder = config["general"]["video_folder"]
translator_name = config["general"]["translator_name"]

timestamp_file = config["scene_caption"]["timestamp_file"]
scene_folder = config["scene_caption"]["scene_folder"]
output_folder = config["scene_caption"]["output_folder"]
scene_output_filename = config["scene_caption"]["scene_output_filename"]

model_path = config["scene_caption"]["model"]

prompt = config["scene_caption"]["prompt"]
max_new_tokens = config["scene_caption"]["max_new_tokens"]
max_num_frames = config["scene_caption"]["max_num_frames"]

use_audio = config["scene_caption"]["audio"]["use_audio"]
mono_audio_folder = config["scene_caption"]["audio"]["mono_audio_folder"]

scene_info_with_audio_scripts_file = config["scene_caption"]["audio"][
    "scene_info_with_audio_scripts_file"
]

if translator_name == "googletrans":
    translator = Translator()
elif translator_name == "deepl":
    auth_key = os.environ.get("DEEPL_API_KEY")
    translator = deepl.Translator(auth_key)


def generate_frame_caption(video_id, timestamp):
    """내부 함수: 프레임 캡션을 생성하는 함수 (예제용)."""
    return f"Caption for frame {video_id} at {timestamp}"


@st.cache_resource
def load_scene_model():
    return initialize_model(model_path)


def clear_cache():
    if "current_model" in st.session_state:
        del st.session_state.current_model
        gc.collect()


if __name__ == "__main__":
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Scene Captioning", "Frame Captioning"])
    clear_cache()

    if page == "Scene Captioning":
        st.title("Scene Captioning")

        # 사용자 입력
        video_id = st.text_input("Video ID", "")
        start = float(st.text_input("Start Time", ""))
        end = float(st.text_input("End Time", ""))

        if st.button("Generate Caption"):
            video_path = os.path.join(video_input_folder, f"{video_id}.mp4")
            # 해당 구간 동영상 임시로 저장
            save_video_scenes_by_timestamps(
                video_path, [(start, end)], temp_save_folder
            )

            # 동영상 보여주기
            temp_video_filename = f"{video_id}_{start}_{end}_001.mp4"
            temp_video_path = os.path.join(temp_save_folder, temp_video_filename)
            st.video(temp_video_path)

            # 해당 구간 모노오디오 임시로 저장
            temp_audio_path = os.path.join(
                temp_save_folder, f"{temp_video_filename[:-4]}_temp.wav"
            )
            temp_mono_audio_path = os.path.join(
                temp_save_folder, f"{temp_video_filename[:-4]}.wav"
            )

            duaration = float(end) - float(start)
            # mp4 파일(Scene)에서 오디오 추출하는 ffmpeg 명령어 (temp_audio_path에 저장)
            ffmpeg_command = f'ffmpeg -i "{temp_video_path}" -to {duaration} -vn -acodec pcm_s16le "{temp_audio_path}" -loglevel error'
            os.system(ffmpeg_command)
            convert_to_mono(temp_audio_path, temp_mono_audio_path)

            os.remove(temp_audio_path)

            model, tokenizer, image_processor = load_scene_model

            caption = single_scene_caption(
                model_path,
                model,
                tokenizer,
                image_processor,
                temp_video_path,
                prompt,
                max_new_tokens,
                max_num_frames,
                use_audio,
                mono_audio_folder,
                None,
                translator,
            )["caption_ko"]

            st.subheader("Generated Caption:")
            st.write(caption)

            # 임시 동영상 삭제
            os.remove(temp_video_path)
            os.remove(temp_mono_audio_path)

    elif page == "Frame Captioning":
        st.title("Frame Captioning")

        # 사용자 입력
        video_id = st.text_input("Video ID", "example_video_frame")
        timestamp = st.text_input("Timestamp", "0")

        if st.button("Generate Frame Caption"):
            video_path = os.path.join(video_input_folder, f"{video_id}.mp4")
            # 파일명 생성
            image_filename = f"{video_id}_{timestamp}.jpg"
            temp_img_path = os.path.join(temp_save_folder, image_filename)

            # 해당 타임스탬프의 프레임 이미지 임시로 저장
            save_frame_at_time(video_path, timestamp, temp_img_path)

            st.image(temp_img_path, caption=f"Frame at {timestamp}")

            # 캡션 생성
            caption = generate_frame_caption(video_id, timestamp)
            st.subheader("Generated Frame Caption:")
            st.write(caption)

            # 임시 이미지 삭제
            os.remove(temp_img_path)
