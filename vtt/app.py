import gc
import os
import torch

import deepl
import streamlit as st
from googletrans import Translator
from utils.vtt_service_utils import load_config, save_frame_at_time, convert_to_mono, trim_video_segment_and_save
from utils.captioning import initialize_llava_video_model, initialize_whisper, single_scene_caption_LlavaVideo, load_qwen2_5_VL_model, single_frame_caption_Qwen2_5_VL

input_video_folder = "../input_video_folder"
extra_video_folder = "../extra_video_folder"
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


def get_video_path(video_id):
    """
    주어진 video_id에 대해 input_video_folder와 extra_video_folder를 순차적으로 확인합니다.
    존재하는 비디오 파일의 경로를 반환하며, 없으면 None을 반환합니다.
    """
    for folder in [input_video_folder, extra_video_folder]:
        video_path = os.path.join(folder, f"{video_id}.mp4")
        if os.path.exists(video_path):
            return video_path
    return None


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
    if "whisper" in st.session_state:
        del st.session_state["whisper"]
    
    gc.collect()
    torch.cuda.empty_cache()


def load_scene_model():
    """Scene Captioning 모델 로드 (필요할 때만)"""
    if "scene_model" not in st.session_state:
        with st.spinner("멋진 영상 캡션을 위해 모델을 불러오는 중입니다..."):
            st.session_state["scene_tokenizer"], st.session_state["scene_model"], st.session_state["scene_processor"] = initialize_llava_video_model()
            st.session_state["whisper"] = initialize_whisper()


def load_frame_model():
    """Frame Captioning 모델 로드 (필요할 때만)"""
    if "frame_model" not in st.session_state:
        with st.spinner("프레임 캡션을 위한 모델을 준비하는 중입니다..."):
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
    
    # 입력 방식 선택: 단일 입력 또는 TXT 파일 업로드
    input_mode = st.radio("Select Input Mode", ("Single Scene Input", "Batch Input from TXT File"))
    
    # 단일 입력 방식
    if input_mode == "Single Scene Input":
        video_id = st.text_input("Video ID", "s2wBtcmE5W8")
        start = st.text_input("Start Time", "2.34")
        end = st.text_input("End Time", "5.67")
        
        if st.button("Generate Caption for Single Scene"):
            with st.spinner("멋진 캡션을 위한 영상 모델을 준비 중입니다..."):
                load_scene_model()  # 필요할 때만 모델 로드
            
            video_path = get_video_path(video_id)
            if video_path is None:
                st.error(f"비디오 파일을 찾을 수 없습니다: {video_id}.mp4 (input/extra 폴더 모두 확인)")
            else:
                # 외부 동영상 사용 여부 확인
                if extra_video_folder in video_path:
                    st.info("외부동영상이 반영되었습니다!")
                    
                with st.spinner(f"{video_id} 영상 구간을 준비 중입니다..."):
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
                
                with st.spinner(f"{video_id} 캡션 생성 중..."):
                    caption = single_scene_caption_LlavaVideo(
                        model=st.session_state["scene_model"],
                        tokenizer=st.session_state["scene_tokenizer"],
                        image_processor=st.session_state["scene_processor"],
                        scene_path=temp_video_path,
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        max_num_frames=max_num_frames,
                        enable_audio_text=True,
                        whisper_model=st.session_state["whisper"],
                        mono_audio_path=temp_mono_audio_path,
                        translator=translator,
                    )
                st.subheader("Generated Caption:")
                st.write(caption[0])
                st.write(caption[1])

                # 임시 파일 삭제
                os.remove(temp_video_path)
                os.remove(temp_mono_audio_path)
    
    # Batch 입력 방식: TXT 파일 업로드
    else:
        txt_file = st.file_uploader("Upload a txt file (each line: video_id start end)", type=["txt"])
        if txt_file is not None:
            with st.spinner("모든 Scene 캡셔닝을 위한 모델을 불러오는 중입니다..."):
                load_scene_model()  # 필요할 때만 모델 로드
            try:
                lines = txt_file.read().decode("utf-8").splitlines()
            except Exception as e:
                st.error(f"파일을 읽는 중 오류 발생: {e}")
                lines = []
            
            results = {}
            
            for line in lines:
                if not line.strip():
                    continue  # 빈 줄 건너뛰기
                parts = line.strip().split()
                if len(parts) != 3:
                    st.error(f"잘못된 형식의 줄: {line} (형식: video_id start end)")
                    continue
                
                video_id, start, end = parts
                video_path = get_video_path(video_id)
                if video_path is None:
                    st.error(f"video_id '{video_id}'에 해당하는 비디오 파일이 존재하지 않습니다 (input/extra 폴더 모두 확인).")
                    continue
                
                # 외부 동영상 사용 여부 확인
                if extra_video_folder in video_path:
                    st.info(f"{video_id}: 외부동영상이 반영되었습니다!")
                
                with st.spinner(f"{video_id} 영상 구간을 준비 중입니다..."):
                    # 동영상 구간 저장
                    trim_video_segment_and_save(video_path, start, end, temp_save_folder)

                    temp_video_filename = f"{video_id}_{start}_{end}.mp4"
                    temp_video_path = os.path.join(temp_save_folder, temp_video_filename)
                    st.video(temp_video_path)

                    # 오디오 변환
                    temp_audio_path = os.path.join(temp_save_folder, f"{temp_video_filename[:-4]}_temp.wav")
                    temp_mono_audio_path = os.path.join(temp_save_folder, f"{temp_video_filename[:-4]}.wav")

                    try:
                        duration = float(end) - float(start)
                    except ValueError:
                        st.error(f"video_id '{video_id}'의 시작/종료 시간이 올바르지 않습니다: start={start}, end={end}")
                        continue
                    ffmpeg_command = f'ffmpeg -i "{temp_video_path}" -to {duration} -vn -acodec pcm_s16le "{temp_audio_path}" -loglevel error'
                    os.system(ffmpeg_command)
                    convert_to_mono(temp_audio_path, temp_mono_audio_path)
                    os.remove(temp_audio_path)
                
                with st.spinner(f"{video_id} 캡션 생성 중..."):
                    try:
                        caption = single_scene_caption_LlavaVideo(
                            model=st.session_state["scene_model"],
                            tokenizer=st.session_state["scene_tokenizer"],
                            image_processor=st.session_state["scene_processor"],
                            scene_path=temp_video_path,
                            prompt=prompt,
                            max_new_tokens=max_new_tokens,
                            max_num_frames=max_num_frames,
                            enable_audio_text=True,
                            whisper_model=st.session_state["whisper"],
                            mono_audio_path=temp_mono_audio_path,
                            translator=translator,
                        )
                        results[f"{video_id}_{start}_{end}"] = caption
                    except Exception as e:
                        st.error(f"video_id '{video_id}'의 캡셔닝 중 오류 발생: {e}")
                
                # 임시 파일 삭제
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
                if os.path.exists(temp_mono_audio_path):
                    os.remove(temp_mono_audio_path)
            
            if results:
                st.subheader("Generated Scene Captions:")
                for key, caption in results.items():
                    st.write(f"**{key}**: {caption}")
            else:
                st.info("캡셔닝된 결과가 없습니다.")

elif page == "Frame Captioning":
    st.title("Frame Captioning")
    
    # 입력 방식 선택: 단일 입력 또는 txt 파일 업로드
    input_mode = st.radio("Select Input Mode", ("Single Frame Input", "Batch Input from TXT File"))
    
    # 단일 입력 방식
    if input_mode == "Single Frame Input":
        video_id = st.text_input("Video ID", "xqsDUwDwdUM")
        timestamp_input = st.text_input("Timestamp", "0")
        
        if st.button("Generate Frame Caption for Single Frame"):
            try:
                timestamp = float(timestamp_input)
            except ValueError:
                st.error("Timestamp는 숫자여야 합니다.")
            else:
                with st.spinner("프레임 캡셔닝을 위해 준비 중입니다..."):
                    load_frame_model()  # 필요할 때만 모델 로드
                    video_path = get_video_path(video_id)
                    if video_path is None:
                        st.error(f"비디오 파일을 찾을 수 없습니다: {video_id}.mp4 (input/extra 폴더 모두 확인)")
                    else:
                        # 외부 동영상 사용 여부 확인
                        if extra_video_folder in video_path:
                            st.info("외부동영상이 반영되었습니다!")
                        
                        image_filename = f"{video_id}_{timestamp}.jpg"
                        temp_img_path = os.path.join(temp_save_folder, image_filename)
                        
                        # 지정한 timestamp의 프레임 추출
                        try:
                            save_frame_at_time(video_path, timestamp, temp_img_path)
                        except Exception as e:
                            st.error(f"프레임 추출 중 오류 발생: {e}")
                        else:
                            st.image(temp_img_path, caption=f"Frame at {timestamp} for video_id {video_id}")
                            
                            try:
                                caption = single_frame_caption_Qwen2_5_VL(
                                    model=st.session_state["frame_model"],
                                    processor=st.session_state["frame_processor"],
                                    frame_path=temp_img_path,
                                    prompt=frame_prompt,
                                    max_new_tokens=frame_max_new_tokens,
                                    translator=translator,
                                )
                                st.subheader("Generated Frame Caption:")
                                st.write(caption[0])
                                st.write(caption[1])
                            except Exception as e:
                                st.error(f"프레임 캡셔닝 중 오류 발생: {e}")
                            finally:
                                if os.path.exists(temp_img_path):
                                    os.remove(temp_img_path)
    
    # Batch 입력 방식: txt 파일 업로드
    else:
        txt_file = st.file_uploader("Upload a txt file (each line: video_id timestamp)", type=["txt"])
        if txt_file is not None:
            with st.spinner("모든 프레임 캡셔닝을 위한 모델을 불러오는 중입니다..."):
                load_frame_model()  # 필요할 때만 모델 로드
            try:
                lines = txt_file.read().decode("utf-8").splitlines()
            except Exception as e:
                st.error(f"파일을 읽는 중 오류 발생: {e}")
                lines = []
            
            results = {}
            
            for line in lines:
                if not line.strip():
                    continue  # 빈 줄 건너뛰기
                parts = line.strip().split()
                if len(parts) != 2:
                    st.error(f"잘못된 형식의 줄: {line} (형식: video_id timestamp)")
                    continue
                
                video_id, timestamp_str = parts
                try:
                    timestamp = float(timestamp_str)
                except ValueError:
                    st.error(f"video_id '{video_id}'의 timestamp가 숫자가 아닙니다: {timestamp_str}")
                    continue
                
                video_path = get_video_path(video_id)
                if video_path is None:
                    st.error(f"video_id '{video_id}'에 해당하는 비디오 파일이 존재하지 않습니다 (input/extra 폴더 모두 확인).")
                    continue
                
                # 외부 동영상 사용 여부 확인
                if extra_video_folder in video_path:
                    st.info(f"{video_id}: 외부동영상이 반영되었습니다!")
                
                image_filename = f"{video_id}_{timestamp}.jpg"
                temp_img_path = os.path.join(temp_save_folder, image_filename)
                
                # 지정한 timestamp의 프레임 추출
                try:
                    with st.spinner(f"{video_id}의 {timestamp}초 프레임을 추출 중입니다..."):
                        save_frame_at_time(video_path, timestamp, temp_img_path)
                except Exception as e:
                    st.error(f"비디오 '{video_id}'의 {timestamp}초에서 프레임을 추출하는 중 오류 발생: {e}")
                    continue
                
                # 추출한 프레임 이미지 표시 (원하는 경우)
                st.image(temp_img_path, caption=f"Frame at {timestamp} for video_id {video_id}")
                
                # 프레임 캡셔닝 수행
                try:
                    with st.spinner(f"{video_id} 프레임 캡션 생성 중..."):
                        caption = single_frame_caption_Qwen2_5_VL(
                            model=st.session_state["frame_model"],
                            processor=st.session_state["frame_processor"],
                            frame_path=temp_img_path,
                            prompt=frame_prompt,
                            max_new_tokens=frame_max_new_tokens,
                            translator=translator,
                        )
                    results[f"{video_id}_{timestamp}"] = caption
                except Exception as e:
                    st.error(f"video_id '{video_id}'의 프레임 캡셔닝 중 오류 발생: {e}")
                
                # 임시 이미지 파일 삭제
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
            
            if results:
                st.subheader("Generated Frame Captions:")
                for key, caption in results.items():
                    st.write(f"**{key}**: {caption}")
            else:
                st.info("캡셔닝된 결과가 없습니다.")