import os
import cv2
import subprocess
import streamlit as st
import deepl
from googletrans import Translator

# retrieval 관련 객체들
from code.video_retrieval import SCENERetrieval, BLIPRetrieval, CLIPRetrieval, Rankfusion
# Advanced re-ranking 함수 (Qwen‑VL 기반)
from code.rerank import rerank_with_qwen, get_qwen_models, get_embedding_model

# ----------------------------------
# 0. 쿼리 번역 함수 정의 (googletrans 또는 DeepL 이용)
# ----------------------------------
def translate_query(query: str, translator, target_lang="en") -> str:
    try:
        if isinstance(translator, Translator):  # googletrans 사용
            return translator.translate(query, dest=target_lang).text
        elif hasattr(translator, "translate_text"):  # DeepL 사용 (deepl.Translator)
            if target_lang == "en":
                target_lang_for_deepl = "EN-US"
            return translator.translate_text(query, target_lang=target_lang_for_deepl).text
        else:
            raise ValueError("지원되지 않는 번역기 객체입니다.")
    except Exception as e:
        st.error(f"번역 실패: {query}. 오류: {e}")
        return ""

# ----------------------------------
# 1. 임베딩 및 Retrieval 객체 캐싱 (한 번만 로드)
# ----------------------------------
@st.cache_resource
def load_retrievers(config_path):
    """
    retrieval 객체들을 초기화하고 Rankfusion 객체를 생성한 후 반환합니다.
    이 함수는 캐싱되어 이후 호출 시 재로딩하지 않습니다.
    """
    # (1) 각 모달리티별 retrieval 객체 생성
    scene_retriever = SCENERetrieval(config_path=config_path)
    clip_retriever = CLIPRetrieval(config_path=config_path)
    blip_retriever = BLIPRetrieval(config_path=config_path)

    # (2) Rankfusion 초기화 (script_retriever 포함)
    rankfusion = Rankfusion(
        scene_retriever=scene_retriever,
        clip_retriever=clip_retriever,
        blip_retriever=blip_retriever,
        weight_clip=0.65, 
        weight_blip=0.34,
        weight_scene=0.1,
    )
    return {"rankfusion": rankfusion}

@st.cache_resource
def load_for_rerank():
    embedding_model = get_embedding_model()
    qwen_model, _, qwen_processor = get_qwen_models()
    return qwen_model, qwen_processor, embedding_model

# ----------------------------------
# 2. 프레임 및 scene 추출 함수 정의
# ----------------------------------
def save_frame_at_time(video_path, time_sec, output_path):
    """
    비디오 파일에서 특정 시간의 프레임을 저장하는 함수.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"비디오 파일을 열 수 없습니다: {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(fps * time_sec)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    success, frame = cap.read()
    if not success:
        cap.release()
        st.error("해당 시간은 비디오의 범위를 벗어납니다.")
        return False

    cv2.imwrite(output_path, frame)
    cap.release()
    return True

def trim_video_segment_and_save(video_path, start, end, output_scene_folder):
    """
    원본 비디오에서 start ~ end 구간의 동영상 부분을 잘라서 저장하는 함수.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"입력 비디오 파일이 존재하지 않습니다: {video_path}")

    os.makedirs(output_scene_folder, exist_ok=True)

    video_id = os.path.splitext(os.path.basename(video_path))[0]
    if video_id.startswith("-"):
        video_id = "_" + video_id.lstrip("-")

    output_scene_path = os.path.join(output_scene_folder, f"{video_id}_{start}_{end}.mp4")
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",  # 기존 파일 덮어쓰기
        "-loglevel", "error",  # 오류 메시지만 출력
        "-i", video_path,  # 입력 파일
        "-ss", str(start),  # 시작 시간
        "-to", str(end),    # 종료 시간
        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "fast",
        "-c:a", "aac",
        "-b:a", "192k",
        "-force_key_frames", f"expr:gte(t,{start})",
        output_scene_path,
    ]
    try:
        subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    except subprocess.CalledProcessError as e:
        st.error(f"FFmpeg 실행 중 오류 발생:\n{e.stderr}")
        raise

    return output_scene_path

# ----------------------------------
# 3. 경로 설정 및 기본 폴더 생성
# ----------------------------------
video_folder = "/data/ephemeral/home/original_videos"        # 동영상 파일들이 저장된 폴더
extra_video_folder = "/data/ephemeral/home/external_videos"     # 추가 동영상 파일들이 저장된 폴더
temp_frame_folder = "temp_frame_folder"         # 추출한 프레임 이미지 저장 폴더
temp_scene_folder = "temp_scene_folder"         # 잘라낸 scene 영상 저장 폴더

os.makedirs(temp_frame_folder, exist_ok=True)
os.makedirs(temp_scene_folder, exist_ok=True)

def get_video_path(video_id):
    """
    주어진 video_id에 대해 video_folder와 extra_video_folder에서 해당 mp4 파일 경로를 반환합니다.
    두 폴더 모두에 없으면 None을 반환합니다.
    """
    video_filename = f"{video_id}.mp4"
    primary_path = os.path.join(video_folder, video_filename)
    if os.path.exists(primary_path):
        return primary_path
    secondary_path = os.path.join(extra_video_folder, video_filename)
    if os.path.exists(secondary_path):
        return secondary_path
    return None

# ----------------------------------
# 4. 번역기 설정
# ----------------------------------
translator_name = "deepl"
if translator_name == "googletrans":
    translator = Translator()
elif translator_name == "deepl":
    auth_key = os.environ.get("DEEPL_API_KEY")
    translator = deepl.Translator(auth_key)
else:
    translator = None

# ----------------------------------
# 5. Streamlit UI: 쿼리 입력 및 검색 결과 출력
# ----------------------------------
st.title("Video Search")
st.markdown("아래 칸에 쿼리를 입력하고 **검색** 버튼을 누르면, Rankfusion을 통해 결과(이미지 및 scene)를 보여줍니다.")

# Advanced Rerank 옵션 (체크 시, 검색 후 후보 이미지들로 Qwen‑VL rerank 수행)
advanced_rerank = st.checkbox("Advanced Search with Rerank", value=False)

# 세션 상태: 이전 검색 결과 존재 여부 추적 (검색 시 임시 폴더 파일을 삭제)
if "prev_search_done" not in st.session_state:
    st.session_state.prev_search_done = False

# 4-1. 임베딩 및 Retrieval 객체 로딩 (한 번만 로드)
config_path = "config/video_retrieval_config.yaml"  # 설정 파일 경로
with st.spinner("임베딩 및 retrieval 객체를 불러오는 중..."):
    retrievers = load_retrievers(config_path)
    qwen_model, qwen_processor, embedding_model = load_for_rerank()
    rankfusion = retrievers["rankfusion"]

# 4-2. 쿼리 입력 위젯 (기본 예시 쿼리)
query = st.text_input("쿼리를 입력하세요:", 
                      "A woman grabs a man from behind and holds a knife to his throat, threatening him.")

# 4-3. 검색 버튼 클릭 시
if st.button("검색"):
    # 이전 검색 결과가 있으면, 이번 검색 시작 전에 임시 폴더 내 파일 삭제
    if st.session_state.prev_search_done:
        for folder in [temp_frame_folder, temp_scene_folder]:
            for file_name in os.listdir(folder):
                file_path = os.path.join(folder, file_name)
                try:
                    os.remove(file_path)
                except Exception as e:
                    st.warning(f"임시 파일 삭제 실패: {e}")
    st.session_state.prev_search_done = True

    if not query.strip():
        st.error("쿼리를 입력하세요.")
    else:
        # 입력 쿼리를 영어로 번역 (필요 시)
        translated_query = translate_query(query, translator)
        st.write("번역된 쿼리:", translated_query)

        # (1) Fusion 검색 수행: Rankfusion 객체를 통해 결과(여러 모달리티의 점수 합산)를 얻음.
        fusion_results = rankfusion.retrieve(query=translated_query, top_k=1000, union_top_n=50000)

        # (2) Diverse 결과 선택 (예시: 클러스터링을 통한 최종 10개 선택)
        diverse_results = rankfusion.select_diverse_results_by_clustering(
            fusion_results, desired_num=10, top_n=500
        )

        # -------------------------------------------
        # 클러스터링 기반 Diversity 최종 결과 (Top 5) 표시
        # -------------------------------------------
        st.subheader("클러스터링 기반 Diversity 최종 결과 (Top 5)")
        for res in diverse_results[:5]:
            st.markdown(f"### Rank {res['rank']} – {res['image_filename']}")
            # 파일명에서 video_id와 timestamp 추출 (파일명 형식: videoID_timestamp)
            filename = os.path.basename(res['image_filename'])
            base = os.path.splitext(filename)[0]
            parts = base.rsplit("_", 1)
            if len(parts) != 2:
                st.warning("파일명 형식이 올바르지 않습니다.")
                continue

            video_id, time_str = parts
            try:
                time_sec = float(time_str)
            except ValueError:
                st.error("파일명에 포함된 timestamp 변환에 실패했습니다.")
                continue

            video_path = get_video_path(video_id)
            if video_path is None:
                st.error(f"동영상 파일을 찾을 수 없습니다: {video_id}.mp4 in {video_folder} or {extra_video_folder}")
                continue

            # scene 정보가 있으면 scene 영상과 프레임 이미지를 같이 보여줌
            if res.get("scene_info") and res["scene_info"].get("scene_id"):
                scene_id = res["scene_info"].get("scene_id")
                # st.write(f"Scene 정보: {scene_id}")
                scene_output_path = None
                scene_parts = scene_id.split("_")
                if len(scene_parts) >= 3:
                    try:
                        # 올바른 형식: videoID_sceneStart_sceneEnd_someIndex
                        scene_start = float(scene_parts[-3])
                        scene_end = float(scene_parts[-2])
                    except ValueError:
                        st.error("scene_start 또는 scene_end 값 변환에 실패했습니다.")
                        continue
                    # st.write(f"Scene 구간: {scene_start}초 ~ {scene_end}초")
                    try:
                        scene_output_path = trim_video_segment_and_save(
                            video_path, scene_start, scene_end, temp_scene_folder
                        )
                    except Exception as e:
                        st.error(f"Scene 추출 실패: {e}")
                else:
                    st.warning("scene_id 형식이 올바르지 않습니다. Scene 추출을 건너뜁니다.")

                # 프레임 이미지 추출 (항상 추출)
                frame_output_path = os.path.join(temp_frame_folder, f"{video_id}_{time_sec}.jpg")
                try:
                    save_frame_at_time(video_path, time_sec, frame_output_path)
                except Exception as e:
                    st.error(f"Frame 추출 실패: {e}")
                    frame_output_path = None

                # scene과 frame이 모두 추출되었으면 side-by-side로 보여줌 (왼쪽: frame, 오른쪽: scene)
                if scene_output_path and frame_output_path:
                    col1, col2 = st.columns(2)
                    with col1:
                        with open(frame_output_path, "rb") as f:
                            frame_bytes = f.read()
                        st.image(frame_bytes)
                    with col2:
                        with open(scene_output_path, "rb") as f:
                            scene_bytes = f.read()
                        st.video(scene_bytes)
                elif frame_output_path:
                    with open(frame_output_path, "rb") as f:
                        frame_bytes = f.read()
                    st.image(frame_bytes)
                elif scene_output_path:
                    with open(scene_output_path, "rb") as f:
                        scene_bytes = f.read()
                    st.video(scene_bytes)
            else:
                st.write(f"프레임 추출 시간: {time_sec}초")
                frame_output_path = os.path.join(temp_frame_folder, f"{video_id}_{time_sec}.jpg")
                try:
                    save_frame_at_time(video_path, time_sec, frame_output_path)
                    with open(frame_output_path, "rb") as f:
                        frame_bytes = f.read()
                    st.image(frame_bytes)
                except Exception as e:
                    st.error(f"Frame 추출 실패: {e}")

        # -------------------------------------------
        # Advanced Rerank 기능: Diverse 결과 이후, 가장 마지막에 표시
        # -------------------------------------------
        if advanced_rerank:
            st.subheader("Advanced Search with Rerank 진행 중...")
            candidate_image_paths = []
            candidate_mapping = {}  # candidate 이미지 경로와 해당 diverse result를 매핑
            # 성능을 위해 diverse_results 상위 5개의 후보 이미지를 사용
            for res in diverse_results[:5]:
                filename = os.path.basename(res['image_filename'])
                base = os.path.splitext(filename)[0]
                parts = base.rsplit("_", 1)
                if len(parts) != 2:
                    continue
                video_id, time_str = parts
                try:
                    time_sec = float(time_str)
                except ValueError:
                    continue
                video_path = get_video_path(video_id)
                if video_path is None:
                    st.error(f"동영상 파일을 찾을 수 없습니다: {video_id}.mp4 in {video_folder} or {extra_video_folder}")
                    continue
                frame_output_path = os.path.join(temp_frame_folder, f"{video_id}_{time_sec}.jpg")
                if not os.path.exists(frame_output_path):
                    if not save_frame_at_time(video_path, time_sec, frame_output_path):
                        continue
                candidate_image_paths.append(frame_output_path)
                candidate_mapping[frame_output_path] = res

            if candidate_image_paths:
                best_image_path, rerank_message = rerank_with_qwen(
                    translated_query, candidate_image_paths, qwen_processor, qwen_model, embedding_model
                )
                st.subheader("Advanced Search Result with Rerank")
                st.markdown(rerank_message)
                # best_image_path와 연결된 diverse result 정보 획득
                best_res = candidate_mapping.get(best_image_path)
                # best_image_path에서 video_id와 timestamp 추출
                filename = os.path.basename(best_image_path)
                base = os.path.splitext(filename)[0]
                parts = base.rsplit("_", 1)
                if len(parts) != 2:
                    st.warning("파일명 형식이 올바르지 않습니다.")
                else:
                    video_id, time_str = parts
                    try:
                        time_sec = float(time_str)
                    except ValueError:
                        st.error("파일명에 포함된 timestamp 변환에 실패했습니다.")
                        time_sec = None
                    video_path = get_video_path(video_id)
                    if video_path is None:
                        st.error(f"동영상 파일을 찾을 수 없습니다: {video_id}.mp4 in {video_folder} or {extra_video_folder}")
                    else:
                        scene_output_path = None
                        if best_res and best_res.get("scene_info") and best_res["scene_info"].get("scene_id"):
                            scene_id = best_res["scene_info"]["scene_id"]
                            scene_parts = scene_id.split("_")
                            if len(scene_parts) >= 3:
                                try:
                                    scene_start = float(scene_parts[-3])
                                    scene_end = float(scene_parts[-2])
                                except ValueError:
                                    st.error("scene_start 또는 scene_end 값 변환에 실패했습니다.")
                                else:
                                    try:
                                        scene_output_path = trim_video_segment_and_save(
                                            video_path, scene_start, scene_end, temp_scene_folder
                                        )
                                    except Exception as e:
                                        st.error(f"Scene 추출 실패: {e}")
                            else:
                                st.warning("scene_id 형식이 올바르지 않습니다. Scene 추출을 건너뜁니다.")

                        # best_image_path에 저장된 프레임 이미지 표시 (scene이 있으면 side-by-side)
                        if scene_output_path and os.path.exists(scene_output_path):
                            col1, col2 = st.columns(2)
                            with col1:
                                with open(best_image_path, "rb") as f:
                                    frame_bytes = f.read()
                                st.image(frame_bytes)
                            with col2:
                                with open(scene_output_path, "rb") as f:
                                    scene_bytes = f.read()
                                st.video(scene_bytes)
                        else:
                            with open(best_image_path, "rb") as f:
                                frame_bytes = f.read()
                            st.image(frame_bytes)
            else:
                st.warning("Advanced Rerank: 후보 이미지가 없습니다.")

        # ------------------------------
        # Fusion 검색 결과 (요약): 기본에는 감추고, 버튼(Expander) 클릭 시 표시
        # ------------------------------
        with st.expander("=== Fusion 검색 결과 (요약) 보기 ===", expanded=False):
            for res in diverse_results[:30]:
                scene_id = res.get("scene_info", {}).get("scene_id") if res.get("scene_info") else None
                st.write(
                    f"Rank {res['rank']}: {res['image_filename']} | "
                    f"clip={res['clip_score']:.4f}, blip={res['blip_score']:.4f}, "
                    f"scene={res['scene_score']:.4f}, fusion={res['fusion_score']:.4f} "
                    f" | scene_id={scene_id}"
                )
