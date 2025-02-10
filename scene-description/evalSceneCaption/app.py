import json
import os

import streamlit as st

st.set_page_config(page_title="evalSceneCaption", layout="wide")

# Initialize session state
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "checks" not in st.session_state:
    st.session_state.checks = {}
if "file_name" not in st.session_state:
<<<<<<<< HEAD:refactor-frame-description/evalCaption/app.py
    st.session_state.file_name = "evaluation_results"
========
    st.session_state.file_name = "scene_evaluation"
>>>>>>>> feat-3/refactor-scene-description:scene-description/evalSceneCaption/app.py
if "evaluator_name" not in st.session_state:
    st.session_state.evaluator_name = ""
if "prompt_number" not in st.session_state:
    st.session_state.prompt_number = "1"
if "version_number" not in st.session_state:
    st.session_state.version_number = "1"


def load_json(file):
    return json.load(file)


def save_results(checks, total_scores, file_name):
    output_data = {
        "checks": checks,
        "total_scores": total_scores,
        "model_name": st.session_state.uploaded_json_file["model_path"],
        "prompt": st.session_state.uploaded_json_file["prompt"],
        "max_new_tokens": st.session_state.uploaded_json_file["max_new_tokens"],
<<<<<<<< HEAD:refactor-frame-description/evalCaption/app.py
========
        "max_num_frames": st.session_state.uploaded_json_file["max_num_frames"],
>>>>>>>> feat-3/refactor-scene-description:scene-description/evalSceneCaption/app.py
        "evaluator_name": st.session_state.evaluator_name,
        "prompt_number": st.session_state.prompt_number,
        "version_number": st.session_state.version_number,
    }
    output_folder = "./output"
    file_path = os.path.join(output_folder, file_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if os.path.exists(file_path):
        st.error(f"File already exists: {file_path}")
        return False
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    return True


# File upload
st.title("evalSceneCaption Tool")
if "uploaded_json_file" not in st.session_state:
    uploaded_file = st.file_uploader("Upload a JSON file", type="json")
    if uploaded_file:
        st.session_state.uploaded_json_file = load_json(uploaded_file)

<<<<<<<< HEAD:refactor-frame-description/evalCaption/app.py
        for idx, item in enumerate(st.session_state.uploaded_json_file["frames"]):
========
        for idx, item in enumerate(st.session_state.uploaded_json_file["scenes"]):
>>>>>>>> feat-3/refactor-scene-description:scene-description/evalSceneCaption/app.py
            if idx not in st.session_state.checks:
                st.session_state.checks[idx] = {
                    "video_id": item["video_id"],
                    "start_time": item["start_time"],
                    "end_time": item["end_time"],
                    "clip_id": item["clip_id"],
                    "scene_path": item["scene_path"],
                    "caption": item["caption"],
                    "caption_ko": item["caption_ko"],
                    "has_problem": 0,
                    "remarks": "",
                    "completed": 0,
                }
        st.rerun()

if "uploaded_json_file" in st.session_state:
<<<<<<<< HEAD:refactor-frame-description/evalCaption/app.py
    data = st.session_state.uploaded_json_file["frames"]
========
    data = st.session_state.uploaded_json_file["scenes"]
>>>>>>>> feat-3/refactor-scene-description:scene-description/evalSceneCaption/app.py

    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Previous") and st.session_state.current_index > 0:
            st.session_state.current_index -= 1
    with col2:
        if st.button("Next") and st.session_state.current_index < len(data) - 1:
            st.session_state.current_index += 1

    # Display the current JSON element
    current_index = st.session_state.current_index
    current_item = data[current_index]

    st.subheader(f"Item {current_index + 1} / {len(data)}")

    scene_col, meta_col = st.columns([2, 3])

    # Show scene
    with scene_col:
        scene_path = os.path.join("../", current_item["scene_path"])
        if os.path.exists(scene_path):
            st.video(scene_path)

    # Display metadata
    with meta_col:
        st.text(f"Video ID: {current_item['video_id']}")
        st.text(f"Start: {current_item['start_time']}")
        st.text(f"End: {current_item['end_time']}")
        st.text(f"Clip ID: {current_item['clip_id']}")
        st.text(f"Scene Path: {current_item['scene_path']}")
        st.text(f"Caption: {current_item['caption']}")
        st.text(f"Caption (Korean): {current_item['caption_ko']}")

    # Evaluation questions
    st.markdown("### Evaluation")
    st.text(f"Model: {st.session_state.uploaded_json_file['model_path']}")
    st.text(f"Prompt: {st.session_state.uploaded_json_file['prompt']}")
    st.text(f"Max New Tokens: {st.session_state.uploaded_json_file['max_new_tokens']}")
<<<<<<<< HEAD:refactor-frame-description/evalCaption/app.py
========
    st.text(f"Max Num Frames: {st.session_state.uploaded_json_file['max_num_frames']}")
>>>>>>>> feat-3/refactor-scene-description:scene-description/evalSceneCaption/app.py
    current_checks = st.session_state.checks[current_index]

    questions = [
        "문제가 있는가?",
        "평가 완료",
    ]

    keys = [
        "has_problem",
        "completed",
    ]

    for question, key in zip(questions, keys):
        current_checks[key] = st.checkbox(
            question, value=bool(current_checks[key]), key=f"{key}_{current_index}"
        )

    # Remarks section
    remarks = st.text_area(
        "기타 사항을 작성하세요:",
        value=current_checks.get("remarks", ""),
        key=f"remarks_{current_index}",
    )

    # Update session state for current item
    for key in keys:
        st.session_state.checks[current_index][key] = int(current_checks[key])
    st.session_state.checks[current_index]["remarks"] = remarks

    # Progress tracking
    completed_count = sum(1 for v in st.session_state.checks.values() if v["completed"])
    total_items = len(data)
    progress = (completed_count / total_items) * 100

    st.progress(progress / 100)
    st.text(f"Progress: {completed_count} / {total_items} ({progress:.2f}%)")

    # Show Save Results section only if progress is 100%
    if progress == 100:
        st.markdown("### Save Results")
        st.text(
            "결과파일은 파일명_p프롬브트번호_v버전번호_평가자이름.json 형식으로 자동으로 저장됩니다."
        )
        st.session_state.file_name = st.text_input(
            "평가결과를 저장할 파일명을 적어주세요 (.json은 빼고 적어주세요) (예: scene_evaluation)",
            value=st.session_state.file_name,
        )
        st.session_state.evaluator_name = st.text_input(
            "평가자 이름을 적어주세요", value=st.session_state.evaluator_name
        )
        st.session_state.prompt_number = st.text_input(
            "프롬프트 번호를 적어주세요", value=st.session_state.prompt_number
        )
        st.session_state.version_number = st.text_input(
            "버전 번호를 적어주세요", value=st.session_state.version_number
        )

        # Save button
        if st.button("Save Results"):
            total_scores = {
                key: sum(item[key] for item in st.session_state.checks.values())
                for key in keys
            }
            full_file_name = f"{st.session_state.file_name}_p{st.session_state.prompt_number}_v{st.session_state.version_number}_{st.session_state.evaluator_name}.json"
            success = save_results(
                st.session_state.checks, total_scores, full_file_name
            )
            if success:
                st.success(f"Results saved to {st.session_state.file_name}!")
            else:
                st.error("Failed to save results. File already exists.")
