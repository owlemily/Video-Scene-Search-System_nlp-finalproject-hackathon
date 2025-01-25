import json
import os

import streamlit as st
from PIL import Image

st.set_page_config(page_title="CaptionEval", layout="wide")

# Initialize session state
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "checks" not in st.session_state:
    st.session_state.checks = {}
if "file_name" not in st.session_state:
    st.session_state.file_name = "evaluation_results.json"


def load_json(file):
    return json.load(file)


def save_results(checks, total_scores, file_name):
    output_data = {"checks": checks, "total_scores": total_scores}
    output_folder = "./output"
    file_path = os.path.join(output_folder, file_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)


# File upload
st.title("CaptionEval Tool")
if "uploaded_json_file" not in st.session_state:
    uploaded_file = st.file_uploader("Upload a JSON file", type="json")
    if uploaded_file:
        st.session_state.uploaded_json_file = load_json(uploaded_file)

        for idx, item in enumerate(st.session_state.uploaded_json_file):
            if idx not in st.session_state.checks:
                st.session_state.checks[idx] = {
                    "video_id": item["video_id"],
                    "timestamp": item["timestamp"],
                    "frame_path": item["frame_image_path"],
                    "caption": item["caption"],
                    "caption_ko": item["caption_ko"],
                    "object_description": 0,
                    "object_features": 0,
                    "object_layout": 0,
                    "background_description": 0,
                    "necessary_information": 0,
                    "no_hallucination": 0,
                    "detailed_caption": 0,
                    "remarks": "",
                    "completed": 0,
                }
        st.rerun()

if "uploaded_json_file" in st.session_state:
    data = st.session_state.uploaded_json_file

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

    img_col, meta_col = st.columns([2, 3])

    # Show image
    with img_col:
        if os.path.exists(current_item["frame_image_path"]):
            image = Image.open(current_item["frame_image_path"])
            width, height = image.size
            new_size = (int(width * 0.9), int(height * 0.9))
            image = image.resize(new_size)
            st.image(image, use_container_width=False)

    # Display metadata
    with meta_col:
        st.text(f"Video ID: {current_item['video_id']}")
        st.text(f"Timestamp: {current_item['timestamp']}")
        st.text(f"Frame Path: {current_item['frame_image_path']}")
        st.text(f"Caption: {current_item['caption']}")
        st.text(f"Caption (Korean): {current_item['caption_ko']}")

    # Evaluation questions
    st.markdown("### Evaluation")
    current_checks = st.session_state.checks[current_index]

    questions = [
        "장면에 있는 모든 객체를 빠짐없이 표현했는가? (사람, 사물 등)",
        "객체의 특징을 빠짐없이 잘 설명하고 있는가? (색깔, 모양, 상태, ex 갈색머리의 여자)",
        "이미지 내 객체 배치(방향, 공간적 정보)가 정확한가?",
        "배경에 대한 설명이 있는가?",
        "필요없는 정보가 과도하게 포함되어 있는걸 막았는가?",
        "환각 현상이 발생하는 것을 막았는가?",
        "전반적으로 캡션이 자세한가?",
        "평가 완료",
    ]

    keys = [
        "object_description",
        "object_features",
        "object_layout",
        "background_description",
        "necessary_information",
        "no_hallucination",
        "detailed_caption",
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
        st.session_state.file_name = st.text_input(
            "Enter file name to save results", value=st.session_state.file_name
        )

        # Save button
        if st.button("Save Results"):
            total_scores = {
                key: sum(item[key] for item in st.session_state.checks.values())
                for key in keys
            }
            save_results(
                st.session_state.checks, total_scores, st.session_state.file_name
            )
            st.success(f"Results saved to {st.session_state.file_name}!")
