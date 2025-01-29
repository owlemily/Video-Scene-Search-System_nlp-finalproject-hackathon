"""
scene_caption_modules.py

함수 목록:
1. initialize_model
2. single_scene_caption_InternVideo2
3. single_scene_caption_LlavaVideo
4. single_scene_caption_InternVideo2_5_Chat
5. single_scene_caption
6. scene_caption - 메인 함수
"""

import json
import logging
import os

import torch
from googletrans import Translator
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from .audio_utils import transcribe_audio
from .specific_model_utils.InternVideo2_5_Chat_utils import DescriptionGenerator
from .specific_model_utils.InternVideo2_utils import load_video
from .specific_model_utils.LlavaVideo_utils import (
    get_video_and_input_ids,
    load_llava_video_model,
)

transformers_logger = logging.getLogger("transformers")


def initialize_model(model_path="OpenGVLab/InternVideo2-Chat-8B"):
    """
    model_path를 받아 모델과 tokenizer를 반환하는 함수

    Args:
        model_path (str): 모델 경로

    Returns:
        model, tokenizer
    """
    if model_path == "OpenGVLab/InternVideo2-Chat-8B":
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
        model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).cuda()
        image_processor = None

    # LlavaVideo 모델에서는 자체적으로 tokenizer, model을 불러옴
    elif model_path == "lmms-lab/LLaVA-Video-7B-Qwen2":
        tokenizer, model, image_processor, max_length = load_llava_video_model()

    elif model_path == "OpenGVLab/InternVideo2_5_Chat_8B":
        generator = DescriptionGenerator(model_path=model_path)
        return generator, None, None

    return model, tokenizer, image_processor


def single_scene_caption_InternVideo2(
    model,
    tokenizer,
    scene_path,
    prompt,
    generation_config,
    use_audio_for_prompt,
    mono_audio_folder,
    scene_info_json_file_path=None,  # 오디오 스크립트 정보 포함
):
    """
    1개의 scene에 대한 캡션을 InternVideo2 모델을 사용하여 생성하는 함수

    Args:
        model (torch.nn.Module): 모델
        tokenizer (transformers.tokenization_utils_base.PreTrainedTokenizer): tokenizer
        scene_path (str): scene 경로
        prompt (dict): prompt 정보
        generation_config (dict): 생성 설정
        use_audio_for_prompt (bool): VideoLM으로 추론할 때, 오디오자막을 프롬프트에 넣어줄지 여부
        mono_audio_folder (str): 모노 오디오 폴더 경로
        scene_info_json_file_path (str): scene 정보 json 파일 경로 (해당 Json에는 오디오 스크립트 정보 포함되어 있음)

        만약, use_audio_for_prompt가 True이고 scene_info_json_file_path가 None이면, whisper 모델을 사용하여 오디오 스크립트 추출
        만약, use_audio_for_prompt가 True이고 scene_info_json_file_path가 있으면, 해당 경로에서 오디오 스크립트 추출

    Returns:
        result (dict): 캡션 결과
    """
    translator = Translator()

    scene_tensor = load_video(scene_path, num_segments=8, return_msg=False)
    scene_tensor = scene_tensor.to(model.device)

    # scene_name 추출 (audio_name이랑 같음 - {video_id}_{start:.3f}_{end:.3f}_{i + 1:03d})
    scene_name = os.path.basename(scene_path)[: -len(".mp4")]
    video_id, start, end, scene_id = scene_name.split("_")

    if use_audio_for_prompt:
        mono_audio_path = os.path.join(mono_audio_folder, scene_name + ".wav")
        if scene_info_json_file_path:
            with open(scene_info_json_file_path, "r") as f:
                audio_text = json.load(f)[video_id][int(scene_id) - 1]["audio_text"]
        else:
            # STT 모델인 Whisper를 불러옴
            import whisper

            whisper_model = whisper.load_model("large-v3")
            audio_text = transcribe_audio(mono_audio_path, whisper_model)

        prompt += f"\n[script]: {audio_text}"

    chat_history = []
    transformers_logger.setLevel(logging.ERROR)  # 경고 메시지 무시 (해당 구간만 적용)
    response, chat_history = model.chat(
        tokenizer,
        "",
        prompt,
        media_type="video",
        media_tensor=scene_tensor,
        chat_history=chat_history,
        return_history=True,
        generation_config=generation_config,
    )
    transformers_logger.setLevel(logging.WARNING)  # 복구

    translated_description = translator.translate(response, src="en", dest="ko").text

    result = {
        "video_id": video_id,
        "start_time": start,
        "end_time": end,
        "clip_id": f"{video_id}_{start}_{end}_{scene_id}",
        "scene_path": scene_path,
        "caption": response,
        "caption_ko": translated_description,
    }
    return result


def single_scene_caption_LlavaVideo(
    model,
    tokenizer,
    image_processor,
    max_frames_num,
    scene_path,
    prompt,
    use_audio_for_prompt,
    mono_audio_folder,
    scene_info_json_file_path=None,  # 오디오 스크립트 정보 포함
):
    """
    1개의 scene에 대한 캡션을 LlavaVideo 모델을 사용하여 생성하는 함수

    Args:
        model (torch.nn.Module): 모델
        tokenizer (transformers.tokenization_utils_base.PreTrainedTokenizer): tokenizer
        image_processor: 이미지 처리기 (lmms-lab/LLaVA-Video-7B-Qwen2에 쓰임)
        max_frames_num (int): 최대 프레임 수
        scene_path (str): scene 경로
        prompt (dict): prompt 정보
        use_audio_for_prompt (bool): VideoLM으로 추론할 때, 오디오자막을 프롬프트에 넣어줄지 여부
        mono_audio_folder (str): 모노 오디오 폴더 경로
        scene_info_json_file_path (str): scene 정보 json 파일 경로 (해당 Json에는 오디오 스크립트 정보 포함되어 있음)

        만약, use_audio_for_prompt가 True이고 scene_info_json_file_path가 None이면, whisper 모델을 사용하여 오디오 스크립트 추출
        만약, use_audio_for_prompt가 True이고 scene_info_json_file_path가 있으면, 해당 경로에서 오디오 스크립트 추출

    Returns:
        result (dict): 캡션 결과
    """
    translator = Translator()

    # scene_name 추출 (audio_name이랑 같음 - {video_id}_{start:.3f}_{end:.3f}_{i + 1:03d})
    scene_name = os.path.basename(scene_path)[: -len(".mp4")]
    video_id, start, end, scene_id = scene_name.split("_")

    if use_audio_for_prompt:
        mono_audio_path = os.path.join(mono_audio_folder, scene_name + ".wav")
        if scene_info_json_file_path:
            with open(scene_info_json_file_path, "r") as f:
                audio_text = json.load(f)[video_id][int(scene_id) - 1]["audio_text"]
        else:
            # STT 모델인 Whisper를 불러옴
            import whisper

            whisper_model = whisper.load_model("large-v3")
            audio_text = transcribe_audio(mono_audio_path, whisper_model)

        prompt += f"\n[script]: {audio_text}"

    video, input_ids = get_video_and_input_ids(
        scene_path, tokenizer, model, image_processor, max_frames_num, prompt
    )

    attention_mask = (input_ids != tokenizer.pad_token_id).long().to("cuda")

    cont = model.generate(
        input_ids,
        attention_mask=attention_mask,
        images=video,
        modalities=["video"],
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
    )
    response = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()

    translated_description = translator.translate(response, src="en", dest="ko").text

    result = {
        "video_id": video_id,
        "start_time": start,
        "end_time": end,
        "clip_id": f"{video_id}_{start}_{end}_{scene_id}",
        "scene_path": scene_path,
        "caption": response,
        "caption_ko": translated_description,
    }
    return result


def single_scene_caption_InternVideo2_5_Chat(
    generator,
    scene_path,
    prompt,
    use_audio_for_prompt,
    mono_audio_folder,
    scene_info_json_file_path=None,  # 오디오 스크립트 정보 포함
):
    """
    1개의 scene에 대한 캡션을 InternVideo2_5_Chat 모델을 사용하여 생성하는 함수

    Args:
        generator (DescriptionGenerator): 모델
        scene_path (str): scene 경로
        prompt (dict): prompt 정보
        use_audio_for_prompt (bool): VideoLM으로 추론할 때, 오디오자막을 프롬프트에 넣어줄지 여부
        mono_audio_folder (str): 모노 오디오 폴더 경로
        scene_info_json_file_path (str): scene 정보 json 파일 경로 (해당 Json에는 오디오 스크립트 정보 포함되어 있음)

        만약, use_audio_for_prompt가 True이고 scene_info_json_file_path가 None이면, whisper 모델을 사용하여 오디오 스크립트 추출
        만약, use_audio_for_prompt가 True이고 scene_info_json_file_path가 있으면, 해당 경로에서 오디오 스크립트 추출

    Returns:
        result (dict): 캡션 결과
    """
    translator = Translator()

    # scene_name 추출 (audio_name이랑 같음 - {video_id}_{start:.3f}_{end:.3f}_{i + 1:03d})
    scene_name = os.path.basename(scene_path)[: -len(".mp4")]
    video_id, start, end, scene_id = scene_name.split("_")

    if use_audio_for_prompt:
        mono_audio_path = os.path.join(mono_audio_folder, scene_name + ".wav")
        if scene_info_json_file_path:
            with open(scene_info_json_file_path, "r") as f:
                audio_text = json.load(f)[video_id][int(scene_id) - 1]["audio_text"]
        else:
            # STT 모델인 Whisper를 불러옴
            import whisper

            whisper_model = whisper.load_model("large-v3")
            audio_text = transcribe_audio(mono_audio_path, whisper_model)

        prompt += f"\n[script]: {audio_text}"

    # transformers_logger.setLevel(logging.ERROR)  # 경고 메시지 무시 (해당 구간만 적용)
    response = generator.describe_scene(scene_path, prompt, num_segments=32)
    # transformers_logger.setLevel(logging.WARNING)  # 복구

    translated_description = translator.translate(response, src="en", dest="ko").text

    result = {
        "video_id": video_id,
        "start_time": start,
        "end_time": end,
        "clip_id": f"{video_id}_{start}_{end}_{scene_id}",
        "scene_path": scene_path,
        "caption": response,
        "caption_ko": translated_description,
    }
    return result


def single_scene_caption(
    model_path,
    model,
    tokenizer,
    image_processor,
    scene_path,
    prompt,
    generation_config,
    use_audio_for_prompt,
    mono_audio_folder,
    scene_info_json_file_path=None,
):
    """
    1개의 scene에 대한 캡션을 생성하는 함수 - 모델 통일

    Args:
        model_path (str): 모델 경로
        model (torch.nn.Module): 모델
        tokenizer (transformers.tokenization_utils_base.PreTrainedTokenizer): tokenizer
        image_processor: 이미지 처리기 (lmms-lab/LLaVA-Video-7B-Qwen2에 쓰임)
        scene_path (str): scene 경로
        prompt (dict): prompt 정보
        generation_config (dict): 생성 설정
        use_audio_for_prompt (bool): VideoLM으로 추론할 때, 오디오자막을 프롬프트에 넣어줄지 여부
        mono_audio_folder (str): 모노 오디오 폴더 경로
        scene_info_json_file_path (str): scene 정보 json 파일 경로 (해당 Json에는 오디오 스크립트 정보 포함되어 있음)

        만약, use_audio_for_prompt가 True이고 scene_info_json_file_path가 None이면, whisper 모델을 사용하여 오디오 스크립트 추출
        만약, use_audio_for_prompt가 True이고 scene_info_json_file_path가 있으면, 해당 경로에서 오디오 스크립트 추출

    Returns:
        result (dict): 캡션 결과
    """
    if model_path == "OpenGVLab/InternVideo2-Chat-8B":
        return single_scene_caption_InternVideo2(
            model,
            tokenizer,
            scene_path,
            prompt,
            generation_config,
            use_audio_for_prompt,
            mono_audio_folder,
            scene_info_json_file_path,
        )
    elif model_path == "lmms-lab/LLaVA-Video-7B-Qwen2":
        max_frames_num = 64
        return single_scene_caption_LlavaVideo(
            model,
            tokenizer,
            image_processor,
            max_frames_num,
            scene_path,
            prompt,
            use_audio_for_prompt,
            mono_audio_folder,
            scene_info_json_file_path,
        )
    elif model_path == "OpenGVLab/InternVideo2_5_Chat_8B":
        return single_scene_caption_InternVideo2_5_Chat(
            model,
            scene_path,
            prompt,
            use_audio_for_prompt,
            mono_audio_folder,
            scene_info_json_file_path,
        )


def scene_caption(
    model_path,
    scene_folder,
    prompt,
    generation_config,
    use_audio_for_prompt,
    mono_audio_folder,
    scene_info_json_file_path,
    output_scene_caption_json_path,
):
    """
    scene_folder 내의 모든 scene에 대한 캡션을 생성하는 함수

    Args:
        model_path (str): 모델 경로
        scene_folder (str): scene 폴더 경로
        prompt (dict): prompt 정보
        generation_config (dict): 생성 설정
        use_audio_for_prompt (bool): VideoLM으로 추론할 때, 오디오자막을 프롬프트에 넣어줄지 여부
        mono_audio_folder (str): 모노 오디오 폴더 경로
        scene_info_json_file_path (str): scene 정보 json 파일 경로 (해당 Json에는 오디오 스크립트 정보 포함되어 있음)
        output_scene_caption_json_path (str): 캡션 결과를 저장할 json 파일 경로

        만약, use_audio_for_prompt가 True이고 scene_info_json_file_path가 None이면, whisper 모델을 사용하여 오디오 스크립트 추출
        만약, use_audio_for_prompt가 True이고 scene_info_json_file_path가 있으면, 해당 경로에서 오디오 스크립트 추출

    Returns:
        None
    """
    model, tokenizer, image_processor = initialize_model(model_path)

    final_json_data = []

    # scene_names를 video_id, start 순으로 정렬
    scene_names = os.listdir(scene_folder)
    scene_names.sort(key=lambda x: (x.split("_")[0], float(x.split("_")[1])))

    for scene_name in tqdm(scene_names):
        scene_path = os.path.join(scene_folder, scene_name)
        result = single_scene_caption(
            model_path,
            model,
            tokenizer,
            image_processor,
            scene_path,
            prompt,
            generation_config,
            use_audio_for_prompt,
            mono_audio_folder,
            scene_info_json_file_path,
        )
        final_json_data.append(result)

    with open(output_scene_caption_json_path, "w", encoding="utf-8") as json_file:
        json.dump(final_json_data, json_file, ensure_ascii=False, indent=4)

    print(f"All outputs have been saved to {output_scene_caption_json_path}.")


if __name__ == "__main__":
    model, tokenizer = initialize_model()

    print(
        single_scene_caption_InternVideo2(
            model,
            tokenizer,
            "../scenes/5qlG1ODkRWw_95.262_100.892_024.mp4",
            "describe the video",
            {
                "do_sample": False,
                "temperature": 0.5,
                "top_p": 0.9,
                "top_k": 50,
                "max_new_tokens": 256,
                "num_beams": 1,
            },
            True,
            "../mono_audio",
            "../scene_info.json",
        )
    )
