"""
캡셔닝할 때 사용하는 함수들을 정의한 파일입니다.

함수 목록:

"""

import os
import whisper

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .vtt_service_utils import transcribe_audio, translate_caption
from .LlavaVideo_utils import (
    get_video_and_input_ids,
    load_llava_video_model,
)


def initialize_llava_video_model():
    """
    LLaVA-Video 모델을 초기화하여 tokenizer, model, image_processor를 반환하는 함수

    Returns:
        tokenizer, model, image_processor
    """
    tokenizer, model, image_processor, _ = load_llava_video_model()
    return tokenizer, model, image_processor

def initialize_whisper():
    whisper_model = whisper.load_model("large-v3")
    return whisper_model 


def single_scene_caption_LlavaVideo(
    model,
    tokenizer,
    image_processor,
    scene_path,
    prompt,
    max_new_tokens,
    max_num_frames,
    enable_audio_text,
    whisper_model,
    mono_audio_path,
    translator,
):
    """
    1개의 scene에 대한 캡션을 LlavaVideo 모델을 사용하여 생성하는 함수

    Args:
        model (torch.nn.Module): 모델
        tokenizer (transformers.tokenization_utils_base.PreTrainedTokenizer): tokenizer
        image_processor: 이미지 처리기
        scene_path (str): scene 경로
        prompt (dict): prompt 정보
        max_new_tokens (int): 최대 토큰 수
        max_num_frames (int): 최대 프레임 수
        enable_audio_text (bool): 오디오 텍스트 사용 여부
        whisper_model (whisper.Whisper): STT 모델
        mono_audio_folder (str): 모노 오디오 폴더 경로
        translator (googletrans.Translator or deepl.Translator): 번역기 객체

    Returns:
        response, translated_description (str, str): 생성된 영어 캡션, 번역된 캡션
    """
    # scene_name 추출 (audio_name이랑 같음)
    # audio_name = scene_name = os.path.basename(scene_path)[: -len(".mp4")]

    if enable_audio_text:
        # STT 모델인 Whisper을 사용하여 오디오 텍스트 추출
        audio_text = transcribe_audio(mono_audio_path, whisper_model)

        # 프롬프트에 오디오 텍스트를 넣어주어 오디오를 반영하여 캡션 생성
        prompt += f"\n[SCRIPT]: {audio_text}[EOS]"

        print(prompt)

    video, input_ids = get_video_and_input_ids(
        scene_path, tokenizer, model, image_processor, max_num_frames, prompt
    )

    attention_mask = (input_ids != tokenizer.pad_token_id).long().to("cuda")

    cont = model.generate(
        input_ids,
        attention_mask=attention_mask,
        images=video,
        modalities=["video"],
        do_sample=False,
        temperature=0,
        max_new_tokens=max_new_tokens,
    )
    response = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()

    translated_description = translate_caption(response, translator, target_lang="ko")

    return response, translated_description


def load_qwen2_5_VL_model():
    """
    Qwen2.5-VL 모델을 초기화하여 model, processor를 반환하는 함수

    Returns:
        model, processor
    """
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    return model, processor


def single_frame_caption_Qwen2_5_VL(
    model,
    processor,
    frame_path,
    prompt,
    max_new_tokens,
    translator,
):
    """
    1개의 scene에 대한 캡션을 Qwen2.5-VL 모델을 사용하여 생성하는 함수

    Args:
        model (torch.nn.Module): 모델
        processor (transformers.tokenization_utils_base.PreTrainedTokenizer): processor
        frame_path (str): scene 경로
        prompt (dict): prompt 정보
        max_new_tokens (int): 최대 토큰 수
        translator (googletrans.Translator or deepl.Translator): 번역기 객체

    Returns:
        response, translated_description (str, str): 생성된 영어 캡션, 번역된 캡션
    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{frame_path}",
                },
                {"type": "text", "text": f"{prompt}"},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    translated_description = translate_caption(response, translator, target_lang="ko")

    return response, translated_description
