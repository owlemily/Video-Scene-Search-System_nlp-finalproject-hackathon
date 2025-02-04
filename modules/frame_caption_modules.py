"""
frame_caption_modules.py

함수 목록:
1. load_model
2. dynamic_preprocess
3. generate_caption_ForGeneralModel
4. generate_caption_ForUnslothModel
5. generate_caption
6. generate_caption_UsingDataset_ForGeneralModel
7. generate_caption_UsingDataset_ForUnslothModel
8. generate_caption_UsingDatasets
9. frame_caption - 메인 함수. config 설정에 따라 프레임별 캡션 생성
"""

import json
import os

import torch
from datasets import load_from_disk
from googletrans import Translator
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import deepl

from .utils import translate_caption
from .frame_utils import get_video_id_and_timestamp
from .specific_model_utils.unsloth_vision_utils import Custom_UnslothVisionDataCollator


# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_model(model_name, device):
    """
    모델, 토크나이저를 반환하는 함수

    Args:
        model_name (str): 모델 이름
        device (str): 디바이스 이름

    Returns:
        torch.nn.Module, transformers.PreTrainedTokenizer: 모델, 토크나이저
    """
    print("*" * 50)
    print(f"Loading model: {model_name}")
    print("*" * 50)

    # 일반 모델 로드
    if "unsloth" not in model_name.lower():
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            model = (
                AutoModel.from_pretrained(model_name, trust_remote_code=True)
                .to(device)
                .eval()
            )
            model.img_context_token_id = tokenizer.cls_token_id
        except Exception as e:
            raise ValueError(f"지원하지 않는 모델: {model_name}\n오류: {e}")

    # unsloth 모델 로드
    else:
        from unsloth import FastVisionModel

        model, tokenizer = FastVisionModel.from_pretrained(
            model_name, load_in_4bit=True, use_gradient_checkpointing="unsloth"
        )
        FastVisionModel.for_inference(model)

    return model, tokenizer


def dynamic_preprocess(image, image_size=448, use_thumbnail=False):
    """
    이미지를 입력으로 받아 전처리된 이미지를 반환하는 함수 (unsloth 모델이 아닌 일반 모델 추론에 사용됩니다)

    Args:
        image (PIL.Image): 전처리할 이미지
        image_size (int): 이미지 크기 (default: 448)
        use_thumbnail (bool): 썸네일 사용 여부 (default: False)

    Returns:
        torch.Tensor: 전처리된 이미지
    """
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    transform = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    images = [transform(image)]
    if use_thumbnail:
        thumbnail = image.resize((image_size, image_size))
        images.append(transform(thumbnail))
    return torch.stack(images)


def generate_caption_ForGeneralModel(
    model, tokenizer, image, prompt, translator, device, max_new_tokens
):
    """
    일반 모델을 사용해 입력 받은 사진에 대한 캡션 생성

    Args:
        model (torch.nn.Module): 모델
        tokenizer (transformers.PreTrainedTokenizer): 토크나이저
        image (PIL.Image): 이미지
        prompt (str): 프롬프트
        translator (googletrans.Translator or deepl.Translator): 번역기 객체
        device (str): 디바이스
        max_new_tokens (int): 최대 토큰 길이

    Returns:
        str, str: 영어 캡션, 한국어 캡션
    """
    pixel_values = dynamic_preprocess(image, image_size=448).to(device)
    generation_config = {
        "max_length": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }
    caption = model.chat(
        tokenizer, pixel_values, prompt, generation_config=generation_config
    )
    caption_ko = translate_caption(caption, translator)
    return caption, caption_ko


def generate_caption_ForUnslothModel(
    model, tokenizer, image, prompt, translator, device, max_new_tokens
):
    """
    UnSloth 모델을 사용해 입력 받은 사진에 대한 캡션 생성

    Args:
        model (torch.nn.Module): 모델
        tokenizer (transformers.PreTrainedTokenizer): 토크나이저
        image (PIL.Image): 이미지
        prompt (str): 프롬프트
        translator (googletrans.Translator or deepl.Translator): 번역기 객체
        device (str): 디바이스
        max_new_tokens (int): 최대 토큰 길이

    Returns:
        str, str: 영어 캡션, 한국어 캡션
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt.replace("<image>\n", "")},
            ],
        }
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    inputs = tokenizer(
        image, input_text, add_special_tokens=False, return_tensors="pt"
    ).to(device)

    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)
    caption = tokenizer.decode(outputs[0], skip_special_tokens=True).split(
        "assistant\n"
    )[1]
    caption_ko = translate_caption(caption, translator)
    return caption, caption_ko


def generate_caption(
    model, tokenizer, image, prompt, translator, device, max_new_tokens, model_name
):
    """
    입력 받은 사진에 대한 캡션 생성 - 일반 모델과 UnSloth 모델 통일

    Args:
        model (torch.nn.Module): 모델
        tokenizer (transformers.PreTrainedTokenizer): 토크나이저
        image (PIL.Image): 이미지
        prompt (str): 프롬프트
        translator (googletrans.Translator or deepl.Translator): 번역기 객체
        device (str): 디바이스
        max_new_tokens (int): 최대 토큰 길이
        model_name (str): 모델 이름

    Returns:
        str, str: 영어 캡션, 한국어 캡션
    """
    if "unsloth" in model_name.lower():
        return generate_caption_ForUnslothModel(
            model, tokenizer, image, prompt, translator, device, max_new_tokens
        )
    else:
        return generate_caption_ForGeneralModel(
            model, tokenizer, image, prompt, translator, device, max_new_tokens
        )


def generate_caption_UsingDataset_ForGeneralModel(
    model, tokenizer, dataset, prompt, translator, batch_size, max_new_tokens, device
):
    """
    Huggingface Dataset을 사용해 General 모델로 전체 데이터셋에 대한 캡션 생성

    Args:
        model (torch.nn.Module): 모델
        tokenizer (transformers.PreTrainedTokenizer): 토크나이저
        dataset (Dataset): Huggingface Dataset
        prompt (str): 프롬프트
        translator (googletrans.Translator or deepl.Translator): 번역기 객체
        batch_size (int): 배치 크기
        max_new_tokens (int): 최대 토큰 길이
        device (str): 디바이스

    Returns:
        List[dict]: 캡션 결과 리스트
    """
    results = []

    for start_idx in tqdm(
        range(0, len(dataset), batch_size), desc="Generating captions in batches"
    ):
        batch = dataset.select(
            range(start_idx, min(start_idx + batch_size, len(dataset)))
        )
        images = [
            dynamic_preprocess(sample["image"], image_size=448).to(device)
            for sample in batch
        ]
        pixel_values = torch.stack(images).to(device)

        generation_config = {
            "max_length": max_new_tokens,
            "pad_token_id": tokenizer.eos_token_id,
        }

        captions = []
        for pixel_value in pixel_values:
            caption = model.chat(
                tokenizer,
                pixel_value,
                prompt,
                generation_config=generation_config,
            )
            captions.append(caption)

        captions_ko = [translate_caption(caption, translator) for caption in captions]

        for i, sample in enumerate(batch):
            results.append(
                {
                    "video_id": sample["video_id"],
                    "timestamp": sample["timestamp"],
                    "frame_image_path": sample["frame_image_path"],
                    "caption": captions[i],
                    "caption_ko": captions_ko[i],
                }
            )

    return results


def generate_caption_UsingDataset_ForUnslothModel(
    model, tokenizer, dataset, prompt, translator, batch_size, max_new_tokens
):
    """
    Huggingface Dataset을 사용해 Unsloth 모델로 전체 데이터셋에 대한 캡션 생성

    Args:
        model (torch.nn.Module): 모델
        tokenizer (transformers.PreTrainedTokenizer): 토크나이저
        dataset (Dataset): Huggingface Dataset
        prompt (str): 프롬프트
        translator (googletrans.Translator or deepl.Translator): 번역기 객체
        batch_size (int): 배치 크기
        max_new_tokens (int): 최대 토큰 길이

    Returns:
        List[dict]: 캡션 결과 리스트
    """
    instruction = prompt.replace("<image>\n", "")

    def convert_to_conversation(sample):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": sample["image"]},
                ],
            }
        ]
        return {
            "video_id": sample["video_id"],
            "timestamp": sample["timestamp"],
            "frame_image_path": sample["frame_image_path"],
            "messages": conversation,
        }

    converted_dataset = [convert_to_conversation(sample) for sample in dataset]

    collator = Custom_UnslothVisionDataCollator(model, tokenizer)
    results = []

    for start_idx in tqdm(
        range(0, len(converted_dataset), batch_size),
        desc="Generating captions in batches",
    ):
        batch = converted_dataset[start_idx : start_idx + batch_size]
        inputs = collator(batch)
        inputs = {key: val.to("cuda") for key, val in inputs.items()}
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, use_cache=True
        )

        captions = [
            tokenizer.decode(output, skip_special_tokens=True).split("assistant\n")[1]
            for output in outputs
        ]

        captions_ko = [translate_caption(caption, translator) for caption in captions]

        for i, sample in enumerate(batch):
            results.append(
                {
                    "video_id": sample["video_id"],
                    "timestamp": sample["timestamp"],
                    "frame_image_path": sample["frame_image_path"],
                    "caption": captions[i],
                    "caption_ko": captions_ko[i],
                }
            )

    return results


def generate_caption_UsingDataset(
    model,
    tokenizer,
    dataset,
    prompt,
    translator,
    batch_size,
    max_new_tokens,
    device,
    model_name,
):
    """
    Huggingface Dataset을 사용해 전체 데이터셋에 대한 캡션 생성 - 일반 모델과 UnSloth 모델 통일

    Args:
        model (torch.nn.Module): 모델
        tokenizer (transformers.PreTrainedTokenizer): 토크나이저
        dataset (Dataset): Huggingface Dataset
        prompt (str): 프롬프트
        translator (googletrans.Translator or deepl.Translator): 번역기 객체
        batch_size (int): 배치 크기
        max_new_tokens (int): 최대 토큰
        device (str): 디바이스
        model_name (str): 모델 이름

    Returns:
        List[dict]: 캡션 결과 리스트
    """
    if "unsloth" in model_name.lower():
        return generate_caption_UsingDataset_ForUnslothModel(
            model, tokenizer, dataset, prompt, translator, batch_size, max_new_tokens
        )
    else:
        return generate_caption_UsingDataset_ForGeneralModel(
            model,
            tokenizer,
            dataset,
            prompt,
            translator,
            batch_size,
            max_new_tokens,
            device,
        )


def frame_caption(
    device,
    frames_folder,
    output_folder,
    datasets_folder,
    datasets_name,
    model_name,
    caption_prompt,
    max_new_tokens,
    batch_size,
    use_datasets,
    frame_output_filename,
    translator_name
):
    """
    프레임 폴더 또는 데이터셋으로부터 캡션을 생성하고 Json 파일로 저장하는 함수

    Args:
        device (str): 디바이스 이름
        frames_folder (str): 프레임 이미지 폴더 경로
        output_folder (str): 결과 저장 폴더 경로
        datasets_folder (str): 데이터셋 폴더 경로
        datasets_name (str): 데이터셋 이름
        model_name (str): 모델 이름
        caption_prompt (str): 캡션 생성 프롬프트
        max_new_tokens (int): 최대 토큰 길이
        batch_size (int): 배치 크기
        use_datasets (bool): 데이터셋 사용 여부
        frame_output_filename (str): 결과 파일 이름
        translator_name (str): 번역기 이름 ("googletrans" 또는 "deepl")

    Returns:
        None
    """
    model, tokenizer = load_model(model_name, device)
    
    if translator_name == "googletrans":
        translator = Translator()
    elif translator_name == "deepl":
        auth_key = os.environ.get("DEEPL_API_KEY")
        translator = deepl.Translator(auth_key)
    else:
        raise ValueError("지원하지 않는 번역기입니다.")

    if use_datasets:
        dataset_path = os.path.join(datasets_folder, datasets_name)
        dataset = load_from_disk(dataset_path)

        frames_caption_list = generate_caption_UsingDataset(
            model,
            tokenizer,
            dataset,
            caption_prompt,
            translator,
            batch_size,
            max_new_tokens,
            device,
            model_name,
        )
        final_json_output = {
            "model_path": model_name,
            "prompt": caption_prompt,
            "max_new_tokens": max_new_tokens,
            "frames": frames_caption_list,
        }
    else:
        frame_files = sorted(
            [f for f in os.listdir(frames_folder) if f.endswith(".jpg")],
            key=lambda x: get_video_id_and_timestamp(x),
        )
        final_json_output = {
            "model_path": model_name,
            "prompt": caption_prompt,
            "max_new_tokens": max_new_tokens,
            "frames": [],
        }

        for frame_file in tqdm(frame_files, desc="Processing frames"):
            frame_path = os.path.join(frames_folder, frame_file)
            image = Image.open(frame_path).convert("RGB")
            caption, caption_ko = generate_caption(
                model,
                tokenizer,
                image,
                caption_prompt,
                translator,
                device,
                max_new_tokens,
                model_name,
            )

            video_id, timestamp = get_video_id_and_timestamp(frame_file)
            final_json_output["frames"].append(
                {
                    "video_id": video_id,
                    "timestamp": timestamp,
                    "frame_image_path": frame_path,
                    "caption": caption,
                    "caption_ko": caption_ko,
                }
            )

    output_path = os.path.join(output_folder, frame_output_filename)
    os.makedirs(output_folder, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_json_output, f, ensure_ascii=False, indent=4)

    print(f"모든 프레임에 대해 캡션 생성 완료. 결과가 {output_path}에 저장되었습니다.")


if __name__ == "__main__":
    # frame_caption 함수 테스트
    frame_caption(
        device="cuda",
        frames_folder="../frames",
        output_folder="../output",
        datasets_folder="../datasets",
        datasets_name="my_test_dataset",
        model_name="unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
        caption_prompt="<image>\nProvide a detailed description of the actions taking place in this image.",
        max_new_tokens=128,
        batch_size=6,
        use_datasets=True,
        frame_output_filename="frame_captions_test.json",
        translator_name="deepl"
    )

