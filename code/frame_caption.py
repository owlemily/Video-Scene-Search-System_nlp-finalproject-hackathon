import json
import os

import torch
from datasets import load_from_disk
from googletrans import Translator
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from unsloth_vision_utils import Custom_UnslothVisionDataCollator
from utils import get_video_id_and_timestamp, load_config

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_model(model_name, device):
    """
    모델, 토크나이저를 반환하는 함수

    Args:
        model_name (str): 모델 이름
        device (str): 디바이스 이름
    """
    print("*" * 50)
    print(f"Loading model: {model_name}")
    print("*" * 50)

    # 일반 모델 로드
    if "unsloth" not in model_name:
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
        except Exception:
            raise ValueError(f"지원하지 않는 모델: {model_name}")

    # unsloth 모델 로드
    else:
        from unsloth import FastVisionModel

        model, tokenizer = FastVisionModel.from_pretrained(
            model_name, load_in_4bit=True, use_gradient_checkpointing="unsloth"
        )
        FastVisionModel.for_inference(model)

    return model, tokenizer


def translate_caption(caption, translator, target_lang="ko"):
    """
    caption을 입력으로 받아 번역된 caption_ko를 반환하는 함수

    Args:
        caption (str): 번역할 캡션
        translator (googletrans.Translator): 번역기 객체
        target_lang (str): 번역할 언어 코드 (default: "ko")
    """
    try:
        return translator.translate(caption, dest=target_lang).text
    except Exception as e:
        print(f"번역 실패: {caption}. 오류: {e}")
        return ""


def dynamic_preprocess(image, image_size=448, use_thumbnail=False):
    """
    이미지를 입력으로 받아 전처리된 이미지를 반환하는 함수 (unsloth 모델이 아닌 일반 모델 추론에 사용됩니다)

    Args:
        image (PIL.Image): 전처리할 이미지
        image_size (int): 이미지 크기 (default: 448)
        use_thumbnail (bool): 썸네일 사용 여부 (default: False)
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
        translator (googletrans.Translator): 번역기
        device (str): 디바이스
        max_new_tokens (int): 최대 토큰 길이
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
        translator (googletrans.Translator): 번역기
        device (str): 디바이스
        max_new_tokens (int): 최대 토큰 길이
    """
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
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
    model, tokenizer, image, prompt, translator, device, max_new_tokens
):
    """
    입력 받은 사진에 대한 캡션 생성 - 일반 모델과 UnSloth 모델 통일

    Args:
        model (torch.nn.Module): 모델
        tokenizer (transformers.PreTrainedTokenizer): 토크나이저
        image (PIL.Image): 이미지
        prompt (str): 프롬프트
        translator (googletrans.Translator): 번역기
        device (str): 디바이스
        max_new_tokens (int): 최대 토큰 길이
    """
    if "unsloth" in model.__class__.__name__.lower():
        return generate_caption_ForUnslothModel(
            model, tokenizer, image, prompt, translator, device, max_new_tokens
        )
    else:
        return generate_caption_ForGeneralModel(
            model, tokenizer, image, prompt, translator, device, max_new_tokens
        )


def generate_caption_UsingDataset(
    model, tokenizer, dataset, prompt, translator, batch_size, max_new_tokens
):
    """
    Huggingface Dataset을 사용해 전체 데이터셋에 대한 캡션 생성
    """
    collator = Custom_UnslothVisionDataCollator(model, tokenizer)
    results = []

    for start_idx in tqdm(
        range(0, len(dataset), batch_size), desc="Generating captions in batches"
    ):
        batch = dataset[start_idx : start_idx + batch_size]
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


def main(config_path):
    # 설정 로드
    config = load_config(config_path)
    device = config["general"]["device"]
    frames_folder = config["general"]["frames_folder"]
    output_folder = config["general"]["output_folder"]
    datasets_folder = config["general"]["datasets_folders"]
    datasets_name = config["general"]["datasets_name"]
    use_datasets = config["general"]["use_datasets"]
    caption_prompt = config["caption"]["prompt"]
    model_name = config["model"]["model_name"]
    max_new_tokens = config["generation"]["max_new_tokens"]
    batch_size = config["generation"]["batch_size"]

    model, tokenizer = load_model(model_name, device)
    translator = Translator()

    if use_datasets:
        dataset_path = os.path.join(datasets_folder, datasets_name)
        dataset = load_from_disk(dataset_path)
        results = generate_caption_UsingDataset(
            model,
            tokenizer,
            dataset,
            caption_prompt,
            translator,
            batch_size,
            max_new_tokens,
        )
    else:
        frame_files = sorted(
            [f for f in os.listdir(frames_folder) if f.endswith(".jpg")],
            key=lambda x: get_video_id_and_timestamp(x),
        )
        results = []

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
            )

            video_id, timestamp = get_video_id_and_timestamp(frame_file)
            results.append(
                {
                    "video_id": video_id,
                    "timestamp": timestamp,
                    "frame_image_path": frame_path,
                    "caption": caption,
                    "caption_ko": caption_ko,
                }
            )

    output_path = os.path.join(output_folder, "frame_output.json")
    os.makedirs(output_folder, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"모든 프레임에 대해 캡션 생성 완료. 결과가 {output_path}에 저장되었습니다.")


if __name__ == "__main__":
    main("fcs_config.yaml")
