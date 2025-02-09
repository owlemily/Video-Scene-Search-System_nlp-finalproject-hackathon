import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import clip  # OpenAI CLIP 패키지
from lavis.models import load_model_and_preprocess  # BLIP2 관련

from utils.captioning import single_scene_caption_LlavaVideo
from utils.external_videos_preprocess_utils import (
    create_json,
    extract_frames_from_folder,
    save_timestamps_to_txt,
)
from utils.LlavaVideo_utils import load_llava_video_model

os.environ["TOKENIZERS_PARALLELISM"] = "false"

#########################################
# ImageDataset: 이미지 로딩 및 전처리 클래스
#########################################
class ImageDataset(Dataset):
    def __init__(self, image_paths: list, preprocess, convert_mode: str = None):
        self.image_paths = image_paths
        self.preprocess = preprocess
        self.convert_mode = convert_mode

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        try:
            img = Image.open(path)
            if self.convert_mode:
                img = img.convert(self.convert_mode)
        except Exception as e:
            print(f"[ImageDataset] 이미지 로딩 오류 ({path}): {e}")
            img = Image.new("RGB", (224, 224))
        processed = self.preprocess(img)
        return processed, path


#########################################
# CLIP 임베딩 생성 함수
#########################################
def generate_clip_embeddings(
    frames_folder: str,
    embedding_folder: str,
    batch_size: int = 512,
    num_workers: int = 4,
    model_name: str = "ViT-L/14@336px"
):
    image_extensions = (".jpg", ".jpeg", ".png")
    image_filenames = [
        os.path.join(frames_folder, fname)
        for fname in sorted(os.listdir(frames_folder))
        if fname.lower().endswith(image_extensions)
    ]
    if not image_filenames:
        print(f"[CLIP] {frames_folder}에서 이미지 파일을 찾을 수 없습니다.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)
    model.eval()

    dataset = ImageDataset(image_filenames, preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    embeds_list, files_list = [], []
    with torch.no_grad():
        for images, paths in tqdm(dataloader, desc="Extracting CLIP embeddings"):
            images = images.to(device)
            features = model.encode_image(images)
            # L2 정규화
            features = features / features.norm(dim=-1, keepdim=True)
            embeds_list.append(features.cpu())
            files_list.extend(paths)
    clip_embeddings = torch.cat(embeds_list, dim=0)

    os.makedirs(embedding_folder, exist_ok=True)
    embedding_path = os.path.join(embedding_folder, "external_clip_embeddings.pt")
    # 저장 시 키를 "features"로 통일
    torch.save({"filenames": files_list, "features": clip_embeddings}, embedding_path)
    print(f"[CLIP] 임베딩을 {embedding_path}에 저장했습니다.")


#########################################
# BLIP 임베딩 생성 함수
#########################################
def generate_blip_embeddings(
    frames_folder: str,
    embedding_folder: str,
    batch_size: int = 512,
    num_workers: int = 4,
    model_type: str = "pretrain"
):
    image_extensions = (".jpg", ".jpeg", ".png")
    image_filenames = [
        os.path.join(frames_folder, fname)
        for fname in sorted(os.listdir(frames_folder))
        if fname.lower().endswith(image_extensions)
    ]
    if not image_filenames:
        print(f"[BLIP] {frames_folder}에서 이미지 파일을 찾을 수 없습니다.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="blip2_feature_extractor",
        model_type=model_type,
        is_eval=True,
        device=device,
    )
    model.eval()

    dataset = ImageDataset(image_filenames, vis_processors["eval"], convert_mode="RGB")
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    embeds_list, files_list = [], []
    with torch.no_grad():
        for images, paths in tqdm(dataloader, desc="Extracting BLIP embeddings"):
            images = images.to(device)
            features = model.extract_features({"image": images}, mode="image")
            # 이미지 임베딩은 평균값을 사용하고 L2 정규화 적용
            image_embed = features.image_embeds_proj.mean(dim=1)
            image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)
            embeds_list.append(image_embed.cpu())
            files_list.extend(paths)
    blip_embeddings = torch.cat(embeds_list, dim=0)

    os.makedirs(embedding_folder, exist_ok=True)
    embedding_path = os.path.join(embedding_folder, "external_blip_embeddings.pt")
    # 저장 시 키를 "features"로 통일
    torch.save({"filenames": files_list, "features": blip_embeddings}, embedding_path)
    print(f"[BLIP] 임베딩을 {embedding_path}에 저장했습니다.")


#########################################
# Scene 임베딩 생성 함수 (results_file 기반)
#########################################
def generate_scene_embeddings(
    results_file: str,
    embedding_folder: str,
    model_name: str = "BAAI/bge-large-en",
    batch_size: int = 64
):
    # results_file에서 scene 정보 로드
    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)
    scenes = results.get("scenes", [])
    if not scenes:
        print(f"[Scene] {results_file}에 scene 정보가 없습니다.")
        return

    # 모든 scene에 대해 caption을 가져오는데, 없으면 빈 문자열로 처리하여 길이를 맞춥니다.
    captions = [scene.get("caption", "") for scene in scenes]

    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    all_embeds = []
    for i in tqdm(range(0, len(captions), batch_size), desc="Encoding scene captions"):
        batch = captions[i : i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # 일반적으로 첫 토큰(CLS) 임베딩을 사용하고, L2 정규화를 적용합니다.
            embeds = outputs.last_hidden_state[:, 0, :]
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        all_embeds.append(embeds.cpu())
    scene_embeddings = torch.cat(all_embeds, dim=0)

    os.makedirs(embedding_folder, exist_ok=True)
    embedding_path = os.path.join(embedding_folder, "external_scene_embeddings.pt")
    # 저장 시 key를 "data_info"로 설정하여 BGERetrieval과 동일한 구조로 만듭니다.
    torch.save({"data_info": scenes, "features": scene_embeddings}, embedding_path)
    print(f"[Scene] 임베딩을 {embedding_path}에 저장했습니다.")



#########################################
# 전체 비디오 처리 파이프라인
#########################################
def process_external_videos(
    video_folder: str = "/data/ephemeral/home/external_videos",
    frames_output_folder: str = "./external_frames_fps10",
    embedding_folder: str = "/data/ephemeral/home/ttv/embeddings",
    prompt: str = (
        "Analyze the provided video, and describe in detail the movements and actions of objects and backgrounds.\n\n"
        "** Instructions **  \n"
        "1. Describe the movement and behaviour of all objects in the video. It clearly distinguishes who does what among the many characters. \n"
        "2. **Describe the movement of other objects** in scenes such as vehicles, animals, and inanimate objects. \n"
        "3. When describing actions, refer to the **camera's point of view** (e.g. ‘move to the left with the knife’). \n"
        "4. If the subject is expressing emotions (e.g., fear, excitement, aggression), describe the subject's facial expressions and body language. **If no emotions are detected, focus on the details of the movements.**\n"
        "5. If the subjects interact with each other (e.g., fighting, talking, helping), clearly describe the subjects' actions and the nature of the interaction. \n"
        "6. Do not speculate or guess about content not covered in the video (avoid hallucinations). \n\n"
        "**Example:**\n"
        "A woman with white hair and a purple dress is talking to a snowman in front of a bonfire. She is looking at the snowman melting because of the fire. The snowman is trying to pull up his face, which is dripping down, so that it doesn't collapse.\n"
    ),
    timestamps_file: str = "./external_timestamps.txt",
    results_file: str = "/data/ephemeral/home/ttv/description/external_video_results.json",
    frame_rate: int = 10,
    threshold: float = 30.0,
    min_scene_len: float = 2,
    max_new_tokens: int = 512,
    max_num_frames: int = 50,
):
    """
    전체 비디오 처리 파이프라인을 실행하는 함수입니다.
    
    단계:
      1. video_folder 내 모든 비디오에서 frames_output_folder로 프레임을 추출합니다.
      2. 비디오의 타임스탬프 정보를 timestamps_file로 저장합니다.
      3. LlavaVideo 모델과 토크나이저, 이미지 프로세서를 불러옵니다.
      4. 각 비디오에 대해 장면 캡션 및 번역을 생성합니다.
      5. create_json 함수를 통해 scene 정보를 불러오고, 캡션 정보를 업데이트 후 results_file로 저장합니다.
      6. frames_output_folder를 바탕으로 CLIP과 BLIP 임베딩을 생성하여 embedding_folder에 저장합니다.
      7. results_file를 바탕으로 scene 임베딩을 생성하여 embedding_folder에 저장합니다.
    """
    # 출력 폴더 생성
    os.makedirs(frames_output_folder, exist_ok=True)

    # 1. 프레임 추출
    extract_frames_from_folder(video_folder, frames_output_folder, frame_rate=frame_rate)
    print("프레임 추출 완료")

    # 2. 타임스탬프 저장
    save_timestamps_to_txt(video_folder, timestamps_file, threshold=threshold, min_scene_len=min_scene_len)
    print("타임스탬프 저장 완료")

    # 3. LlavaVideo 모델 로드
    tokenizer, model, image_processor, max_length = load_llava_video_model()

    caption_dict = {}

    # 4. 각 비디오에 대해 캡션 생성 및 번역 수행
    video_list = os.listdir(video_folder)
    for video_name in tqdm(video_list, desc="비디오 처리중"):
        video_path = os.path.join(video_folder, video_name)
        response, translated_description = single_scene_caption_LlavaVideo(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            scene_path=video_path,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            max_num_frames=max_num_frames,
        )
        video_id = os.path.splitext(video_name)[0]  # 확장자 제거하여 video_id 생성
        caption_dict[video_id] = {
            "caption": response,
            "caption_ko": translated_description,
        }

    # 5. scene 정보 불러오기 및 캡션 정보 업데이트 후 JSON 저장
    results = create_json(timestamps_file, results_file)
    for i, scene in enumerate(results.get("scenes", [])):
        video_id = scene.get("video_id")
        if video_id in caption_dict:
            results["scenes"][i]["caption"] = caption_dict[video_id]["caption"]
            results["scenes"][i]["caption_ko"] = caption_dict[video_id]["caption_ko"]
        else:
            print(f"경고: {video_id}에 대한 캡션 정보가 없습니다.")

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print("scene 정보 업데이트 및 JSON 저장 완료")

    # 6. frames_output_folder 기반 CLIP, BLIP 임베딩 생성 및 저장
    generate_clip_embeddings(frames_output_folder, embedding_folder)
    generate_blip_embeddings(frames_output_folder, embedding_folder)

    # 7. results_file 기반 scene 임베딩 생성 및 저장
    generate_scene_embeddings(results_file, embedding_folder)

    return results


if __name__ == "__main__":
    process_external_videos()
