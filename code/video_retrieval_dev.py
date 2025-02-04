import bisect
import json
import os

import clip
import numpy as np
import torch
import yaml
from lavis.models import load_model_and_preprocess
from PIL import Image
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


##############################################
# Helper Function: 이미지 파일 로딩 및 전처리
##############################################
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
            print(f"이미지 로딩 오류 ({path}): {e}")
            img = Image.new("RGB", (224, 224))
        processed = self.preprocess(img)
        return processed, path


##############################################
# 1. Scene Text Retrieval (BGE 기반)
##############################################
class BGERetrieval:
    """
    JSON 파일에서 scene 정보를 로드하고, BGE 모델로 캡션 텍스트를 임베딩하여
    사용자 쿼리와의 유사도를 계산합니다.
    retrieval 결과는 각 scene의 업데이트된 score를 반환하며,
    내부적으로 scene index를 구축해 빠른 검색을 지원합니다.
    """

    def __init__(self, config_path: str = "config/merged_config.yaml"):
        full_config = self._load_config(config_path)
        self.config = full_config["bge-scene"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # BGE 모델 및 토크나이저 로드
        self.model_name = self.config["model_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        # JSON 파일 읽기 및 scene index 구축 (클래스 내부 메서드 사용)
        self.json_file = self.config["json_file"]
        self.scene_index = self._build_scene_index_from_json(self.json_file)
        # 원본 scene 정보 전체 (retrieval 결과 구성 시 사용)
        with open(self.json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.data_info = data.get("scenes", [])
        self.texts = [scene["caption"] for scene in self.data_info]

        # 임베딩 파일 경로 및 존재시 로드, 없으면 생성
        self.embedding_file = self.config["output_file"]
        os.makedirs(os.path.dirname(self.embedding_file), exist_ok=True)
        if os.path.exists(self.embedding_file):
            self._load_embeddings(self.embedding_file)
        else:
            self.embeddings = self._encode_texts(self.texts)
            self._save_embeddings(self.embedding_file)

    def _load_config(self, config_path: str) -> dict:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _build_scene_index_from_json(self, json_file: str) -> dict:
        """
        JSON 파일 내의 scene 정보를 읽어, 각 video_id별로
        start_time에 따라 정렬된 scene index를 구축합니다.
        반환 예:
          {
             "video1": {
                 "starts": [0.0, 10.0, ...],
                 "scenes": [scene_dict1, scene_dict2, ...]
             },
             ...
          }
        """
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        scenes = data.get("scenes", [])
        scene_index = {}
        for scene in scenes:
            video_id = scene.get("video_id")
            if not video_id:
                continue
            try:
                start_time = float(scene["start_time"])
                end_time = float(scene["end_time"])
            except Exception as e:
                print(f"scene 파싱 오류: {e}")
                continue

            # 초기 score는 0.0으로 설정 (추후 BGE retrieval 결과로 업데이트)
            scene_entry = {
                "scene_id": scene.get("scene_id"),
                "start_time": start_time,
                "end_time": end_time,
                "caption": scene.get("caption"),
                "score": 0.0,
            }
            if video_id not in scene_index:
                scene_index[video_id] = {"starts": [], "scenes": []}
            scene_index[video_id]["starts"].append(start_time)
            scene_index[video_id]["scenes"].append(scene_entry)
        for vid in scene_index:
            combined = list(zip(scene_index[vid]["starts"], scene_index[vid]["scenes"]))
            combined.sort(key=lambda x: x[0])
            scene_index[vid]["starts"] = [item[0] for item in combined]
            scene_index[vid]["scenes"] = [item[1] for item in combined]
        return scene_index

    def _encode_texts(self, texts: list, batch_size: int = 1024) -> torch.Tensor:
        all_embeds = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding scene texts"):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeds = outputs.last_hidden_state[:, 0, :]
                embeds = embeds / embeds.norm(dim=-1, keepdim=True)
                all_embeds.append(embeds.cpu())
        return torch.cat(all_embeds, dim=0).to(self.device)

    def _save_embeddings(self, file_path: str) -> None:
        torch.save(
            {"data_info": self.data_info, "features": self.embeddings}, file_path
        )
        print(f"[BGE] 임베딩을 {file_path}에 저장했습니다.")

    def _load_embeddings(self, file_path: str) -> None:
        data = torch.load(file_path)
        self.data_info = data["data_info"]
        self.embeddings = data["features"]
        print(f"[BGE] 임베딩을 {file_path}에서 불러왔습니다.")

    def _compute_similarity(
        self, query_vec: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        query_vec = query_vec.to(features.device).float()
        features = features.float()
        return (features @ query_vec.T).squeeze(1)

    def retrieve(self, user_query: str, top_k: int = 5) -> list:
        """
        사용자 쿼리에 대해 scene retrieval을 수행합니다.
        반환 결과는 각 scene의 업데이트된 score와 정보를 담은 리스트입니다.
        """
        query_vec = self._encode_texts([user_query])
        scores = self._compute_similarity(query_vec, self.embeddings)
        scores_np = scores.cpu().numpy()
        score_mapping = {
            self.data_info[i]["scene_id"]: float(score)
            for i, score in enumerate(scores_np)
        }
        top_idxs = scores_np.argsort()[-top_k:][::-1]
        results = []
        for rank, idx in enumerate(top_idxs, start=1):
            info = self.data_info[idx]
            scene_id = info["scene_id"]
            results.append(
                {
                    "rank": rank,
                    "video_id": info.get("video_id"),
                    "scene_start_time": info["start_time"],
                    "scene_end_time": info["end_time"],
                    "scene_description": info["caption"],
                    "scene_id": scene_id,
                    "score": score_mapping[scene_id],
                }
            )
        return results


##############################################
# BaseRetrieval: 이미지 임베딩 관련 공통 기능
##############################################
class BaseRetrieval:
    def __init__(self, config_path: str, config_section: str) -> None:
        full_config = self._load_config(config_path)
        if config_section not in full_config:
            raise KeyError(
                f"'{config_section}' 설정이 {config_path}에 존재하지 않습니다."
            )
        self.config = full_config[config_section]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_folder = self.config["image_folder"]
        self.embedding_file = self.config["embedding_file"]
        self.image_extensions = tuple(self.config["image_extensions"])
        self.image_filenames = []
        self.image_embeddings = None
        self._ensure_output_dir()

    def _load_config(self, config_path: str) -> dict:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _ensure_output_dir(self) -> None:
        out_dir = os.path.dirname(self.embedding_file)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)

    def collect_image_filenames(self) -> None:
        self.image_filenames = [
            os.path.join(self.image_folder, fname)
            for fname in sorted(os.listdir(self.image_folder))
            if fname.endswith(self.image_extensions)
        ]


##############################################
# CLIPRetrieval: CLIP을 사용한 이미지 임베딩 및 검색
##############################################
class CLIPRetrieval(BaseRetrieval):
    def __init__(self, config_path: str = "config/image_retrieval_config.yaml") -> None:
        super().__init__(config_path, config_section="clip")
        self.model, self.preprocess = clip.load(
            self.config["model_name"], device=self.device
        )
        self.model.eval()
        if os.path.exists(self.embedding_file):
            self._load_image_embeddings()
        else:
            self._extract_and_save_image_embeddings()

    def _extract_and_save_image_embeddings(
        self, batch_size: int = 1024, num_workers: int = 4
    ) -> None:
        self.collect_image_filenames()
        dataset = ImageDataset(self.image_filenames, self.preprocess)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
        )
        embeds_list, files_list = [], []
        with torch.no_grad():
            for images, paths in tqdm(dataloader, desc="Extracting CLIP embeddings"):
                images = images.to(self.device)
                features = self.model.encode_image(images)
                features = features / features.norm(dim=-1, keepdim=True)
                embeds_list.append(features.cpu())
                files_list.extend(paths)
        self.image_embeddings = torch.cat(embeds_list, dim=0)
        torch.save(
            {"filenames": files_list, "features": self.image_embeddings},
            self.embedding_file,
        )
        print(f"CLIP 임베딩을 {self.embedding_file}에 저장했습니다.")

    def _load_image_embeddings(self) -> None:
        data = torch.load(self.embedding_file)
        self.image_filenames = data["filenames"]
        self.image_embeddings = data["features"]
        print(f"CLIP 임베딩을 {self.embedding_file}에서 불러왔습니다.")

    def retrieve(self, query: str, top_k: int = 10) -> list:
        text_tokens = clip.tokenize([query]).to(self.device)
        with torch.no_grad():
            text_embedding = self.model.encode_text(text_tokens)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        similarity = (self.image_embeddings.to(self.device) @ text_embedding.T).squeeze(
            1
        )
        top_idxs = similarity.topk(top_k).indices.cpu().numpy()
        return [
            {
                "rank": rank,
                "image_filename": self.image_filenames[idx],
                "score": float(similarity[idx].item()),
            }
            for rank, idx in enumerate(top_idxs, start=1)
        ]


##############################################
# BLIPRetrieval: BLIP2를 사용한 이미지 임베딩 및 검색
##############################################
class BLIPRetrieval(BaseRetrieval):
    def __init__(self, config_path: str = "config/image_retrieval_config.yaml") -> None:
        super().__init__(config_path, config_section="blip")
        self.model, self.vis_processors, self.txt_processors = (
            load_model_and_preprocess(
                name="blip2_feature_extractor",
                model_type=self.config.get("model_type", "pretrain"),
                is_eval=True,
                device=self.device,
            )
        )
        self.model.eval()
        if os.path.exists(self.embedding_file):
            self._load_image_embeddings()
        else:
            self._extract_and_save_image_embeddings()

    def _extract_and_save_image_embeddings(
        self, batch_size: int = 1024, num_workers: int = 4
    ) -> None:
        self.collect_image_filenames()
        dataset = ImageDataset(
            self.image_filenames, self.vis_processors["eval"], convert_mode="RGB"
        )
        dataloader = DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
        )
        embeds_list, files_list = [], []
        with torch.no_grad():
            for images, paths in tqdm(dataloader, desc="Extracting BLIP embeddings"):
                images = images.to(self.device)
                features = self.model.extract_features({"image": images}, mode="image")
                image_embed = features.image_embeds_proj.mean(dim=1)
                image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)
                embeds_list.append(image_embed.cpu())
                files_list.extend(paths)
        self.image_embeddings = torch.cat(embeds_list, dim=0)
        torch.save(
            {"filenames": files_list, "features": self.image_embeddings},
            self.embedding_file,
        )
        print(f"BLIP 임베딩을 {self.embedding_file}에 저장했습니다.")

    def _load_image_embeddings(self) -> None:
        data = torch.load(self.embedding_file)
        self.image_filenames = data["filenames"]
        self.image_embeddings = data["features"]
        print(f"BLIP 임베딩을 {self.embedding_file}에서 불러왔습니다.")

    def retrieve(self, query: str, top_k: int = 10) -> list:
        text_input = self.txt_processors["eval"](query)
        sample = {"text_input": [text_input]}
        with torch.no_grad():
            features_text = self.model.extract_features(sample, mode="text")
            text_embed = features_text.text_embeds_proj
            if text_embed.dim() == 3 and text_embed.size(1) > 1:
                text_embed = text_embed.mean(dim=1)
            elif text_embed.dim() != 2:
                raise ValueError(
                    f"예상하지 못한 텍스트 임베딩 차원: {text_embed.shape}"
                )
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
        similarity = (self.image_embeddings.to(self.device) @ text_embed.T).squeeze(1)
        top_idxs = similarity.topk(top_k).indices.cpu().numpy()
        return [
            {
                "rank": rank,
                "image_filename": self.image_filenames[idx],
                "score": float(similarity[idx].item()),
            }
            for rank, idx in enumerate(top_idxs, start=1)
        ]


##############################################
# Rankfusion: CLIP, BLIP, Scene retrieval 점수를 융합하여 최종 결과 생성
##############################################
class Rankfusion:
    """
    Rankfusion은 CLIP과 BLIP의 retrieval 결과에 더해,
    이미지 파일명(예: "video1_12.5.jpg")에서 video_id와 timestamp를 추출하여,
    BGE retrieval 결과에서 업데이트된 scene score(즉, scene_id → score 매핑)를 반영합니다.
    최종 fusion score는 각 retrieval 점수에 가중치를 곱해 산출합니다.
    """

    def __init__(
        self,
        scene_retriever: BGERetrieval,
        clip_retriever: CLIPRetrieval,
        blip_retriever: BLIPRetrieval,
        weight_clip: float = 0.33,
        weight_blip: float = 0.33,
        weight_scene: float = 0.34,
    ):
        self.scene_retriever = scene_retriever
        self.clip_retriever = clip_retriever
        self.blip_retriever = blip_retriever
        self.weight_clip = weight_clip
        self.weight_blip = weight_blip
        self.weight_scene = weight_scene
        self._clip_embedding_dict = None

    def _get_clip_embedding_dict(self) -> dict:
        if self._clip_embedding_dict is None:
            self._clip_embedding_dict = {
                fname: emb.cpu().numpy()
                for fname, emb in zip(
                    self.clip_retriever.image_filenames,
                    self.clip_retriever.image_embeddings,
                )
            }
        return self._clip_embedding_dict

    def _find_scene_for_image(
        self, video_id: str, timestamp: float, scene_index: dict
    ) -> dict:
        """
        주어진 video_id와 timestamp에 대해, scene_index를 이용하여
        해당 이미지가 속한 scene을 이진 탐색으로 찾습니다.
        """
        if video_id not in scene_index:
            return None
        starts = scene_index[video_id]["starts"]
        scenes = scene_index[video_id]["scenes"]
        idx = bisect.bisect_right(starts, timestamp) - 1
        if idx >= 0:
            scene = scenes[idx]
            if timestamp <= scene["end_time"]:
                return scene
        return None

    def _get_scene_scores(
        self, image_filenames: set, scene_score_mapping: dict
    ) -> dict:
        scores = {}
        for img in image_filenames:
            try:
                # (1) 디렉토리 경로 제거, 확장자 제거
                basename = os.path.basename(img)  # 예: "dflkc_ea31j_74.3659.jpg"
                file_stem, _ = os.path.splitext(
                    basename
                )  # ("dflkc_ea31j_74.3659", ".jpg")

                # (2) 마지막 '_' 기준으로 분리
                video_id, ts_str = file_stem.rsplit("_", 1)
                #  예:
                #    file_stem = "dflkc_ea31j_74.3659"
                #    => video_id="dflkc_ea31j", ts_str="74.3659"

                timestamp = float(ts_str)

            except Exception as e:
                print(f"파일명 파싱 오류 ({img}): {e}")
                scores[img] = 0.0
                continue

            # (4) scene_index에서 scene 검색
            scene = self._find_scene_for_image(
                video_id, timestamp, self.scene_retriever.scene_index
            )
            if scene:
                s_id = scene["scene_id"]
                scores[img] = scene_score_mapping.get(s_id, 0.0)
            else:
                scores[img] = 0.0

        return scores

    def retrieve(self, query: str, top_k: int = 10, union_top_n: int = None) -> list:
        """
        1) CLIP과 BLIP retrieval 결과(각 이미지 점수)를 구합니다.
        2) BGE retrieval을 통해 전체 scene에 대해 retrieval을 수행하여 scene score 매핑을 생성합니다.
        3) 이미지 파일명에서 video_id와 timestamp를 추출해 해당 scene의 업데이트된 score를 lookup합니다.
        4) 각 이미지에 대해 가중합하여 fusion score를 계산합니다.
        """
        if union_top_n is None:
            union_top_n = max(
                len(self.clip_retriever.image_filenames),
                len(self.blip_retriever.image_filenames),
            )

        clip_results = self.clip_retriever.retrieve(query, top_k=union_top_n)
        blip_results = self.blip_retriever.retrieve(query, top_k=union_top_n)
        scene_results = self.scene_retriever.retrieve(
            query, top_k=len(self.scene_retriever.data_info)
        )
        scene_score_mapping = {res["scene_id"]: res["score"] for res in scene_results}

        clip_dict = {
            res["image_filename"]: res.get("score", 0.0) for res in clip_results
        }
        blip_dict = {
            res["image_filename"]: res.get("score", 0.0) for res in blip_results
        }
        all_imgs = set(clip_dict.keys()).union(set(blip_dict.keys()))
        scene_dict = self._get_scene_scores(all_imgs, scene_score_mapping)
        fused = []
        for img in all_imgs:
            s_clip = clip_dict.get(img, 0.0)
            s_blip = blip_dict.get(img, 0.0)
            s_scene = scene_dict.get(img, 0.0)
            fusion_score = (
                self.weight_clip * s_clip
                + self.weight_blip * s_blip
                + self.weight_scene * s_scene
            )
            fused.append(
                {
                    "image_filename": img,
                    "clip_score": s_clip,
                    "blip_score": s_blip,
                    "scene_score": s_scene,
                    "fusion_score": fusion_score,
                }
            )
        fused.sort(key=lambda x: x["fusion_score"], reverse=True)
        for i, item in enumerate(fused, start=1):
            item["rank"] = i
        return fused[:top_k]

    def select_diverse_results_by_clustering(
        self, retrieval_results: list, desired_num: int = 5, top_n: int = 100
    ) -> list:
        candidates = retrieval_results[:top_n]
        emb_dict = self._get_clip_embedding_dict()
        features = []
        for res in candidates:
            fname = res["image_filename"]
            if fname in emb_dict:
                features.append(emb_dict[fname])
        if not features:
            return []
        features = np.stack(features, axis=0)
        kmeans = KMeans(n_clusters=desired_num, random_state=0).fit(features)
        labels = kmeans.labels_
        diverse = []
        for cid in range(desired_num):
            cluster = [candidates[i] for i, lab in enumerate(labels) if lab == cid]
            if not cluster:
                continue
            best = max(cluster, key=lambda x: x["fusion_score"])
            diverse.append(best)
        return diverse


##############################################
# 메인 실행 (Usage Example)
##############################################
if __name__ == "__main__":
    text_query = "A woman grabs a man from behind and holds a knife to his throat, threatening him."
    config_path = "config/video_retrieval_config.yaml"  # video 및 image retrieval 설정 파일 (예: merged_config.yaml)

    # Scene Retrieval (BGE)
    scene_retriever = BGERetrieval(config_path=config_path)
    # Image Retrieval: CLIP과 BLIP
    clip_retriever = CLIPRetrieval(config_path=config_path)
    blip_retriever = BLIPRetrieval(config_path=config_path)
    # Rankfusion: CLIP, BLIP, Scene retrieval 점수를 융합
    rankfusion = Rankfusion(
        scene_retriever=scene_retriever,
        clip_retriever=clip_retriever,
        blip_retriever=blip_retriever,
        weight_clip=0.4,
        weight_blip=0.4,
        weight_scene=0.2,
    )

    fusion_results = rankfusion.retrieve(text_query, top_k=1000)
    print("\n=== Rankfusion 결과 (상위 10) ===")
    for res in fusion_results[:10]:
        print(
            f"Rank {res['rank']}: {res['image_filename']} (CLIP: {res['clip_score']:.4f}, "
            f"BLIP: {res['blip_score']:.4f}, Scene: {res['scene_score']:.4f}, Fusion: {res['fusion_score']:.4f})"
        )

    diverse_results = rankfusion.select_diverse_results_by_clustering(
        fusion_results, desired_num=10, top_n=100
    )
    print("\n=== Diverse Results (Clustering) ===")
    for res in diverse_results:
        print(
            f"Rank {res['rank']}: {res['image_filename']} (Fusion: {res['fusion_score']:.4f})"
        )
