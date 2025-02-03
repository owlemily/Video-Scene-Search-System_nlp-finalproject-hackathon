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
# 1. Scene Text Retrieval (BGE 기반)
##############################################
class BGERetrieval:
    """
    JSON 파일에서 scene 정보를 불러와 BGE 모델로 임베딩한 후,
    사용자 쿼리와 코사인 유사도를 계산하여 유사한 scene을 검색합니다.
    """

    def __init__(self, config_path: str = "config/merged_config.yaml"):
        """
        :param config_path: 통합 설정 파일 경로. 내부에서 "bge-scene" 섹션을 사용합니다.
        """
        # 전체 설정 파일에서 "bge-scene" 섹션만 사용
        full_config = self.load_config(config_path)
        self.config = full_config["bge-scene"]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 모델 및 토크나이저 로드
        self.model_name = self.config["model_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)

        # JSON 파일에서 scene 데이터 로드 ("scenes" 키 사용)
        self.json_file = self.config["json_file"]
        data_json = self.load_json(self.json_file)
        self.data_info = data_json["scenes"]
        self.texts = [scene["caption"] for scene in self.data_info]

        # 임베딩 저장 경로 설정 및 디렉토리 생성
        self.embedding_file = self.config["output_file"]
        os.makedirs(os.path.dirname(self.embedding_file), exist_ok=True)

        # 임베딩 파일이 있으면 로드, 없으면 생성
        if os.path.exists(self.embedding_file):
            self.load_embeddings(self.embedding_file)
        else:
            self.embeddings = self.encode_texts(self.texts)
            self.save_embeddings(self.embedding_file)

    def load_config(self, config_path: str) -> dict:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def load_json(self, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def encode_texts(self, texts: list, batch_size: int = 1024) -> torch.Tensor:
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
            batch_texts = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 여기서는 CLS 토큰(첫 번째 토큰) 사용
                embeddings = outputs.last_hidden_state[:, 0, :]
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                all_embeddings.append(embeddings.cpu())
        return torch.cat(all_embeddings, dim=0).to(self.device)

    def save_embeddings(self, file_path: str) -> None:
        torch.save(
            {"data_info": self.data_info, "features": self.embeddings}, file_path
        )
        print(f"[BGE] Embeddings saved to {file_path}")

    def load_embeddings(self, file_path: str) -> None:
        data = torch.load(file_path, weights_only=True)
        self.data_info = data["data_info"]
        self.embeddings = data["features"]
        print(f"[BGE] Embeddings loaded from {file_path}")

    def compute_similarity(
        self, query_vector: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        query_vector = query_vector.to(features.device).float()
        features = features.float()
        return (features @ query_vector.T).squeeze(1)

    def retrieve(self, user_query: str, top_k: int = 5) -> list:
        query_vector = self.encode_texts([user_query])
        scores = self.compute_similarity(query_vector, self.embeddings)
        scores_np = scores.cpu().numpy()
        top_indices = scores_np.argsort()[-top_k:][::-1]
        results = []
        for rank, idx in enumerate(top_indices, 1):
            info = self.data_info[idx]
            results.append(
                {
                    "rank": rank,
                    "scene_start_time": info["start_time"],
                    "scene_end_time": info["end_time"],
                    "scene_description": info["caption"],
                    "scene_id": info["scene_id"],
                    "score": float(scores_np[idx]),
                }
            )
        return results


##############################################
# 2. Image Retrieval 관련 공통 클래스 및 Dataset
##############################################
class ImageDataset(Dataset):
    """
    이미지 파일 경로와 전처리 함수를 사용해 이미지를 로드합니다.
    """

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
            print(f"[ImageDataset] Error loading {path}: {e}")
            img = Image.new("RGB", (224, 224))
        return self.preprocess(img), path


class BaseRetrieval:
    """
    이미지 검색에 필요한 파일 목록 수집 및 임베딩 저장/불러오기 기능 제공.
    """

    def __init__(self, config_path: str, config_section: str) -> None:
        full_config = self.load_config(config_path)
        if config_section not in full_config:
            raise KeyError(f"'{config_section}' section not found in {config_path}")
        self.config = full_config[config_section]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_folder = self.config["image_folder"]
        self.embedding_file = self.config["embedding_file"]
        self.image_extensions = tuple(self.config["image_extensions"])

        self.image_filenames: list[str] = []
        self.image_embeddings: torch.Tensor | None = None

        self._ensure_output_dir()

    def load_config(self, config_path: str) -> dict:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _ensure_output_dir(self) -> None:
        output_dir = os.path.dirname(self.embedding_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def collect_image_filenames(self) -> None:
        self.image_filenames = [
            os.path.join(self.image_folder, fname)
            for fname in sorted(os.listdir(self.image_folder))
            if fname.endswith(self.image_extensions)
        ]


##############################################
# 2-1. CLIP 기반 이미지 Retrieval
##############################################
class CLIPRetrieval(BaseRetrieval):
    """
    CLIP 모델을 사용하여 이미지 임베딩을 추출하고, 텍스트 쿼리와의 유사도로 검색합니다.
    """

    def __init__(self, config_path: str = "config/merged_config.yaml") -> None:
        super().__init__(config_path, config_section="clip")
        self.model, self.preprocess = clip.load(
            self.config["model_name"], device=self.device
        )
        if os.path.exists(self.embedding_file):
            self.load_image_embeddings()
        else:
            self.extract_and_save_image_embeddings()

    def extract_and_save_image_embeddings(
        self, batch_size: int = 1024, num_workers: int = 4
    ) -> None:
        self.collect_image_filenames()
        dataset = ImageDataset(self.image_filenames, self.preprocess)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
        )
        embeddings_list = []
        filenames_list = []
        with torch.no_grad():
            for images, paths in tqdm(dataloader, desc="Extracting CLIP embeddings"):
                images = images.to(self.device)
                features = self.model.encode_image(images)
                features = features / features.norm(dim=-1, keepdim=True)
                embeddings_list.append(features.cpu())
                filenames_list.extend(paths)
        self.image_embeddings = torch.cat(embeddings_list, dim=0)
        torch.save(
            {"filenames": filenames_list, "features": self.image_embeddings},
            self.embedding_file,
        )
        print(f"[CLIP] Embeddings saved to {self.embedding_file}")

    def load_image_embeddings(self) -> None:
        data = torch.load(self.embedding_file, weights_only=True)
        self.image_filenames = data["filenames"]
        self.image_embeddings = data["features"]
        print(f"[CLIP] Embeddings loaded from {self.embedding_file}")

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        text_tokens = clip.tokenize([query]).to(self.device)
        with torch.no_grad():
            text_embedding = self.model.encode_text(text_tokens)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        similarity = (self.image_embeddings.to(self.device) @ text_embedding.T).squeeze(
            1
        )
        top_indices = similarity.topk(top_k).indices.cpu().numpy()
        return [
            {
                "rank": rank,
                "image_filename": self.image_filenames[idx],
                "score": float(similarity[idx].item()),
            }
            for rank, idx in enumerate(top_indices, start=1)
        ]


##############################################
# 2-2. BLIP 기반 이미지 Retrieval
##############################################
class BLIPRetrieval(BaseRetrieval):
    """
    BLIP2 모델을 사용하여 이미지 임베딩을 추출하고, 텍스트 쿼리와의 유사도로 검색합니다.
    """

    def __init__(self, config_path: str = "config/merged_config.yaml") -> None:
        super().__init__(config_path, config_section="blip")
        self.model, self.vis_processors, self.txt_processors = (
            load_model_and_preprocess(
                name="blip2_feature_extractor",
                model_type=self.config.get("model_type", "pretrain"),
                is_eval=True,
                device=self.device,
            )
        )
        if os.path.exists(self.embedding_file):
            self.load_image_embeddings()
        else:
            self.extract_and_save_image_embeddings()

    def extract_and_save_image_embeddings(
        self, batch_size: int = 1024, num_workers: int = 4
    ) -> None:
        self.collect_image_filenames()
        dataset = ImageDataset(
            self.image_filenames, self.vis_processors["eval"], convert_mode="RGB"
        )
        dataloader = DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
        )
        embeddings_list = []
        filenames_list = []
        with torch.no_grad():
            for images, paths in tqdm(dataloader, desc="Extracting BLIP embeddings"):
                images = images.to(self.device)
                features = self.model.extract_features({"image": images}, mode="image")
                image_embed = features.image_embeds_proj.mean(dim=1)
                image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)
                embeddings_list.append(image_embed.cpu())
                filenames_list.extend(paths)
        self.image_embeddings = torch.cat(embeddings_list, dim=0)
        torch.save(
            {"filenames": filenames_list, "features": self.image_embeddings},
            self.embedding_file,
        )
        print(f"[BLIP] Embeddings saved to {self.embedding_file}")

    def load_image_embeddings(self) -> None:
        data = torch.load(self.embedding_file, weights_only=True)
        self.image_filenames = data["filenames"]
        self.image_embeddings = data["features"]
        print(f"[BLIP] Embeddings loaded from {self.embedding_file}")

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        text_input = self.txt_processors["eval"](query)
        sample = {"text_input": [text_input]}
        with torch.no_grad():
            features_text = self.model.extract_features(sample, mode="text")
            text_embed = features_text.text_embeds_proj
            if text_embed.dim() == 3 and text_embed.size(1) > 1:
                text_embed = text_embed.mean(dim=1)
            elif text_embed.dim() != 2:
                raise ValueError(f"Unexpected text embedding shape: {text_embed.shape}")
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
        similarity = (self.image_embeddings.to(self.device) @ text_embed.t()).squeeze(1)
        top_indices = similarity.topk(top_k).indices.cpu().numpy()
        return [
            {
                "rank": rank,
                "image_filename": self.image_filenames[idx],
                "score": float(similarity[idx].item()),
            }
            for rank, idx in enumerate(top_indices, start=1)
        ]


##############################################
# 3. 앙상블 및 Diversity 선택 (클러스터링 기반)
##############################################
class ImageRetrievalEnsemble:
    """
    CLIP과 BLIP 결과를 가중합하여 앙상블하고,
    BLIP 임베딩 기반 클러스터링으로 diverse한 대표 결과를 선택합니다.
    """

    def __init__(
        self,
        clip_retriever: CLIPRetrieval,
        blip_retriever: BLIPRetrieval,
        weight_clip: float = 0.5,
        weight_blip: float = 0.5,
    ):
        self.clip_retriever = clip_retriever
        self.blip_retriever = blip_retriever
        self.weight_clip = weight_clip
        self.weight_blip = weight_blip

    def retrieve(self, query: str, top_k: int = 100) -> list[dict]:
        clip_results = self.clip_retriever.retrieve(query, top_k=top_k)
        blip_results = self.blip_retriever.retrieve(query, top_k=top_k)
        clip_score_dict = {res["image_filename"]: res["score"] for res in clip_results}
        blip_score_dict = {res["image_filename"]: res["score"] for res in blip_results}
        all_images = set(clip_score_dict.keys()).union(set(blip_score_dict.keys()))
        fused_scores = {}
        for img in all_images:
            score_clip = clip_score_dict.get(img, 0.0)
            score_blip = blip_score_dict.get(img, 0.0)
            fused_scores[img] = (
                self.weight_clip * score_clip + self.weight_blip * score_blip
            )
        sorted_results = []
        for rank, img in enumerate(
            sorted(all_images, key=lambda x: fused_scores[x], reverse=True), start=1
        ):
            sorted_results.append(
                {
                    "rank": rank,
                    "image_filename": img,
                    "clip_score": clip_score_dict.get(img, 0.0),
                    "blip_score": blip_score_dict.get(img, 0.0),
                    "fusion_score": fused_scores[img],
                }
            )
            if rank >= top_k:
                break
        return sorted_results

    def select_diverse_results_by_clustering(
        self, retrieval_results: list[dict], desired_num: int = 5, top_n: int = 100
    ) -> list[dict]:
        candidates = retrieval_results[:top_n]
        # BLIP 임베딩을 기반으로 diversity 선택
        blip_embedding_dict = {
            fname: emb.cpu().numpy()
            for fname, emb in zip(
                self.blip_retriever.image_filenames,
                self.blip_retriever.image_embeddings,
            )
        }
        features = []
        for res in candidates:
            fname = res["image_filename"]
            if fname in blip_embedding_dict:
                features.append(blip_embedding_dict[fname])
        if not features:
            return []
        features = np.stack(features, axis=0)
        kmeans = KMeans(n_clusters=desired_num, random_state=0).fit(features)
        labels = kmeans.labels_
        diverse_results = []
        for cluster_id in range(desired_num):
            cluster_candidates = [
                candidates[i] for i, label in enumerate(labels) if label == cluster_id
            ]
            if not cluster_candidates:
                continue
            best_candidate = max(cluster_candidates, key=lambda x: x["fusion_score"])
            diverse_results.append(best_candidate)
        return diverse_results


class Rankfusion:
    """
    scene_retriever, clip_retriever, blip_retriever의 retrieval 결과를 각각의 가중치 (weight_scene, weight_clip, weight_blip)
    로 결합하여 최종 랭크 퓨전을 수행합니다.

    특히, 이미지 파일명 (예: "efqjl_jfdk_95.720625.jpg" 또는 "efqjl_jfdk_extra_95.720625.jpg") 에서
    video_id와 timestamp를 추출한 후, scene_retriever가 반환한 scene 정보 (scene_start_time, scene_end_time, score)를 참고하여
    해당 이미지 프레임이 속한 scene의 점수를 fusion에 포함합니다.
    """

    def __init__(
        self,
        scene_retriever,
        clip_retriever,
        blip_retriever,
        weight_clip: float = 0.33,
        weight_blip: float = 0.33,
        weight_scene: float = 0.34,
    ):
        """
        :param scene_retriever: BGE 기반 scene 검색 인스턴스
        :param clip_retriever: CLIP 기반 이미지 검색 인스턴스
        :param blip_retriever: BLIP 기반 이미지 검색 인스턴스
        :param weight_clip: CLIP 결과에 부여할 가중치
        :param weight_blip: BLIP 결과에 부여할 가중치
        :param weight_scene: Scene 결과에 부여할 가중치
        """
        self.scene_retriever = scene_retriever
        self.clip_retriever = clip_retriever
        self.blip_retriever = blip_retriever
        self.weight_clip = weight_clip
        self.weight_blip = weight_blip
        self.weight_scene = weight_scene

    def normalize_scores(self, results):
        """
        주어진 결과 리스트에서 score 값을 0~1 사이로 정규화(Min-Max)하는 함수.
        (Z-score 등 다른 방법도 가능)
        """
        if not results:
            return results

        scores = [d["score"] for d in results]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            # 모든 score가 동일한 경우 0.5로 설정
            normalized_scores = [0.5] * len(scores)
        else:
            normalized_scores = [
                (x - min_score) / (max_score - min_score) for x in scores
            ]

        for i, d in enumerate(results):
            d["score"] = normalized_scores[i]

        return results

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        union_top_n: int = None,  # None이면 전체 이미지 수를 사용
    ) -> list[dict]:
        """
        :param query: 검색할 텍스트 쿼리
        :param top_k: 최종 반환할 결과 수
        :param union_top_n: CLIP, BLIP retrieval 시 union으로 고려할 상위 결과 수 (None이면 전체)
        :return: 각 결과 dict에는 image_filename, clip_score, blip_score, scene_score, fusion_score, rank 등이 포함됨
        """
        import os

        # 만약 union_top_n이 None이면, 현재 로드된 모든 이미지 개수를 사용
        if union_top_n is None:
            union_top_n = max(
                len(self.clip_retriever.image_filenames),
                len(self.blip_retriever.image_filenames),
            )

        # 1. CLIP, BLIP retrieval 결과 (union_top_n으로 검색)
        clip_results = self.clip_retriever.retrieve(query, top_k=union_top_n)
        clip_results = self.normalize_scores(clip_results)

        blip_results = self.blip_retriever.retrieve(query, top_k=union_top_n)
        blip_results = self.normalize_scores(blip_results)

        # 각 retrieval 결과에서 이미지 파일명과 점수를 딕셔너리로 정리 (없으면 0)
        clip_score_dict = {
            res["image_filename"]: res.get("score", 0.0) for res in clip_results
        }
        blip_score_dict = {
            res["image_filename"]: res.get("score", 0.0) for res in blip_results
        }

        # 2. Scene retrieval 결과: scene_retriever는 전체 scene에 대해 query 유사도 점수를 반환함.
        scene_results = self.scene_retriever.retrieve(
            query, top_k=len(self.scene_retriever.data_info)
        )
        scene_results = self.normalize_scores(scene_results)

        # scene 정보를 video_id별로 그룹핑
        scenes_by_video = {}
        for scene in scene_results:
            # scene_id 예시: "myvideo_12.34_15.20_001"
            # video_id 추출 시, 필요에 따라 rsplit 파라미터 조정
            parts = scene["scene_id"].rsplit("_", 4)
            video_id = parts[0]
            if video_id not in scenes_by_video:
                scenes_by_video[video_id] = []
            scenes_by_video[video_id].append(scene)

        # 3. CLIP, BLIP retrieval 결과의 union
        all_images = set(list(clip_score_dict.keys()) + list(blip_score_dict.keys()))
        fused_results = []
        for img in all_images:
            clip_score = clip_score_dict.get(img, 0.0)
            blip_score = blip_score_dict.get(img, 0.0)
            scene_score = 0.0  # 초기값

            # 파일명에서 video_id와 timestamp를 추출
            # 파일명 예: "efqjl_jfdk_95.720625.jpg"
            base = os.path.basename(img)
            try:
                video_id_part, ts_with_ext = base.rsplit("_", 1)
                video_id_extracted = video_id_part
                timestamp_str, _ = os.path.splitext(ts_with_ext)
                timestamp = float(timestamp_str)
            except Exception as e:
                print(f"[Rankfusion] 파일명 {img} 파싱 에러: {e}")
                timestamp = None
                video_id_extracted = None

            # scene_score 찾기
            if timestamp is not None and video_id_extracted in scenes_by_video:
                candidate_scenes = [
                    s
                    for s in scenes_by_video[video_id_extracted]
                    if float(s["scene_start_time"])
                    <= timestamp
                    <= float(s["scene_end_time"])
                ]
                if candidate_scenes:
                    # 해당 프레임이 속하는 scene들 중 최대 점수를 사용
                    scene_score = max(s["score"] for s in candidate_scenes)

            # 최종 fusion score 계산
            fusion_score = (
                self.weight_clip * clip_score
                + self.weight_blip * blip_score
                + self.weight_scene * scene_score
            )

            fused_results.append(
                {
                    "image_filename": img,
                    "clip_score": clip_score,
                    "blip_score": blip_score,
                    "scene_score": scene_score,
                    "fusion_score": fusion_score,
                }
            )

        # fusion score 내림차순 정렬 및 rank 부여
        fused_results = sorted(
            fused_results, key=lambda x: x["fusion_score"], reverse=True
        )
        for rank, res in enumerate(fused_results, start=1):
            res["rank"] = rank

        return fused_results[:top_k]

    def select_diverse_results_by_clustering(
        self, fused_results: list[dict], desired_num: int = 5, top_n: int = 100
    ) -> list[dict]:
        """
        BLIP 임베딩을 기반으로 클러스터링 후, 다양한(results) 대표 이미지를 뽑아냅니다.
        :param fused_results: 이미 scene, clip, blip를 합산하여 rankfusion된 결과 리스트
        :param desired_num: 뽑고자 하는 다양성 대표 결과 수 (클러스터 수)
        :param top_n: 우선적으로 상위 top_n 후보를 대상으로 클러스터링을 수행
        :return: 클러스터링 후, 각 클러스터 내에서 가장 fusion_score 높은 이미지를 뽑아낸 결과 리스트
        """
        import numpy as np
        from sklearn.cluster import KMeans

        # 1. 우선 fused_results 상위 top_n만 후보로 선정
        candidates = fused_results[:top_n]

        # 2. BLIP 임베딩 딕셔너리 생성
        blip_embedding_dict = {
            fname: emb.cpu().numpy()
            for fname, emb in zip(
                self.blip_retriever.image_filenames,
                self.blip_retriever.image_embeddings,
            )
        }

        # 3. 후보들의 BLIP 임베딩을 모아서 KMeans 클러스터링
        features = []
        valid_candidates = []
        for res in candidates:
            fname = res["image_filename"]
            if fname in blip_embedding_dict:
                features.append(blip_embedding_dict[fname])
                valid_candidates.append(res)

        if not features:
            # 후보들의 임베딩을 찾을 수 없는 경우
            return []

        features = np.stack(features, axis=0)

        # 4. k-means 클러스터링
        kmeans = KMeans(n_clusters=desired_num, random_state=0)
        labels = kmeans.fit_predict(features)

        # 5. 각 클러스터 내에서 fusion_score가 가장 높은 결과 선택
        diverse_results = []
        for cluster_id in range(desired_num):
            cluster_candidates = [
                valid_candidates[i]
                for i, label in enumerate(labels)
                if label == cluster_id
            ]
            if not cluster_candidates:
                continue
            best_candidate = max(cluster_candidates, key=lambda x: x["fusion_score"])
            diverse_results.append(best_candidate)

        # 6. 클러스터링 결과를 최종 점수 순으로 정렬 (선택 사항)
        diverse_results = sorted(
            diverse_results, key=lambda x: x["fusion_score"], reverse=True
        )

        return diverse_results


##############################################
# 메인 실행 (Usage Example)
##############################################
if __name__ == "__main__":
    # 검색할 텍스트 쿼리
    text_query = "A woman grabs a man from behind and holds a knife to his throat, threatening him."

    # 1. Scene Text Retrieval
    config_path = "config/video_retrieval_config.yaml"
    scene_retriever = BGERetrieval(config_path=config_path)
    # scene_results = scene_retriever.retrieve(text_query, top_k=5)
    # print("=== BGERetrieval (Scene 검색) 결과 ===")
    # for res in scene_results:
    #     print(
    #         f"Rank {res['rank']}: Start={res['scene_start_time']}, End={res['scene_end_time']}, "
    #         f"Scene_id={res['scene_id']}, Score={res['score']:.4f}"
    #     )

    # 2. Image Retrieval: CLIP, BLIP, 앙상블 및 Diversity 선택
    clip_retriever = CLIPRetrieval(config_path=config_path)
    blip_retriever = BLIPRetrieval(config_path=config_path)
    # ensemble_retriever = ImageRetrievalEnsemble(
    #     clip_retriever, blip_retriever, weight_clip=0.4, weight_blip=0.6
    # )
    # ensemble_results = ensemble_retriever.retrieve(text_query, top_k=100)
    # print("\n=== CLIP & BLIP 앙상블 Retrieval 결과 (후처리 전, 상위 5) ===")
    # for res in ensemble_results[:5]:
    #     print(
    #         f"Rank {res['rank']}: {res['image_filename']} (CLIP: {res['clip_score']:.4f}, "
    #         f"BLIP: {res['blip_score']:.4f}, Fusion: {res['fusion_score']:.4f})"
    #     )

    # diverse_results = ensemble_retriever.select_diverse_results_by_clustering(
    #     ensemble_results, desired_num=5, top_n=100
    # )
    # print("\n=== 클러스터링 기반 Diversity 적용 후 결과 ===")
    # for res in diverse_results:
    #     print(
    #         f"Rank {res['rank']}: {res['image_filename']} (CLIP: {res['clip_score']:.4f}, "
    #         f"BLIP: {res['blip_score']:.4f}, Fusion: {res['fusion_score']:.4f})"
    #     )

    # Rankfusion 객체 생성 (각 retrieval 결과의 가중치는 상황에 따라 조절)
    rankfusion = Rankfusion(
        scene_retriever=scene_retriever,
        clip_retriever=clip_retriever,
        blip_retriever=blip_retriever,
        weight_clip=0.4,
        weight_blip=0.4,
        weight_scene=0.2,
    )

    # 최종 Rankfusion 결과 (예: 상위 10개 결과)
    fusion_results = rankfusion.retrieve(text_query, top_k=1000)
    print("\n=== Rankfusion 결과 ===")
    for res in fusion_results[:10]:
        print(
            f"Rank {res['rank']}: {res['image_filename']} "
            f"(CLIP: {res['clip_score']:.4f}, BLIP: {res['blip_score']:.4f}, "
            f"Scene: {res['scene_score']:.4f}, Fusion: {res['fusion_score']:.4f})"
        )

    diverse_results = rankfusion.select_diverse_results_by_clustering(
        fusion_results, desired_num=10, top_n=100
    )
    print("=== Diverse Results from Clustering ===")
    for res in diverse_results:
        print(
            f"Rank {res['rank']}: {res['image_filename']} "
            f"(Fusion: {res['fusion_score']:.4f})"
        )
