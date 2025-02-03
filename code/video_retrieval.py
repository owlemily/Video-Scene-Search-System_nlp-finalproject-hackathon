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

    특히, 이미지 파일명 (예: "efqjl_jfdk_95.720625.jpg") 에서 video_id와 timestamp를 추출한 후,
    scene_retriever가 반환한 scene 정보 (scene_start_time, scene_end_time, score)를 참고하여
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
        self.scene_retriever = scene_retriever
        self.clip_retriever = clip_retriever
        self.blip_retriever = blip_retriever
        self.weight_clip = weight_clip
        self.weight_blip = weight_blip
        self.weight_scene = weight_scene

    def normalize_scores(self, results):
        if not results:
            return results

        scores = [d["score"] for d in results]
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            normalized_scores = [0.5] * len(scores)
        else:
            normalized_scores = [
                (x - min_score) / (max_score - min_score) for x in scores
            ]

        for i, d in enumerate(results):
            d["score"] = normalized_scores[i]
        return results

    def _build_scene_lookup_table(self, image_filenames, scenes_by_video):
        """
        이미지 파일명 리스트에 대해, 각 이미지가 속한 scene의 점수를 미리 계산하여 룩업 테이블을 생성합니다.
        :param image_filenames: 이미지 파일명들의 집합 (set 혹은 list)
        :param scenes_by_video: video_id를 키로 갖는 scene 리스트 딕셔너리
        :return: { image_filename: scene_score, ... }
        """
        scene_lookup = {}
        import os

        for img in image_filenames:
            base = os.path.basename(img)
            name, ext = os.path.splitext(base)
            parts = name.split("_")
            # 마지막 토큰은 timestamp로 간주합니다.
            try:
                timestamp = float(parts[-1])
            except ValueError as e:
                print(f"[Rankfusion] Timestamp parsing error for {img}: {e}")
                timestamp = None

            # video_id는 마지막 토큰을 제외한 모든 토큰을 결합하여 추출합니다.
            video_id_extracted = "_".join(parts[:-1])

            scene_score = 0.0
            # video_id를 먼저 찾고, 해당 video_id에 속하는 scene 목록에서 timestamp 범위에 포함되는 scene을 찾음
            if timestamp is not None and video_id_extracted in scenes_by_video:
                candidate_scenes = []
                for scene in scenes_by_video[video_id_extracted]:
                    try:
                        start_time = float(scene["scene_start_time"])
                        end_time = float(scene["scene_end_time"])
                    except Exception as e:
                        print(f"[Rankfusion] Scene time parsing error: {e}")
                        continue
                    if start_time <= timestamp <= end_time:
                        candidate_scenes.append(scene)
                if candidate_scenes:
                    scene_score = max(s["score"] for s in candidate_scenes)
            scene_lookup[img] = scene_score

        return scene_lookup

    def retrieve(
        self, query: str, top_k: int = 10, union_top_n: int = None
    ) -> list[dict]:
        """
        :param query: 검색할 텍스트 쿼리
        :param top_k: 최종 반환할 결과 수
        :param union_top_n: CLIP, BLIP retrieval 시 union으로 고려할 상위 결과 수 (None이면 전체)
        :return: 결과 dict에는 image_filename, clip_score, blip_score, scene_score, fusion_score, rank 등이 포함됨
        """
        if union_top_n is None:
            union_top_n = max(
                len(self.clip_retriever.image_filenames),
                len(self.blip_retriever.image_filenames),
            )

        clip_results = self.clip_retriever.retrieve(query, top_k=union_top_n)
        # clip_results = self.normalize_scores(clip_results)

        blip_results = self.blip_retriever.retrieve(query, top_k=union_top_n)
        # blip_results = self.normalize_scores(blip_results)

        clip_score_dict = {
            res["image_filename"]: res.get("score", 0.0) for res in clip_results
        }
        blip_score_dict = {
            res["image_filename"]: res.get("score", 0.0) for res in blip_results
        }

        # 1. Scene retrieval 결과 (전체 scene에 대해 검색)
        scene_results = self.scene_retriever.retrieve(
            query, top_k=len(self.scene_retriever.data_info)
        )
        # scene_results = self.normalize_scores(scene_results)

        # 2. scene 정보를 video_id별로 그룹핑
        # video_id가 언더스코어를 포함할 수 있으므로, scene_id에서 마지막 3개의 토큰(시작시간, 종료시간, 시퀀스 번호)을 제거하고 나머지를 video_id로 사용합니다.
        scenes_by_video = {}
        for scene in scene_results:
            parts = scene["scene_id"].rsplit("_", 3)
            video_id = parts[0]
            if video_id not in scenes_by_video:
                scenes_by_video[video_id] = []
            scenes_by_video[video_id].append(scene)

        # 3. CLIP, BLIP 결과의 union
        all_images = set(list(clip_score_dict.keys()) + list(blip_score_dict.keys()))
        scene_lookup = self._build_scene_lookup_table(all_images, scenes_by_video)

        fused_results = []
        for img in all_images:
            clip_score = clip_score_dict.get(img, 0.0)
            blip_score = blip_score_dict.get(img, 0.0)
            scene_score = scene_lookup.get(img, 0.0)
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

        fused_results = sorted(
            fused_results, key=lambda x: x["fusion_score"], reverse=True
        )
        for rank, res in enumerate(fused_results, start=1):
            res["rank"] = rank

        return fused_results[:top_k]

    def select_diverse_results_by_clustering(
        self, fused_results: list[dict], desired_num: int = 5, top_n: int = 100
    ) -> list[dict]:
        import numpy as np
        from sklearn.cluster import KMeans

        candidates = fused_results[:top_n]
        # 여기서는 CLIP 임베딩 기반으로 diversity 선택 (필요에 따라 BLIP 임베딩 등으로 변경 가능)
        clip_embedding_dict = {
            fname: emb.cpu().numpy()
            for fname, emb in zip(
                self.clip_retriever.image_filenames,
                self.clip_retriever.image_embeddings,
            )
        }

        features = []
        valid_candidates = []
        for res in candidates:
            fname = res["image_filename"]
            if fname in clip_embedding_dict:
                features.append(clip_embedding_dict[fname])
                valid_candidates.append(res)

        if not features:
            return []

        features = np.stack(features, axis=0)
        kmeans = KMeans(n_clusters=desired_num, random_state=0)
        labels = kmeans.fit_predict(features)

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
