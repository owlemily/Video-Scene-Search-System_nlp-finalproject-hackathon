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


###############################################################################
# 공통: 이미지 로딩 및 전처리를 위한 Dataset 클래스 (병렬처리에 활용)
###############################################################################
class ImageDataset(Dataset):
    """
    주어진 이미지 파일 경로 리스트와 전처리 함수를 사용하여 이미지를 로딩하고 전처리합니다.
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
            print(f"이미지 로딩 오류 ({path}): {e}")
            img = Image.new("RGB", (224, 224))
        processed_img = self.preprocess(img)
        return processed_img, path


###############################################################################
# 공통: BaseRetrieval 클래스 (설정 파일 및 파일 관련 공통 기능 제공)
###############################################################################
class BaseRetrieval:
    """
    설정 파일(image_retrieval_config.yaml) 내의 특정 섹션(예: 'clip' 또는 'blip')을 로드하고,
    이미지 파일 수집, 임베딩 저장/불러오기 관련 기능을 제공합니다.
    """

    def __init__(self, config_path: str, config_section: str) -> None:
        full_config = self.load_config(config_path)
        if config_section not in full_config:
            raise KeyError(
                f"'{config_section}' 설정이 {config_path}에 존재하지 않습니다."
            )
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

    def save_config(self, config_path: str) -> None:
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, default_flow_style=False)

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


###############################################################################
# CLIPRetrieval: CLIP 모델을 사용한 이미지 검색 (병렬 임베딩 추출)
###############################################################################
class CLIPRetrieval(BaseRetrieval):
    """
    OpenAI CLIP 모델을 사용하여 이미지 임베딩을 추출하고,
    텍스트 쿼리에 대해 유사도 기반 이미지 검색을 수행합니다.
    """

    def __init__(self, config_path: str = "config/image_retrieval_config.yaml") -> None:
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
        print(f"CLIP 이미지 임베딩을 {self.embedding_file}에 저장했습니다.")

    def load_image_embeddings(self) -> None:
        data = torch.load(self.embedding_file)
        self.image_filenames = data["filenames"]
        self.image_embeddings = data["features"]
        print(f"CLIP 임베딩을 {self.embedding_file}에서 불러왔습니다.")

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


###############################################################################
# BLIPRetrieval: BLIP2 모델을 사용한 이미지 검색 (병렬 임베딩 추출)
###############################################################################
class BLIPRetrieval(BaseRetrieval):
    """
    LAVIS의 BLIP2 모델을 사용하여 이미지 임베딩을 추출하고,
    텍스트 쿼리에 대해 유사도 기반 이미지 검색을 수행합니다.
    """

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

        if os.path.exists(self.embedding_file):
            self.load_image_embeddings()
        else:
            self.extract_and_save_image_embeddings()

    def extract_and_save_image_embeddings(
        self, batch_size: int = 1024, num_workers: int = 4
    ) -> None:
        self.collect_image_filenames()
        # BLIP은 이미지 변환 시 RGB 변환이 필요합니다.
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
                # features.image_embeds_proj: [B, N, D] → 평균을 취해 [B, D]로 변환
                image_embed = features.image_embeds_proj.mean(dim=1)
                image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)
                embeddings_list.append(image_embed.cpu())
                filenames_list.extend(paths)

        self.image_embeddings = torch.cat(embeddings_list, dim=0)
        torch.save(
            {"filenames": filenames_list, "features": self.image_embeddings},
            self.embedding_file,
        )
        print(f"BLIP 이미지 임베딩을 {self.embedding_file}에 저장했습니다.")

    def load_image_embeddings(self) -> None:
        data = torch.load(self.embedding_file)
        self.image_filenames = data["filenames"]
        self.image_embeddings = data["features"]
        print(f"BLIP 임베딩을 {self.embedding_file}에서 불러왔습니다.")

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
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


###############################################################################
# RetrievalScoreFusion: CLIP과 BLIP의 결과를 rank fusion 방식으로 앙상블 및
# 클러스터링 기반 다양성 선택 기능 내장
###############################################################################
class ImageRetrievalEnsemble:
    """
    두 리트리버(CLIP, BLIP)의 점수 결과를 가중합하여 앙상블을 수행하고,
    클러스터링 기반 다양성 선택 기능을 제공합니다.
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
        # 각각의 시스템에서 top_k 결과를 얻습니다.
        clip_results = self.clip_retriever.retrieve(query, top_k=top_k)
        blip_results = self.blip_retriever.retrieve(query, top_k=top_k)

        # 각 시스템의 점수를 dict 형태로 변환 (key: 이미지 파일명, value: 점수)
        clip_score_dict = {res["image_filename"]: res["score"] for res in clip_results}
        blip_score_dict = {res["image_filename"]: res["score"] for res in blip_results}

        # 두 시스템에서 나온 모든 이미지를 합집합으로 획득
        all_images = set(clip_score_dict.keys()).union(set(blip_score_dict.keys()))

        # 각 이미지에 대해 CLIP/BLIP 점수를 가중합
        fused_scores = {}
        for img in all_images:
            score_clip = clip_score_dict.get(img, 0.0)
            score_blip = blip_score_dict.get(img, 0.0)
            fused_score = self.weight_clip * score_clip + self.weight_blip * score_blip
            fused_scores[img] = fused_score

        # 가중합 점수를 기준으로 내림차순 정렬
        sorted_images = sorted(all_images, key=lambda x: fused_scores[x], reverse=True)

        # 최종 결과 리스트 생성 (CLIP/BLIP 점수도 함께 반환)
        fused_results = []
        for rank, img in enumerate(sorted_images, start=1):
            fused_results.append(
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

        return fused_results

    def _get_clip_embedding_dict(self) -> dict:
        """
        내부적으로 CLIP 리트리버의 이미지 임베딩을 numpy 배열로 변환하여
        이미지 파일명을 key로 갖는 딕셔너리를 반환합니다.
        """
        embedding_dict = {}
        for fname, emb in zip(
            self.clip_retriever.image_filenames, self.clip_retriever.image_embeddings
        ):
            embedding_dict[fname] = emb.cpu().numpy()
        return embedding_dict

    def select_diverse_results_by_clustering(
        self,
        retrieval_results: list[dict],
        desired_num: int = 5,
        top_n: int = 100,
    ) -> list[dict]:
        """
        retrieval_results: retrieval 시스템이 반환한 결과 리스트 (예: fusion_score 기준 내림차순 정렬)
        desired_num: 최종 선택할 결과 수 (예: 5)
        top_n: diversity 적용을 위한 후보 수 (예: 상위 100개 결과)
        """
        # 후보 집합 (상위 top_n 결과)
        candidates = retrieval_results[:top_n]

        # 내부적으로 CLIP 임베딩 dict 생성
        embedding_dict = self._get_clip_embedding_dict()

        features = []
        candidate_filenames = []
        for res in candidates:
            fname = res["image_filename"]
            if fname in embedding_dict:
                features.append(embedding_dict[fname])
                candidate_filenames.append(fname)
        if len(features) == 0:
            return []

        features = np.stack(features, axis=0)

        # 클러스터 수를 desired_num으로 설정하여 KMeans 클러스터링 수행
        kmeans = KMeans(n_clusters=desired_num, random_state=0).fit(features)
        labels = kmeans.labels_

        diverse_results = []
        # 각 클러스터별로 최고 fusion_score를 가진 후보 선택
        for cluster_id in range(desired_num):
            cluster_candidates = [
                candidates[i] for i, label in enumerate(labels) if label == cluster_id
            ]
            if not cluster_candidates:
                continue
            best_candidate = max(cluster_candidates, key=lambda x: x["fusion_score"])
            diverse_results.append(best_candidate)

        return diverse_results


###############################################################################
# 사용 예시 (메인)
###############################################################################
if __name__ == "__main__":
    config_file = "config/image_retrieval_config.yaml"

    # 개별 CLIP, BLIP 검색 객체 생성
    clip_retriever = CLIPRetrieval(config_path=config_file)
    blip_retriever = BLIPRetrieval(config_path=config_file)

    # 앙상블 검색: 두 리트리버의 결과에 대해 가중치 부여 후 rank fusion 수행
    ensemble_retriever = ImageRetrievalEnsemble(
        clip_retriever,
        blip_retriever,
        weight_clip=0.4,
        weight_blip=0.6,
    )
    ensemble_query = "cabin"
    # 우선 상위 1000개의 후보 결과를 획득
    ensemble_results = ensemble_retriever.retrieve(ensemble_query, top_k=1000)
    print("\n=== 앙상블 Retrieval 결과 (후처리 전) ===")
    for res in ensemble_results[:5]:
        print(
            f"Rank {res['rank']}: {res['image_filename']} (CLIP: {res['clip_score']:.4f}, "
            f"BLIP: {res['blip_score']:.4f}, Fusion: {res['fusion_score']:.4f})"
        )

    # 클러스터링 기반 다양성 선택: 최종 desired_num 결과 추출
    diverse_results = ensemble_retriever.select_diverse_results_by_clustering(
        ensemble_results, desired_num=10, top_n=300
    )
    print("\n=== 클러스터링 기반 Diversity 적용 후 최종 결과 ===")
    for res in diverse_results:
        print(
            f"Rank {res['rank']}: {res['image_filename']} (CLIP: {res['clip_score']:.4f}, "
            f"BLIP: {res['blip_score']:.4f}, Fusion: {res['fusion_score']:.4f})"
        )
