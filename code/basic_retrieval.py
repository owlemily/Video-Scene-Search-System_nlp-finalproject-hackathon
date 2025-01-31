import json
import os

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


class BGERetrieval:
    """
    Text-based retrieval using a BGE model (e.g., a T5 or BERT-like model).
    This class loads text frame or scene descriptions from a JSON file and
    encodes them. It can then retrieve the most similar frames or scenes for a user query.
    """

    def __init__(self, config_path: str = "basic_config.yaml"):
        """
        Initializes the BGERetrieval by loading configuration, model, and data.

        :param config_path: The path to the YAML configuration file.
        """
        # Load configuration
        self.config = self.load_config(config_path)

        # Device setting
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model and tokenizer
        self.model_name = self.config["model_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)

        # Load and process text data from JSON
        self.type = self.config["type"]
        self.json_file = self.config["json_file"]
        data_json = self.load_json(self.json_file)

        # 새로운 JSON 포맷에 맞춰서 frames 또는 scenes 추출
        if self.type == "frame":
            # frames 리스트를 바로 data_info로 사용
            self.data_info = data_json["frames"]
            # 텍스트 목록: 각 frame의 "caption" 사용
            self.texts = [desc["caption"] for desc in self.data_info]
        elif self.type == "scene":
            # scenes 리스트를 바로 data_info로 사용
            self.data_info = data_json["scenes"]
            # 텍스트 목록: 각 scene의 "caption" 사용
            self.texts = [desc["caption"] for desc in self.data_info]
        else:
            raise ValueError(
                "Invalid type specified in the config file. Supported types: 'frame', 'scene'."
            )

        # Set embedding file path
        self.embedding_file = self.config["output_file"]

        # Ensure the output directory exists
        output_dir = os.path.dirname(self.embedding_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Load or encode embeddings
        if os.path.exists(self.embedding_file):
            self.load_embeddings(self.embedding_file)
        else:
            self.embeddings = self.encode_texts(self.texts)
            self.save_embeddings(self.embedding_file)

    def load_config(self, config_path: str) -> dict:
        """
        Loads YAML configuration file.

        :param config_path: Path to the YAML file.
        :return: Dictionary of configuration parameters.
        """
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def save_config(self, config_path: str) -> None:
        """
        Saves the current configuration to a YAML file.

        :param config_path: Path to save the YAML file.
        """
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def load_json(self, file_path: str):
        """
        Loads a JSON file.

        :param file_path: Path to the JSON file.
        :return: Parsed JSON (list or dict).
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def encode_texts(self, texts: list, batch_size: int = 1024) -> torch.Tensor:
        """
        Encodes a list of texts into embeddings using the BGE model.

        :param texts: List of input sentences to encode.
        :param batch_size: Number of texts to process in one batch.
        :return: Torch tensor of shape (len(texts), embedding_dim).
        """
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
                # 예: T5 계열 혹은 BERT 계열 모델에서 CLS 토큰에 해당하는 부분을 사용
                # 일반적으로 BERT 계열: outputs.last_hidden_state[:, 0, :]
                # T5 계열: 인코더 출력의 첫 토큰 등 모델별 차이 고려
                embeddings = outputs.last_hidden_state[:, 0, :]
                embeddings = embeddings / embeddings.norm(
                    dim=-1, keepdim=True
                )  # Normalize
                all_embeddings.append(
                    embeddings.cpu()
                )  # Move to CPU to save GPU memory

        return torch.cat(all_embeddings, dim=0).to(
            self.device
        )  # Move back to GPU if needed

    def save_embeddings(self, file_path: str) -> None:
        """
        Saves embeddings and descriptions to a file.

        :param file_path: Path to save the embeddings.
        """
        torch.save(
            {"data_info": self.data_info, "features": self.embeddings}, file_path
        )
        print(f"Embeddings saved to {file_path}")

    def load_embeddings(self, file_path: str) -> None:
        """
        Loads embeddings and descriptions from a file.

        :param file_path: Path to load the embeddings from.
        """
        # 일반적인 torch.load에는 weights_only 인자가 없으므로 제거
        data = torch.load(file_path)
        self.data_info = data["data_info"]
        self.embeddings = data["features"]
        print(f"Embeddings loaded from {file_path}")

    def compute_similarity(
        self, query_vector: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes cosine similarity between query vector and feature vectors.

        :param query_vector: Query embedding of shape (1, embedding_dim).
        :param features: Feature embeddings of shape (N, embedding_dim).
        :return: Similarity scores of shape (N,).
        """
        query_vector = query_vector.to(features.device).float()
        features = features.float()
        return (features @ query_vector.T).squeeze(1)

    def retrieve(self, user_query: str, top_k: int = 5) -> list:
        """
        Retrieves top-k most similar frames or scenes to the given user query.

        :param user_query: A user query string.
        :param top_k: Number of results to return.
        :return: A list of dictionaries containing rank, info, and score.
        """
        # Encode user query
        query_vector = self.encode_texts([user_query])  # shape: (1, emb_dim)

        # Compute similarity
        scores = self.compute_similarity(query_vector, self.embeddings)
        scores_np = scores.cpu().numpy()

        # Get top-k indices
        top_indices = scores_np.argsort()[-top_k:][::-1]

        # Compile results
        results = []
        for rank, idx in enumerate(top_indices, 1):
            info = self.data_info[idx]
            if self.type == "frame":
                # frame일 경우
                results.append(
                    {
                        "rank": rank,
                        "frame_timestamp": info["timestamp"],
                        "frame_image_path": info["frame_image_path"],
                        "frame_description": info["caption"],
                        "scene_id": info["scene_id"],
                        "score": float(scores_np[idx]),
                    }
                )
            elif self.type == "scene":
                # scene일 경우
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


# ================================
# Usage Example (if run directly)
# ================================
if __name__ == "__main__":
    # Initialize the retriever with the configuration
    text_config_path = "config/scene_description_config.yaml"
    text_retriever = BGERetrieval(config_path=text_config_path)

    text_query = "Reindeer"  # Example query
    text_results = text_retriever.retrieve(
        text_query, top_k=5
    )  # Retrieve top-5 similar results

    # Print the retrieval results
    print("=== BGERetrieval Results ===")
    for res in text_results:
        if "frame_timestamp" in res:
            print(
                f"Rank {res['rank']}: Timestamp={res['frame_timestamp']}, "
                f"Image={res['frame_image_path']}, Scene_id={res['scene_id']}, Score={res['score']:.4f}"
            )
        elif "scene_start_time" in res:
            print(
                f"Rank {res['rank']}: Start={res['scene_start_time']}, "
                f"End={res['scene_end_time']}, Scene_id={res['scene_id']}, Score={res['score']:.4f}"
            )
