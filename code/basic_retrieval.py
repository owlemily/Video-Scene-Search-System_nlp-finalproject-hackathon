import json
import os

import torch
import yaml
from transformers import AutoModel, AutoTokenizer


class BGERetrieval:
    """
    Text-based retrieval using a BGE model (e.g., a T5 or BERT-like model).
    This class loads text frame descriptions from a JSON file and
    encodes them. It can then retrieve the most similar frames for a user query.
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

        # Load text data from JSON
        self.frame_json_file = self.config["frame_json_file"]
        self.frame_info = self.load_json(self.frame_json_file)

        # Preprocess (collect captions)
        self.frame_texts = [desc["caption"] for desc in self.frame_info]

        # Set embedding file path
        self.embedding_file = self.config["output_file"]

        # Load or encode embeddings
        if os.path.exists(self.embedding_file):
            self.load_embeddings(self.embedding_file)
        else:
            self.frame_vectors = self.encode_texts(self.frame_texts)
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

    def load_json(self, file_path: str) -> list:
        """
        Loads a JSON file.

        :param file_path: Path to the JSON file.
        :return: Parsed JSON (list or dict).
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def encode_texts(self, texts: list) -> torch.Tensor:
        """
        Encodes a list of texts into embeddings using the BGE model.

        :param texts: List of input sentences to encode.
        :return: Torch tensor of shape (len(texts), embedding_dim).
        """
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt", max_length=512
        )
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token representation
            embeddings = outputs.last_hidden_state[:, 0, :]
            # Normalize embeddings
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings

    def save_embeddings(self, file_path: str) -> None:
        """
        Saves embeddings and descriptions to a file.

        :param file_path: Path to save the embeddings.
        """
        torch.save(
            {"frame_info": self.frame_info, "features": self.frame_vectors},
            file_path,
        )
        print(f"Embeddings saved to {file_path}")

    def load_embeddings(self, file_path: str) -> None:
        """
        Loads embeddings and descriptions from a file.

        :param file_path: Path to load the embeddings from.
        """
        data = torch.load(file_path)
        self.frame_info = data["frame_info"]
        self.frame_vectors = data["features"]
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
        Retrieves top-k most similar frames to the given user query.

        :param user_query: A user query string.
        :param top_k: Number of results to return.
        :return: A list of dictionaries containing rank, frame info, and score.
        """
        # Encode user query
        query_vector = self.encode_texts([user_query])  # shape: (1, emb_dim)

        # Compute similarity
        scores = self.compute_similarity(query_vector, self.frame_vectors)
        scores_np = scores.cpu().numpy()

        # Get top-k indices
        top_indices = scores_np.argsort()[-top_k:][::-1]

        # Compile results
        results = []
        for rank, idx in enumerate(top_indices, 1):
            frame_info = self.frame_info[idx]
            results.append(
                {
                    "rank": rank,
                    "frame_timestamp": frame_info["timestamp"],
                    "frame_image_path": frame_info["frame_image_path"],
                    "frame_description": frame_info["caption"],
                    "score": float(scores_np[idx]),
                }
            )

        return results


# ================================
# Usage Example (if run directly)
# ================================
if __name__ == "__main__":
    # Example for text retrieval

    # Initialize the retriever with the configuration
    text_config_path = "config/basic_config.yaml"
    text_retriever = BGERetrieval(config_path=text_config_path)

    text_query = "Reindeer"  # Example query
    text_results = text_retriever.retrieve(
        text_query, top_k=5
    )  # Retrieve top-5 similar frames

    # Print the retrieval results
    print("=== BGERetrieval Results ===")
    for res in text_results:
        print(
            f"Rank {res['rank']}:"
            f" Timestamp={res['frame_timestamp']},"
            f" Image={res['frame_image_path']},"
            # f" Desc={res['frame_description']},"
            f" Score={res['score']:.4f}"
        )
