import os

import clip
import torch
import yaml
from lavis.models import load_model_and_preprocess
from PIL import Image
from tqdm import tqdm


class CLIPRetrieval:
    """
    Image-based retrieval using OpenAI CLIP model.
    This class extracts or loads precomputed features for images, and
    retrieves the most similar images given a user query.
    """

    def __init__(self, config_path: str = "clip_config.yaml"):
        """
        Initializes the CLIPRetrieval by loading configuration and model.

        :param config_path: The path to the YAML configuration file.
        """
        # Load configuration
        self.config = self.load_config(config_path)

        # Device setting
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load CLIP model and preprocess
        self.model, self.preprocess = clip.load(
            self.config["model_name"], device=self.device
        )

        # Set file/directory paths
        self.frame_folder = self.config["frame_folder"]
        self.embedding_file = self.config["output_file"]
        self.image_extensions = tuple(self.config["image_extensions"])

        # Internal attributes
        self.image_filenames = []
        self.image_embeddings = None

        # Ensure the output directory exists
        output_dir = os.path.dirname(self.embedding_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Load precomputed image features or extract them
        if os.path.exists(self.embedding_file):
            self.load_image_embeddings()
        else:
            self.extract_and_save_image_embeddings()

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

    def collect_image_filenames(self) -> None:
        """
        Collects and sorts image filenames from the frame folder.
        """
        all_files = os.listdir(self.frame_folder)
        self.image_filenames = [
            os.path.join(self.frame_folder, fname)
            for fname in sorted(all_files)
            if fname.endswith(self.image_extensions)
        ]

    def extract_and_save_image_embeddings(self) -> None:
        """
        Extracts features for each image using CLIP and saves them to embedding_file.
        """
        self.collect_image_filenames()
        image_embeddings = []

        for image_path in tqdm(
            self.image_filenames, desc="Processing images embeddings"
        ):
            try:
                image = Image.open(image_path)  # .convert("RGB")
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    feature = self.model.encode_image(image_input)
                    # Normalize feature
                    feature /= feature.norm(dim=-1, keepdim=True)
                    image_embeddings.append(feature.cpu())

            except Exception as e:
                print(f"Image processing error ({image_path}): {e}")

        self.image_embeddings = torch.cat(image_embeddings, dim=0)
        torch.save(
            {"filenames": self.image_filenames, "features": self.image_embeddings},
            self.embedding_file,
        )
        print(f"Image features(embeddings) saved to: {self.embedding_file}")

    def load_image_embeddings(self) -> None:
        """
        Loads precomputed image features from the output file.
        """
        data = torch.load(self.embedding_file, weights_only=True)
        self.image_filenames = data["filenames"]
        self.image_embeddings = data["features"]
        print(f"Embeddings loaded from {self.embedding_file}")

    def retrieve(self, query: str, top_k: int = 10) -> list:
        """
        Finds the top-k most similar images given a text query.

        :param query: Text query string.
        :param top_k: Number of results to retrieve.
        :return: A list of dictionaries containing rank, image_filename, and similarity score.
        """
        # Encode query text
        text_tokens = clip.tokenize([query]).to(self.device)
        with torch.no_grad():
            text_embeddings = self.model.encode_text(text_tokens)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)

        # Compute similarity
        similarity = (
            self.image_embeddings.to(self.device) @ text_embeddings.T
        ).squeeze(1)

        # Get top-k indices
        top_indices = similarity.topk(top_k).indices.cpu().numpy()

        # Compile results
        results = []
        for rank, index in enumerate(top_indices, 1):
            results.append(
                {
                    "rank": rank,
                    "image_filename": self.image_filenames[index],
                    "score": float(similarity[index].item()),
                }
            )
        return results


# ================================
# BLIPRetrieval Class Definition
# ================================


class BLIPRetrieval:
    """
    Image-based retrieval using BLIP2 model from LAVIS.
    This class extracts or loads precomputed features for images, and
    retrieves the most similar images given a user query.
    """

    def __init__(self, config_path: str = "blip_config.yaml"):
        """
        Initializes the BLIPRetrieval by loading configuration and model.

        :param config_path: The path to the YAML configuration file.
        """
        # Load configuration
        self.config = self.load_config(config_path)

        # Device setting
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load BLIP2 model and preprocess
        self.model, self.vis_processors, self.txt_processors = (
            load_model_and_preprocess(
                name="blip2_feature_extractor",
                model_type=self.config.get("model_type", "pretrain"),
                is_eval=True,
                device=self.device,
            )
        )

        # Set file/directory paths
        self.frame_folder = self.config["frame_folder"]
        self.embedding_file = self.config["output_file"]
        self.image_extensions = tuple(self.config["image_extensions"])

        # Internal attributes
        self.image_filenames = []
        self.image_embeddings = None

        # Ensure the output directory exists
        output_dir = os.path.dirname(self.embedding_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Load precomputed image features or extract them
        if os.path.exists(self.embedding_file):
            self.load_image_embeddings()
        else:
            self.extract_and_save_image_embeddings()

    def load_config(self, config_path: str) -> dict:
        """
        Loads YAML configuration file and validates required keys.

        :param config_path: Path to the YAML file.
        :return: Dictionary of configuration parameters.
        """
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Validate required keys
        required_keys = ["frame_folder", "output_file", "image_extensions"]
        for key in required_keys:
            if key not in config:
                raise KeyError(f"Missing required config key: {key}")

        return config

    def save_config(self, config_path: str) -> None:
        """
        Saves the current configuration to a YAML file.

        :param config_path: Path to save the YAML file.
        """
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def collect_image_filenames(self) -> None:
        """
        Collects and sorts image filenames from the frame folder.
        """
        all_files = os.listdir(self.frame_folder)
        self.image_filenames = [
            os.path.join(self.frame_folder, fname)
            for fname in sorted(all_files)
            if fname.endswith(self.image_extensions)
        ]

    def extract_and_save_image_embeddings(self) -> None:
        """
        Extracts features for each image using BLIP2 and saves them to embedding_file.
        """
        self.collect_image_filenames()
        image_embeddings = []

        for image_path in tqdm(
            self.image_filenames, desc="Processing BLIP image embeddings"
        ):
            try:
                raw_image = Image.open(image_path).convert("RGB")
                image = (
                    self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
                )

                with torch.no_grad():
                    # Extract image features in 'image' mode
                    features = self.model.extract_features(
                        {"image": image}, mode="image"
                    )
                    # Project to low-dimensional space
                    image_embed = features.image_embeds_proj  # Shape: [1, 32, 256]
                    # Aggregate embeddings (e.g., average over queries)
                    image_embed = image_embed.mean(dim=1)  # Shape: [1, 256]
                    # Normalize
                    image_embed /= image_embed.norm(dim=-1, keepdim=True)
                    image_embeddings.append(image_embed.cpu())

            except Exception as e:
                print(f"Image processing error ({image_path}): {e}")

        self.image_embeddings = torch.cat(
            image_embeddings, dim=0
        )  # Shape: [num_images, 256]
        torch.save(
            {"filenames": self.image_filenames, "features": self.image_embeddings},
            self.embedding_file,
        )
        print(f"BLIP image features saved to: {self.embedding_file}")

    def load_image_embeddings(self) -> None:
        """
        Loads precomputed image features from the output file.
        """
        data = torch.load(self.embedding_file)
        self.image_filenames = data["filenames"]
        self.image_embeddings = data["features"]
        print(f"BLIP embeddings loaded from {self.embedding_file}")

    def retrieve(self, query: str, top_k: int = 10) -> list:
        """
        Finds the top-k most similar images given a text query.

        :param query: Text query string.
        :param top_k: Number of results to retrieve.
        :return: A list of dictionaries containing rank, image_filename, and similarity score.
        """
        # Encode query text
        text_input = self.txt_processors["eval"](query)
        sample = {"text_input": [text_input]}

        with torch.no_grad():
            # Extract text features in 'text' mode
            features_text = self.model.extract_features(sample, mode="text")
            text_embed = features_text.text_embeds_proj  # Shape: [1, N, 256]

            # Debug: Print shapes
            print(f"text_embed shape: {text_embed.shape}")  # Expected: [1, N, 256]

            # Aggregate multiple text embeddings if necessary
            if text_embed.dim() == 3 and text_embed.size(1) > 1:
                # Example aggregation: mean across the second dimension
                text_embed = text_embed.mean(dim=1)  # Shape: [1, 256]
                print(
                    f"Aggregated text_embed shape: {text_embed.shape}"
                )  # Should be [1, 256]
            elif text_embed.dim() == 2:
                # Already in the correct shape
                pass
            else:
                raise ValueError(
                    f"Unexpected text_embed shape: {text_embed.shape}. "
                    "Expected [1, 256] or [1, N, 256]."
                )

            # Normalize text embedding
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)

            # Compute similarity
            # Using matrix multiplication: [num_images, 256] @ [256, 1] = [num_images, 1]
            similarity = (
                self.image_embeddings.to(self.device) @ text_embed.t()
            ).squeeze(1)

            # Debug: Print similarity shape
            print(f"similarity shape: {similarity.shape}")  # Expected: [num_images]

        # Get top-k indices
        top_indices = similarity.topk(top_k).indices.cpu().numpy()

        # Compile results
        results = []
        for rank, index in enumerate(top_indices, 1):
            results.append(
                {
                    "rank": rank,
                    "image_filename": self.image_filenames[index],
                    "score": float(similarity[index].item()),
                }
            )
        return results


# ================================
# Usage Example (if run directly)
# ================================

if __name__ == "__main__":
    # # Initialize the CLIPRetrieval object
    # clip_config_path = "config/clip_config.yaml"
    # if not os.path.exists(clip_config_path):
    #     raise FileNotFoundError(
    #         f"CLIP configuration file not found: {clip_config_path}"
    #     )
    # clip_retriever = CLIPRetrieval(config_path=clip_config_path)

    # # Retrieve images using CLIP
    # clip_query = "Spider"
    # clip_results = clip_retriever.retrieve(clip_query, top_k=5)

    # print("\n=== CLIPRetrieval Results ===")
    # for res in clip_results:
    #     print(
    #         f"Rank {res['rank']}: Image={res['image_filename']}, Score={res['score']:.4f}"
    #     )
    # print("CLIP Retrieval Results:", clip_results)

    # Initialize the BLIPRetrieval object
    blip_config_path = "config/blip_config.yaml"
    if not os.path.exists(blip_config_path):
        raise FileNotFoundError(
            f"BLIP configuration file not found: {blip_config_path}"
        )
    blip_retriever = BLIPRetrieval(config_path=blip_config_path)

    # Retrieve images using BLIP
    blip_query = "monkey"
    blip_results = blip_retriever.retrieve(blip_query, top_k=5)

    print("\n=== BLIPRetrieval Results ===")
    for res in blip_results:
        print(
            f"Rank {res['rank']}: Image={res['image_filename']}, Score={res['score']:.4f}"
        )
