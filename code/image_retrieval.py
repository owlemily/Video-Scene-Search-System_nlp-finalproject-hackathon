import os

import clip
import torch
import yaml
from PIL import Image


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

        for image_path in self.image_filenames:
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
        print(f"Image features saved to: {self.embedding_file}")

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
# Usage Example (if run directly)
# ================================
if __name__ == "__main__":
    # Initialize the CLIPRetrieval object
    clip_config_path = "config/clip_config.yaml"
    image_retriever = CLIPRetrieval(config_path=clip_config_path)

    # Retrieve images
    image_query = "Spider"
    image_results = image_retriever.retrieve(image_query, top_k=5)

    print("\n=== CLIPRetrieval Results ===")
    for res in image_results:
        print(
            f"Rank {res['rank']}: Image={res['image_filename']}, Score={res['score']:.4f}"
        )
    print("image_results", image_results)
