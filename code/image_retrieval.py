import os

import clip
import torch
import yaml
from PIL import Image


class ImageRetrieval:
    def __init__(self, config_path="config.yaml"):
        # Load configuration
        self.config = self.load_config(config_path)

        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load CLIP model and preprocess
        self.model, self.preprocess = clip.load(
            self.config["model_name"], device=self.device
        )

        # Set paths
        self.frame_folder = self.config["frame_folder"]
        self.output_file = self.config["output_file"]

        # Initialize attributes
        self.image_filenames = []
        self.image_features = None

        # Ensure the output directory exists
        output_dir = os.path.dirname(self.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def load_config(self, config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def save_config(self, config_path):
        with open(config_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def collect_image_filenames(self):
        """Collects and sorts image filenames from the frame folder."""
        self.image_filenames = [
            os.path.join(self.frame_folder, fname)
            for fname in sorted(os.listdir(self.frame_folder))
            if fname.endswith(tuple(self.config["image_extensions"]))
        ]

    def extract_image_features(self):
        """Extracts image features using the CLIP model and saves them."""
        self.collect_image_filenames()
        image_features = []

        for image_path in self.image_filenames:
            try:
                image = (
                    self.preprocess(Image.open(image_path).convert("RGB"))
                    .unsqueeze(0)
                    .to(self.device)
                )
                with torch.no_grad():
                    feature = self.model.encode_image(image)
                    feature /= feature.norm(dim=-1, keepdim=True)
                    image_features.append(feature.cpu())
            except Exception as e:
                print(f"Image processing error: {image_path}, Error: {e}")

        self.image_features = torch.cat(image_features, dim=0)
        torch.save(
            {"filenames": self.image_filenames, "features": self.image_features},
            self.output_file,
        )
        print(f"Image features saved to {self.output_file}")

    def load_image_features(self):
        """Loads precomputed image features from the output file."""
        data = torch.load(self.output_file, weights_only=True)
        self.image_filenames = data["filenames"]
        self.image_features = data["features"]

    def find_similar_images(self, query, top_k=10):
        """Finds and prints the top-k most similar images for a given query."""
        # Tokenize and encode the query
        text_tokens = clip.tokenize([query]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        # Calculate similarity (ensure tensors are on the same device)
        similarity = (
            100.0 * self.image_features.to(self.device) @ text_features.T
        ).squeeze(1)

        # Get top-k indices
        top_indices = similarity.topk(top_k).indices.cpu().numpy()

        # Print results
        print(f"\nQuery: '{query}'")
        for rank, index in enumerate(top_indices, 1):
            print(f"Rank {rank}: {self.image_filenames[index]}")


# Usage Example
if __name__ == "__main__":
    config_path = "config/clip_config.yaml"
    retrieval = ImageRetrieval(config_path)
    retrieval.extract_image_features()  # To preprocess and save image features
    retrieval.load_image_features()  # To load precomputed features
    retrieval.find_similar_images("Spider", top_k=5)  # Query example
