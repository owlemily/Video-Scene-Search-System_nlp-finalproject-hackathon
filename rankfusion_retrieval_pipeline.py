import json
import os

import clip
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

clip_device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=clip_device)

bge_model_name = "BAAI/bge-large-en"
bge_tokenizer = AutoTokenizer.from_pretrained(bge_model_name)
bge_model = AutoModel.from_pretrained(bge_model_name).to(clip_device)
image_vector_file = "./image_vec/frame_vectors.pt"
image_data = torch.load(image_vector_file)
image_filenames = image_data["filenames"]
image_features = image_data["features"]

frame_json_file = "./description/5qlG1ODkRWw_frame.json"
clip_json_file = "./description/5qlG1ODkRWw_clip.json"

with open(frame_json_file, "r", encoding="utf-8") as f:
    frame_descriptions = json.load(f)

with open(clip_json_file, "r", encoding="utf-8") as f:
    clip_descriptions = json.load(f)


def encode_texts(texts, model, tokenizer, device="cpu"):
    inputs = tokenizer(
        texts, padding=True, truncation=True, return_tensors="pt", max_length=512
    )
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    return embeddings


frame_texts = [desc["description"] for desc in frame_descriptions]
clip_texts = [desc["description"] for desc in clip_descriptions]

frame_vectors = encode_texts(frame_texts, bge_model, bge_tokenizer, clip_device)
clip_vectors = encode_texts(clip_texts, bge_model, bge_tokenizer, clip_device)
clip_ranges = [
    (desc["start"], desc["end"], i) for i, desc in enumerate(clip_descriptions)
]
user_query = "Reindeer looking at a fallen person in the winter forest"
bge_query_vector = encode_texts([user_query], bge_model, bge_tokenizer, clip_device)


def compute_similarity(query, features):
    query = query.to(features.device).to(torch.float32)
    features = features.to(torch.float32)
    return torch.matmul(features, query.T).squeeze(1).cpu().numpy()


clip_query_tokens = clip.tokenize([user_query]).to(clip_device)
with torch.no_grad():
    clip_query_vector = clip_model.encode_text(clip_query_tokens)
    clip_query_vector /= clip_query_vector.norm(dim=-1, keepdim=True)

image_similarity = compute_similarity(clip_query_vector, image_features)
frame_similarity = compute_similarity(bge_query_vector, frame_vectors)
clip_similarity = compute_similarity(bge_query_vector, clip_vectors)
weight_image = 0.3
weight_frame = 0.3
weight_clip = 0.3

final_scores = np.zeros(len(frame_descriptions))

for i, frame in enumerate(frame_descriptions):
    frame_time = frame["start"]
    clip_score = 0
    for start, end, clip_idx in clip_ranges:
        if start <= frame_time <= end:
            clip_score = clip_similarity[clip_idx]
            break

    final_scores[i] = (
        weight_image * image_similarity[i]
        + weight_frame * frame_similarity[i]
        + weight_clip * clip_score
    )
top_n = 5
top_indices = final_scores.argsort()[-top_n:][::-1]

base_path = "./image_vec/frames"

print("\n가장 유사한 프레임들:")
for rank, idx in enumerate(top_indices, 1):
    frame_info = frame_descriptions[idx]
    frame_time = frame_info["start"]
    video_id = frame_info["video_id"]

    clip_description = next(
        (
            desc["description"]
            for desc in clip_descriptions
            if desc["start"] <= frame_time <= desc["end"]
        ),
        "No clip description found.",
    )

    image_filename = f"{video_id}_{frame_time:.3f}.jpg"
    image_path = os.path.join(base_path, image_filename)

    print(f"Rank {rank}: {image_filename} (Score: {final_scores[idx]:.4f})")
    print(f"Frame Description: {frame_info['description']}")
    print(f"Clip Description: {clip_description}")

    try:
        image = Image.open(image_path)
        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.title(f"Rank {rank}: {image_filename}")
        plt.axis("off")
        plt.show()
    except FileNotFoundError:
        print(f"이미지 파일을 찾을 수 없습니다: {image_path}")
