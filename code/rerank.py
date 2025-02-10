import os
import torch
import numpy as np
from PIL import Image
from qwen_vl_utils import process_vision_info
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)
from wordfreq import zipf_frequency

###########################################################
# 전역 설정 및 캐시(모델 로드 한 번만 수행)
###########################################################

# Device 설정 (cuda 사용 가능 시)
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_embedding_model():
    """SentenceTransformer 임베딩 모델을 캐시하여 반환"""
    embedding_model = SentenceTransformer("all-mpnet-base-v2").to(device)
    return embedding_model

def get_qwen_models():
    """Qwen-VL 모델, 토크나이저, 프로세서를 캐시하여 반환"""
    qwen_model, qwen_tokenizer, qwen_processor = load_qwen_model()
    return qwen_model, qwen_tokenizer, qwen_processor

###########################################################
# 유틸리티 함수들
###########################################################

def translate_to_english(query):
    """
    번역 함수 (현재는 단순 placeholder)
    만약 실제 번역이 필요하면, 번역 API나 라이브러리를 이용하세요.
    """
    return query

def extract_keywords(query, top_k=5):
    """
    TF-IDF 기반으로 입력 문장에서 상위 top_k개의 키워드를 추출합니다.
    """
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform([query])
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(X.toarray()).flatten()[::-1]
    keywords = feature_array[tfidf_sorting][:top_k]
    return set(keywords)

def calculate_keyword_uniqueness(keyword):
    """
    단어의 희귀성을 계산합니다.
    wordfreq의 zipf_frequency를 사용하며, 로그 스케일로 변환합니다.
    """
    freq = zipf_frequency(keyword, "en") + 1e-5
    return np.log(1 + 1/freq)

def load_qwen_model():
    """
    Qwen2.5-VL-7B-Instruct 모델과 관련 토크나이저, 프로세서를 로드합니다.
    """
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer, processor

def generate_image_descriptions(image_paths, qwen_processor, qwen_model):
    """
    이미지 목록(파일 경로)을 받아 배치 단위로 Qwen-VL 모델에 이미지 설명 생성을 요청합니다.
    """
    batch_size = 5
    descriptions = []

    for i in range(0, len(image_paths), batch_size):
        # 이미지 로드
        batch_images = [Image.open(img_path).convert("RGB") for img_path in image_paths[i:i+batch_size]]

        # 각 이미지에 대해 Qwen-VL 모델에 "Describe this image." 메시지 전송
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
            for image in batch_images
        ]
        # 채팅 템플릿 적용
        texts = [
            qwen_processor.apply_chat_template(
                [msg], tokenize=False, add_generation_prompt=True
            )
            for msg in messages
        ]

        # 이미지 및 기타 입력 처리
        image_inputs, _ = process_vision_info(messages)
        inputs = qwen_processor(
            text=texts, images=image_inputs, return_tensors="pt", padding=True
        ).to(device)

        # 모델로부터 설명 생성
        with torch.no_grad():
            generated_ids = qwen_model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            responses = qwen_processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

        descriptions.extend([resp.strip() for resp in responses])

    return descriptions

def rerank_with_qwen(query, candidate_image_paths, qwen_processor, qwen_model, embedding_model):
    """
    Advanced 검색 시, 후보 이미지들에 대해 Qwen-VL 모델로 생성한 이미지 설명을
    바탕으로 입력 query의 키워드와의 유사도 및 희귀도(uniqueness)를 계산하여 최적의 이미지를 선택합니다.
    """
    descriptions = generate_image_descriptions(candidate_image_paths, qwen_processor, qwen_model)

    # 입력 query에서 키워드 추출 및 희귀성(uniqueness) 계산
    keywords = extract_keywords(query)
    uniqueness_scores = {
        keyword: calculate_keyword_uniqueness(keyword) for keyword in keywords
    }

    similarity_threshold = 0.5
    keyword_scores = []

    for desc in descriptions:
        desc_words = set(desc.lower().split())
        score = 0

        # 각 키워드와 설명의 단어 간 코사인 유사도를 계산하여 threshold 이상이면 점수 반영
        for keyword in keywords:
            for desc_word in desc_words:
                embeddings = embedding_model.encode(
                    [keyword, desc_word], convert_to_tensor=True
                )
                cosine_sim = cosine_similarity(
                    embeddings[0].cpu().numpy().reshape(1, -1),
                    embeddings[1].cpu().numpy().reshape(1, -1),
                )[0][0]
                if cosine_sim >= similarity_threshold:
                    score += uniqueness_scores.get(keyword, 0)
                    break  # 하나의 키워드는 한 번만 반영
        average_score = score / len(keywords) if keywords else 0
        keyword_scores.append(average_score)

    # 첫 번째 후보 이미지 점수와 비교하여 의미 있는 차이가 있으면 재정렬
    rank1_score = keyword_scores[0] if keyword_scores else 0
    best_match_idx = max(range(len(keyword_scores)), key=lambda i: keyword_scores[i]) if keyword_scores else 0
    score_difference = keyword_scores[best_match_idx] - rank1_score if keyword_scores else 0
    print(f"score_difference: {score_difference:.4f}")
    threshold = 0.08  # 임계값

    if score_difference > threshold:
        if best_match_idx != 0:
            print(f"Re-ranked: Image {best_match_idx + 1} (Score Difference: {score_difference:.4f})")
        return candidate_image_paths[best_match_idx], f"Re-ranked: Image {best_match_idx + 1} (Score Difference: {score_difference:.4f})"
    else:
        return candidate_image_paths[0], "Retained initial Rank 1 image"

###########################################################
# 메인 실행 흐름: Advanced 체크에 따른 검색 및 리랭킹
###########################################################

if __name__ == "__main__":
    # advanced 플래그가 체크되어 있으면 advanced 검색 모드로 동작
    advanced = True  # Advanced 모드: True, 기본 검색: False

    # 예시 검색 query (실제 사용 시 입력받거나 API 등으로 처리)
    query = "A scenic mountain landscape with a clear blue sky"

    # temp_save_folder에 저장된 이미지 불러오기 (첫 검색 결과가 저장되어 있다고 가정)
    image_dir = "./temp_save_folder"
    if not os.path.exists(image_dir):
        print(f"Image directory '{image_dir}' does not exist.")
        exit(1)

    image_filenames = [
        fname for fname in os.listdir(image_dir)
        if fname.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    image_filenames.sort()  # 순서를 보장
    # 여기서는 모든 이미지를 후보로 사용 (필요에 따라 초기 검색 결과의 인덱스를 선택할 수 있음)
    candidate_image_paths = [os.path.join(image_dir, fname) for fname in image_filenames]

    # 모델들(임베딩, Qwen-VL) 캐시 로드 (이미 로드되었으면 재사용)
    get_embedding_model()
    qwen_model, qwen_tokenizer, qwen_processor = get_qwen_models()

    if advanced:
        best_image_path, message = rerank_with_qwen(query, candidate_image_paths)
        print("Advanced search result:")
        print(best_image_path)
        print(message)
    else:
        # 기본 검색 모드: 후보 중 첫 번째 이미지를 사용
        best_image_path = candidate_image_paths[0] if candidate_image_paths else None
        print("Basic search result:")
        print(best_image_path)
