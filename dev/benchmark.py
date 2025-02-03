import os

# -----------------------------------------------------------------------------
# 1. RetrievalScoreFusion 인스턴스 생성 (이미 이전에 구현한 클래스를 활용)
# -----------------------------------------------------------------------------
# 예시: config 파일 경로 및 가중치 값은 상황에 맞게 수정하세요.
from code.image_retrieval_clusturing import (
    BLIPRetrieval,
    CLIPRetrieval,
    RetrievalScoreFusion,
    select_diverse_results_by_clustering,
)

import pandas as pd

config_file = "config/video_retrieval_config.yaml"

# 개별 리트리버 생성
clip_retriever = CLIPRetrieval(config_path=config_file)
blip_retriever = BLIPRetrieval(config_path=config_file)
# scene_retriever = BGERetrieval(config_path=config_file)

# Rankfusion 객체 생성 (각 retrieval 결과의 가중치는 상황에 따라 조절)
# rankfusion = Rankfusion(
#     scene_retriever=scene_retriever,
#     clip_retriever=clip_retriever,
#     blip_retriever=blip_retriever,
#     weight_clip=0.4,
#     weight_blip=0.6,
#     weight_scene=0,
# )
ensemble_retriever = RetrievalScoreFusion(
    clip_retriever,
    blip_retriever,
    weight_clip=0.4,
    weight_blip=0.6,
)

# -----------------------------------------------------------------------------
# 2. benchmark_en.csv 파일 읽기
#    CSV 파일 컬럼: index,query,type,video,start,end
# -----------------------------------------------------------------------------
benchmark_df = pd.read_csv("dev/benchmark_en.csv")

# -----------------------------------------------------------------------------
# 3. 각 benchmark 쿼리에 대해 RetrievalScoreFusion 수행 및 결과 CSV 데이터 생성
#    최종 CSV 컬럼: "query_index", "original_query", "rank", "video_id", "start", "end"
# -----------------------------------------------------------------------------
output_rows = []
top_k = 5  # retrieval 결과 상위 몇 개를 평가에 사용할지 (필요에 따라 조정)

for _, row in benchmark_df.iterrows():
    query_index = row["index"]
    original_query = row["query"]

    # # 최종 Rankfusion 결과 (예: 상위 10개 결과)
    # fusion_results = rankfusion.retrieve(original_query, top_k=1000)

    # diverse_results = rankfusion.select_diverse_results_by_clustering(
    #     fusion_results, desired_num=10, top_n=100
    # )

    ensemble_results = ensemble_retriever.retrieve(original_query, top_k=1000)

    # diversity를 위해 CLIP의 이미지 임베딩을 활용하여 이미지 파일명 -> 임베딩 dict 생성
    clip_embedding_dict = {}
    # clip_retriever.image_embeddings는 torch.Tensor이므로 numpy 배열로 변환
    for fname, emb in zip(
        clip_retriever.image_filenames, clip_retriever.image_embeddings
    ):
        clip_embedding_dict[fname] = emb.cpu().numpy()

    # 클러스터링 기반 다양성 선택: 최종 top 5 결과 추출
    diverse_results = select_diverse_results_by_clustering(
        ensemble_results, clip_embedding_dict, desired_num=10, top_n=100
    )

    for res in diverse_results:
        rank = res["rank"]
        image_filename = res["image_filename"]

        # image_filename 예시:
        # "/data/ephemeral/home/junhan_blip2_streamlit/full_frames/UdZuHy_tbw_99.76633333333332.jpg"
        # 마지막 언더스코어를 기준으로 분리하여 video_id와 timestamp를 추출합니다.
        base_name = os.path.basename(
            image_filename
        )  # 예: "UdZuHy_tbw_99.76633333333332.jpg"
        name_without_ext, _ = os.path.splitext(
            base_name
        )  # "UdZuHy_tbw_99.76633333333332"
        # rpartition를 사용하면 마지막 언더스코어를 기준으로 세 부분으로 나눕니다.
        video_id, sep, timestamp = name_without_ext.rpartition("_")
        if sep == "":
            # underscore가 없을 경우 전체를 video_id로, timestamp는 빈 문자열로 처리
            video_id = name_without_ext
            timestamp = ""

        # retrieval 결과는 프레임 기반이므로, start와 end 모두 timestamp 값을 사용합니다.
        output_rows.append(
            {
                "query_index": query_index,
                "original_query": original_query,
                "rank": rank,
                "video_id": video_id,
                "start": timestamp,
                "end": timestamp,
            }
        )

# -----------------------------------------------------------------------------
# 4. 결과를 DataFrame으로 저장 후 CSV 파일로 출력
# -----------------------------------------------------------------------------
output_df = pd.DataFrame(
    output_rows,
    columns=["query_index", "original_query", "rank", "video_id", "start", "end"],
)
output_csv_path = "/data/ephemeral/home/level4-nlp-finalproject-hackathon-nlp-01-lv3/retrieval_results.csv"
output_df.to_csv(output_csv_path, index=False)

print(f"Retrieval 평가 CSV 파일이 {output_csv_path}에 저장되었습니다.")
