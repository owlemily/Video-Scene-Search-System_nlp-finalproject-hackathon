import os

# -----------------------------------------------------------------------------
# 1. RetrievalScoreFusion 인스턴스 생성 (이미 이전에 구현한 클래스를 활용)
# -----------------------------------------------------------------------------
# 예시: config 파일 경로 및 가중치 값은 상황에 맞게 수정하세요.
# from code.image_retrieval_clusturing import (
#     BLIPRetrieval,
#     CLIPRetrieval,
#     RetrievalScoreFusion,
#     select_diverse_results_by_clustering,
# )
from code.video_retrieval import (
    BGERetrieval,
    BLIPRetrieval,
    CLIPRetrieval,
    Rankfusion,
)

import pandas as pd
from tqdm import tqdm

config_file = "config/video_retrieval_config.yaml"

# 개별 리트리버 생성
clip_retriever = CLIPRetrieval(config_path=config_file)
blip_retriever = BLIPRetrieval(config_path=config_file)
scene_retriever = BGERetrieval(config_path=config_file)

# # Rankfusion 객체 생성 (각 retrieval 결과의 가중치는 상황에 따라 조절)
ensemble_retriever = Rankfusion(
    scene_retriever=scene_retriever,
    clip_retriever=clip_retriever,
    blip_retriever=blip_retriever,
    weight_clip=0.4,
    weight_blip=0.6,
    weight_scene=0,
)

# ensemble_retriever = ImageRetrievalEnsemble(
#     clip_retriever, blip_retriever, weight_clip=0.4, weight_blip=0.6
# )


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

for _, row in tqdm(
    benchmark_df.iterrows(), total=len(benchmark_df), desc="Processing queries"
):
    query_index = row["index"]
    original_query = row["query"]

    ensemble_results = ensemble_retriever.retrieve(
        original_query, top_k=1000, union_top_n=1000
    )

    diverse_results = ensemble_retriever.select_diverse_results_by_clustering(
        ensemble_results, desired_num=10, top_n=100
    )

    for res in diverse_results:
        rank = res["rank"]
        image_filename = res["image_filename"]

        base_name = os.path.basename(image_filename)
        name_without_ext, _ = os.path.splitext(base_name)
        video_id, sep, timestamp = name_without_ext.rpartition("_")
        if sep == "":
            video_id = name_without_ext
            timestamp = ""

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
