import os

# -----------------------------------------------------------------------------
# 1. Retrieval 모듈 import
#    - 사용되는 Retrieval 클래스: SCENERetrieval, BLIPRetrieval, CLIPRetrieval, Rankfusion
# -----------------------------------------------------------------------------
from code.video_retrieval import (
    BLIPRetrieval,
    CLIPRetrieval,
    Rankfusion,
    SCENERetrieval,
    SCRIPTRetrieval,
)

import pandas as pd
from tqdm import tqdm


def parse_video_info(image_filename: str):
    """
    image_filename으로부터 video_id, start, end 정보를 파싱하는 헬퍼 함수.
    이미지 파일 이름 예시: "video123_15.0.jpg" -> video_id: "video123", start/end: "15.0"
    """
    base_name = os.path.basename(image_filename)
    name_without_ext, _ = os.path.splitext(base_name)
    video_id, sep, timestamp = name_without_ext.rpartition("_")

    # If there is no underscore, the entire filename (minus extension) is video_id
    if sep == "":
        video_id = name_without_ext
        timestamp = ""

    return video_id, timestamp, timestamp


def retrieve_and_save_results(
    config_file: str,
    benchmark_csv_path: str,
    output_csv_path: str,
    weight_clip: float = 0.4,
    weight_blip: float = 0.5,
    weight_scene: float = 0.1,
    weight_script: float = 0,
    top_k: int = 1000,
    union_top_n: int = 1000,
    desired_num_diverse: int = 10,
    top_n_for_clustering: int = 100,
):
    """
    1) config_file을 이용해 Retrieval 인스턴스를 생성.
    2) benchmark_csv_path를 읽어 쿼리별로 Retrieval을 수행하고,
       diverse ranking 결과를 CSV 파일(output_csv_path)에 저장.
    """

    # -----------------------------
    # 1. 개별 리트리버 및 랭크퓨전 객체 생성
    # -----------------------------
    scene_retriever = SCENERetrieval(config_path=config_file)
    script_retriever = SCRIPTRetrieval(config_path=config_file)
    clip_retriever = CLIPRetrieval(config_path=config_file)
    blip_retriever = BLIPRetrieval(config_path=config_file)

    # Rankfusion(Ensemble) 객체 생성
    ensemble_retriever = Rankfusion(
        scene_retriever=scene_retriever,
        script_retriever=script_retriever,
        clip_retriever=clip_retriever,
        blip_retriever=blip_retriever,
        weight_clip=weight_clip,
        weight_blip=weight_blip,
        weight_scene=weight_scene,
        weight_script=weight_script,
    )

    # ensemble_retriever = ImageRetrievalEnsemble(
    #     clip_retriever=clip_retriever,
    #     blip_retriever=blip_retriever,
    #     weight_clip=weight_clip,
    #     weight_blip=weight_blip,
    # )

    # -----------------------------
    # 2. benchmark CSV 불러오기
    # -----------------------------
    benchmark_df = pd.read_csv(benchmark_csv_path)

    # -----------------------------
    # 3. 쿼리별 Retrieval & 결과 생성
    # -----------------------------
    output_rows = []

    for _, row in tqdm(
        benchmark_df.iterrows(), total=len(benchmark_df), desc="Processing queries"
    ):
        query_index = row["index"]
        original_query = row["query"]

        # (1) Retrieval 수행
        ensemble_results = ensemble_retriever.retrieve(
            original_query, top_k=top_k, union_top_n=union_top_n
        )

        # (2) TopN 결과에서 클러스터링을 통해 diverse 결과 선택
        diverse_results = ensemble_retriever.select_diverse_results_by_clustering(
            ensemble_results,
            desired_num=desired_num_diverse,
            top_n=top_n_for_clustering,
        )

        # (3) CSV 레코드 생성
        for res in diverse_results:
            video_id, start, end = parse_video_info(res["image_filename"])
            output_rows.append(
                {
                    "query_index": query_index,
                    "original_query": original_query,
                    "rank": res["rank"],
                    "video_id": video_id,
                    "start": start,
                    "end": end,
                }
            )

    # -----------------------------
    # 4. 결과 CSV로 저장
    # -----------------------------
    output_df = pd.DataFrame(
        output_rows,
        columns=["query_index", "original_query", "rank", "video_id", "start", "end"],
    )
    output_df.to_csv(output_csv_path, index=False)
    print(f"Retrieval 결과가 '{output_csv_path}'에 저장되었습니다.")


def evaluate_results(
    result_csv_path: str,
    ground_truth_csv_path: str,
    output_score_csv_path: str,
    top_k: int = 5,
):
    """
    Evaluate results by comparing with the ground truth.

    Parameters:
    -----------
    - result_csv_path: Retrieval 결과 CSV 경로
    - ground_truth_csv_path: 정답(benchmark) CSV 경로
    - output_score_csv_path: 스코어링 결과를 저장할 CSV 경로
    - top_k: 상위 K개의 결과에서 정답이 존재하는지 확인
    """

    # -----------------------------
    # 1. CSV 로드
    # -----------------------------
    result_data = pd.read_csv(result_csv_path)
    ground_truth_data = pd.read_csv(ground_truth_csv_path)

    scoring_results = []

    # -----------------------------
    # 2. 쿼리별 정답 매칭 확인
    # -----------------------------
    for _, gt_row in ground_truth_data.iterrows():
        ground_truth_video_id = gt_row["video"]
        ground_truth_start = gt_row["start"]
        ground_truth_end = gt_row["end"]
        gt_index = gt_row["index"]

        # video_id가 "-"로 시작하는 케이스는 제거
        if str(ground_truth_video_id).startswith("-"):
            ground_truth_video_id = str(ground_truth_video_id)[1:]

        # 해당 쿼리(query)에 대해 result_data에서 상위 top_k개 결과만 확인
        query_results = result_data[
            result_data["original_query"] == gt_row["query"]
        ].head(top_k)

        is_correct = 0
        is_video_id_match = 0

        for _, res_row in query_results.iterrows():
            result_video_id = res_row["video_id"]
            result_start = res_row["start"]
            result_end = res_row["end"]

            if str(result_video_id).startswith("-"):
                result_video_id = str(result_video_id)[1:]

            # video_id가 일치하면, 이후 시간 구간 교집합 여부로 정답 여부 체크
            if result_video_id == ground_truth_video_id:
                is_video_id_match = 1
                # (start > end) or (end < start)이면 교집합 없음
                if not (
                    result_end < ground_truth_start or result_start > ground_truth_end
                ):
                    is_correct = 1
                    break

        scoring_results.append(
            {
                "index": gt_index,
                "is_correct": is_correct,
                "is_video_id_match": is_video_id_match,
            }
        )

    # -----------------------------
    # 3. 평가 결과 저장
    # -----------------------------
    scoring_results_df = pd.DataFrame(scoring_results)
    scoring_results_df.to_csv(output_score_csv_path, index=False, encoding="utf-8-sig")
    print(f"Evaluation results saved to: {output_score_csv_path}")

    # -----------------------------
    # 4. 전체 및 Type별 통계 출력
    # -----------------------------
    ground_truth_data = ground_truth_data.drop_duplicates(subset=["index"])
    scoring_results_df = scoring_results_df.drop_duplicates(subset=["index"])
    merged_data = ground_truth_data.merge(scoring_results_df, on="index", how="inner")

    total_samples = len(merged_data)
    video_id_match_count = merged_data["is_video_id_match"].sum()
    correct_count = merged_data["is_correct"].sum()

    video_id_match_ratio = (video_id_match_count / total_samples) * 100
    correct_ratio = (correct_count / total_samples) * 100

    print("\n=== 전체 결과 ===")
    print(f"전체 데이터셋 크기: {total_samples}")
    print(f"Video ID 정답 개수: {video_id_match_count} ({video_id_match_ratio:.2f}%)")
    print(f"정답 개수: {correct_count}/{total_samples} ({correct_ratio:.2f}%)")

    # Type별 통계
    type_stats = (
        merged_data.groupby("type")
        .agg(
            total=("type", "size"),
            video_id_match_count=("is_video_id_match", "sum"),
            correct_count=("is_correct", "sum"),
        )
        .reset_index()
    )

    type_stats["video_id_match_ratio"] = (
        type_stats["video_id_match_count"] / type_stats["total"] * 100
    )
    type_stats["correct_ratio"] = (
        type_stats["correct_count"] / type_stats["total"] * 100
    )

    print("\n=== Type별 결과 ===")
    for _, row in type_stats.iterrows():
        print(f"\nType: {row['type']}")
        print(f"  총 데이터: {row['total']}")
        print(
            f"  Video ID 정답 개수: {row['video_id_match_count']} ({row['video_id_match_ratio']:.2f}%)"
        )
        print(f"  정답 개수: {row['correct_count']} ({row['correct_ratio']:.2f}%)")


def main():
    """
    실제 스크립트 실행 로직을 담은 메인 함수.
    필요 시 경로 및 파라미터를 수정하여 사용하세요.
    """
    # -------------------------------------------------------------------------
    # 1. Retrieval 결과 생성 후 CSV 저장
    # -------------------------------------------------------------------------
    config_file = "config/video_retrieval_config.yaml"
    benchmark_csv_path = "dev/benchmark_en.csv"
    output_csv_path = "dev/retrieval_results.csv"

    retrieve_and_save_results(
        config_file=config_file,
        benchmark_csv_path=benchmark_csv_path,
        output_csv_path=output_csv_path,
        weight_clip=0.4,
        weight_blip=0.6,
        weight_scene=0,
        weight_script=0,
        top_k=1000,
        union_top_n=1000,
        desired_num_diverse=10,
        top_n_for_clustering=100,
    )

    # -------------------------------------------------------------------------
    # 2. 생성된 결과를 기반으로 평가
    # -------------------------------------------------------------------------
    evaluate_results(
        result_csv_path=output_csv_path,
        ground_truth_csv_path=benchmark_csv_path,
        output_score_csv_path="dev/eval.csv",
        top_k=5,
    )


if __name__ == "__main__":
    main()
