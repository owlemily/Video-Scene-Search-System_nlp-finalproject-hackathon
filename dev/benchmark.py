import csv
import os

from rankfusion_retrieval_pipeline import RankFusionSystem


def run_rank_fusion_for_csv(
    csv_input_path="benchmark_en.csv",
    csv_output_path="benchmark_fused.csv",
    frame_config_path="config/frame_description_config.yaml",
    scene_config_path="config/scene_description_config.yaml",
    clip_config_path="config/clip_config.yaml",
    top_k=5,
    w_frame=0.3,
    w_scene=0.4,
    w_clip=0.3,
):
    """
    CSV에서 한 줄씩 (query) 꺼내서 rank_fusion을 수행한 뒤,
    결과를 새로운 CSV로 저장합니다.
    """
    # -- (2) RankFusionSystem 객체 생성 --
    rank_fusion_system = RankFusionSystem(
        frame_text_config_path=frame_config_path,
        scene_text_config_path=scene_config_path,
        clip_config_path=clip_config_path,
        w_frame=w_frame,
        w_scene=w_scene,
        w_clip=w_clip,
    )

    # -- (3) 결과 저장할 CSV 준비 --
    with (
        open(csv_input_path, "r", encoding="utf-8") as in_f,
        open(csv_output_path, "w", newline="", encoding="utf-8") as out_f,
    ):
        reader = csv.DictReader(in_f)
        fieldnames = [
            "query_index",
            "original_query",
            "rank",
            "video_id",
            "start",
            "end",
        ]
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            query_idx = row["index"]  # benchmark_en.csv 상의 인덱스
            original_query = row["query"]  # benchmark_en.csv 상의 쿼리

            # -- (4) Retrieve & Rank Fusion --
            fused_top_k = rank_fusion_system.retrieve_and_fuse(
                original_query, top_k=top_k
            )

            # 결과가 없으면 넘어감
            if not fused_top_k:
                continue

            # -- (5) 상위 top_k개 결과 저장 --
            for rank, top_item in enumerate(fused_top_k, start=1):
                fused_filename = top_item["filename"]  # 예: "5qlG1ODkRWw_135.750.jpg"

                # 확장자 제거 -> "tBDHJCVi7_0_43.250"
                file_without_ext = os.path.splitext(fused_filename)[0]
                # '_'를 기준으로 전체 분리
                parts = file_without_ext.split("_")

                # 마지막 값이 time_str, 나머지를 video_id로 병합
                time_str = parts[-1]
                video_id = "_".join(parts[:-1])

                # time_str이 "135.750" 형태이면, start/end가 동일
                start_time = time_str
                end_time = time_str

                # -- (6) 새로운 CSV에 기록 --
                out_row = {
                    "query_index": query_idx,
                    "original_query": original_query,
                    "rank": rank,
                    "video_id": video_id,
                    "start": start_time,
                    "end": end_time,
                }
                writer.writerow(out_row)

    print(f"[INFO] Rank fusion results saved to: {csv_output_path}")


if __name__ == "__main__":
    # -- (7) 실제 함수 실행 --
    run_rank_fusion_for_csv(
        csv_input_path="dev/benchmark_en.csv",
        csv_output_path="/data/ephemeral/home/level4-nlp-finalproject-hackathon-nlp-01-lv3/benchmark_fused.csv",
        top_k=1,
        w_frame=1,
        w_scene=2,
        w_clip=4,
    )
