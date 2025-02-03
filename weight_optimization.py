import os
import re
import pandas as pd
import numpy as np
import csv
from bayes_opt import BayesianOptimization
from rankfusion_retrieval_pipeline import RankFusionSystem  

def extract_video_id_and_time(filename):
    """
    파일명에서 비디오 ID와 시간을 추출 (예: 'C4y_tu3LYlo_63.500.jpg' → ('C4y_tu3LYlo', 63.500))
    """
    match = re.match(r"(.+?)_(\d+\.\d+)\.jpg", filename)
    if match:
        return match.group(1), float(match.group(2))
    return None, None

def evaluate_results(result_csv_path, ground_truth_csv_path, output_score_csv_path, top_k=5):
    """
    Retrieve & re-rank된 결과를 Ground Truth와 비교하여 정확도를 평가하는 함수.
    """
    result_data = pd.read_csv(result_csv_path)
    ground_truth_data = pd.read_csv(ground_truth_csv_path)

    if "filename" not in result_data.columns:
        raise KeyError("CSV 파일에 'filename' 열이 없습니다.")

    scoring_results = []
    for _, row in ground_truth_data.iterrows():
        ground_truth_video_id = row["video"]
        ground_truth_start = row["start"]
        ground_truth_end = row["end"]
        index = row["index"]

        if str(ground_truth_video_id).startswith("-"):
            ground_truth_video_id = str(ground_truth_video_id)[1:]

        matching_results = result_data[result_data["query"] == row["query"]].head(top_k)
        is_correct = 0
        is_video_id_match = 0

        for _, match in matching_results.iterrows():
            result_filename = match["filename"]
            result_video_id, result_time = extract_video_id_and_time(result_filename)

            if str(result_video_id).startswith("-"):
                result_video_id = str(result_video_id)[1:]

            if result_video_id == ground_truth_video_id:
                is_video_id_match = 1
                if ground_truth_start <= result_time <= ground_truth_end:
                    is_correct = 1
                    break

        scoring_results.append({
            "index": index,
            "is_correct": is_correct,
            "is_video_id_match": is_video_id_match,
        })

    scoring_results_df = pd.DataFrame(scoring_results)
    scoring_results_df.to_csv(output_score_csv_path, index=False, encoding="utf-8-sig")
    print(f"Evaluation results saved to: {output_score_csv_path}")

    ground_truth_data = ground_truth_data.drop_duplicates(subset=["index"])
    scoring_results_df = scoring_results_df.drop_duplicates(subset=["index"])
    merged_data = ground_truth_data.merge(scoring_results_df, on="index", how="inner")

    total_samples = len(merged_data)
    video_id_match_count = merged_data["is_video_id_match"].sum()
    correct_count = merged_data["is_correct"].sum()

    video_id_match_ratio = video_id_match_count / total_samples * 100
    correct_ratio = correct_count / total_samples * 100

    print("\n=== 전체 결과 ===")
    print(f"전체 데이터셋 크기: {total_samples}")
    print(f"Video ID 정답 개수: {video_id_match_count} ({video_id_match_ratio:.2f}%)")
    print(f"정답 개수: {correct_count}/{total_samples} ({correct_ratio:.2f}%)")

    return correct_ratio

optimization_results = []

def objective(w_frame, w_scene, w_clip, w_blip):
    """
    주어진 가중치로 RankFusionSystem 실행 후 평가 점수를 반환
    correct_ratio (정확도) 최대화
    """
    weights = np.array([w_frame, w_scene, w_clip, w_blip])
    weights /= weights.sum()  

    rank_fusion_system = RankFusionSystem(
        frame_text_config_path="config/frame_description_config.yaml",
        scene_text_config_path="config/scene_description_config.yaml",
        clip_config_path="config/clip_config.yaml",
        blip_config_path="config/blip_config.yaml",
        w_frame=weights[0],
        w_scene=weights[1],
        w_clip=weights[2],
        w_blip=weights[3],
    )

    result_csv_path = "output/results.csv"
    ground_truth_csv_path = "own_dataset_v2.csv"
    output_score_csv_path = "output/scores.csv"

    ground_truth_data = pd.read_csv(ground_truth_csv_path)
    all_results = []

    for _, row in ground_truth_data.iterrows():
        user_query = row["query"]
        fused_top_k = rank_fusion_system.retrieve_and_fuse(user_query=user_query, top_k=5)

        for item in fused_top_k:
            item["query"] = user_query
            all_results.append(item)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(result_csv_path, index=False, encoding="utf-8-sig")

    correct_ratio = evaluate_results(result_csv_path, ground_truth_csv_path, output_score_csv_path, top_k=5)

    optimization_results.append([weights[0], weights[1], weights[2], weights[3], correct_ratio])

    print(f"가중치: {weights}, 평가 점수 (correct_ratio): {correct_ratio:.2f}%\n")

    return correct_ratio  

pbounds = {"w_frame": (0, 1), "w_scene": (0, 1), "w_clip": (0, 1), "w_blip": (0, 1)}

optimizer = BayesianOptimization(
    f=objective,  
    pbounds=pbounds,
    random_state=42,
)

optimizer.maximize(init_points=5, n_iter=10)

best_weights = optimizer.max["params"]
best_weights = np.array([best_weights["w_frame"], best_weights["w_scene"], best_weights["w_clip"], best_weights["w_blip"]])
best_weights /= best_weights.sum()  

print(f"\n=== 최적 가중치 (top_k=5 기준) ===")
print(f"Frame: {best_weights[0]:.3f}")
print(f"Scene: {best_weights[1]:.3f}")
print(f"Clip: {best_weights[2]:.3f}")
print(f"Blip: {best_weights[3]:.3f}")
print(f"최적 점수 (correct_ratio): {optimizer.max['target']:.2f}%")

def save_optimization_results(output_csv_path="output/weights_results.csv"):
    with open(output_csv_path, mode="w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["w_frame", "w_scene", "w_clip", "w_blip", "correct_ratio"]) 
        writer.writerows(optimization_results)

    print(f"가중치 최적화 결과 저장 완료: {output_csv_path}")

save_optimization_results()
