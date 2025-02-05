import os
import re
import csv
import yaml
import pandas as pd
import numpy as np
from bayes_opt import BayesianOptimization
from code.video_retrieval import BGERetrieval, CLIPRetrieval, BLIPRetrieval, Rankfusion

with open("config/optimization_config.yaml", "r", encoding="utf-8") as file:
    CONFIG = yaml.safe_load(file)

scene_retriever = BGERetrieval(config_path="config/video_retrieval_config.yaml")
clip_retriever = CLIPRetrieval(config_path="config/video_retrieval_config.yaml")
blip_retriever = BLIPRetrieval(config_path="config/video_retrieval_config.yaml")


def extract_video_id_and_time(filename):
    """
    파일명에서 비디오 ID와 시간을 추출 (디렉토리 경로 제외)
    """
    filename = os.path.basename(filename)
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

    if "image_filename" not in result_data.columns:
        raise KeyError("CSV 파일에 'image_filename' 열이 없습니다.")

    scoring_results = []
    for _, row in ground_truth_data.iterrows():
        ground_truth_video_id = row["video"]
        ground_truth_start = row["start"]
        ground_truth_end = row["end"]
        index = row["index"]

        matching_results = result_data[result_data["query"] == row["query"]].head(top_k)
        is_correct = 0
        is_video_id_match = 0

        for _, match in matching_results.iterrows():
            result_filename = match["image_filename"]
            result_video_id, result_time = extract_video_id_and_time(result_filename)

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

    correct_ratio = scoring_results_df["is_correct"].sum() / len(scoring_results_df) * 100
    print(f"정확도: {correct_ratio:.2f}%")
    return correct_ratio


optimization_results = []


def objective(w_scene, w_clip, w_blip):
    """
    가중치 최적화 수행
    """
    rankfusion = Rankfusion(
        scene_retriever=scene_retriever,
        clip_retriever=clip_retriever,
        blip_retriever=blip_retriever,
        weight_scene=w_scene,
        weight_clip=w_clip,
        weight_blip=w_blip,
    )

    result_csv_path = CONFIG["paths"]["result_csv"]
    ground_truth_csv_path = CONFIG["paths"]["ground_truth_csv"]
    output_score_csv_path = CONFIG["paths"]["output_score_csv"]

    ground_truth_data = pd.read_csv(ground_truth_csv_path)
    all_results = []

    print(f"=== Running with Weights: Scene={w_scene}, Clip={w_clip}, Blip={w_blip} ===")

    for _, row in ground_truth_data.iterrows():
        user_query = row["query"]

        fused_results = rankfusion.retrieve(user_query, top_k=100)
        diverse_results = rankfusion.select_diverse_results_by_clustering(
            retrieval_results=fused_results, desired_num=5, top_n=100
        )

        for item in diverse_results:
            item["query"] = user_query
            all_results.append(item)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(result_csv_path, index=False, encoding="utf-8-sig")
    correct_ratio = evaluate_results(result_csv_path, ground_truth_csv_path, output_score_csv_path, top_k=5)

    optimization_results.append([w_scene, w_clip, w_blip, correct_ratio])
    print(f"=== Results: Correct Ratio: {correct_ratio:.2f}% ===")
    return correct_ratio


pbounds = {"w_scene": (0, 1), "w_clip": (0, 1), "w_blip": (0, 1)}

init_points = CONFIG["bayesian_optimization"]["init_points"]
n_iter = CONFIG["bayesian_optimization"]["n_iter"]

initial_weights = CONFIG["weights"]

optimizer = BayesianOptimization(f=objective, pbounds=pbounds, random_state=42)

optimizer.probe(
    params={"w_scene": initial_weights["w_scene"], 
            "w_clip": initial_weights["w_clip"], 
            "w_blip": initial_weights["w_blip"]},
    lazy=True
)

optimizer.maximize(init_points=max(init_points - 1, 0), n_iter=n_iter)


def save_optimization_results():
    output_csv_path = CONFIG["paths"]["weights_output_csv"]
    with open(output_csv_path, mode="w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["w_scene", "w_clip", "w_blip", "correct_ratio"])
        writer.writerows(optimization_results)
    print(f"가중치 최적화 결과 저장 완료: {output_csv_path}")


save_optimization_results()