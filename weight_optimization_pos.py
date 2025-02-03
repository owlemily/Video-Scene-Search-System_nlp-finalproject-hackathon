import os
import re
import pandas as pd
import numpy as np
import csv
import spacy
from bayes_opt import BayesianOptimization
from rankfusion_retrieval_pipeline import RankFusionSystem  

# spaCy 영어 모델 로드
nlp = spacy.load("en_core_web_sm")

def extract_video_id_and_time(filename):
    """
    파일명에서 비디오 ID와 시간을 추출 (예: 'C4y_tu3LYlo_63.500.jpg' → ('C4y_tu3LYlo', 63.500))
    """
    match = re.match(r"(.+?)_(\d+\.\d+)\.jpg", filename)
    if match:
        return match.group(1), float(match.group(2))
    return None, None

def process_query(query):
    """
    Query를 품사 태깅 후 두 개의 Query 생성 (가중치 적용)
    """
    doc = nlp(query)
    
    query_nadj_verb = []
    query_verb_nadj = []
    
    for token in doc:
        query_nadj_verb.append(token.text)
        query_verb_nadj.append(token.text)
    
    return " ".join(query_nadj_verb), " ".join(query_verb_nadj)

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

    correct_ratio = scoring_results_df["is_correct"].sum() / len(scoring_results_df) * 100
    print(f"정확도: {correct_ratio:.2f}%")
    return correct_ratio

optimization_results = []

def objective(w_scene, w_clip, w_blip):
    """
    주어진 가중치로 RankFusionSystem 실행 후 평가 점수를 반환
    """
    rank_fusion_system = RankFusionSystem(
        scene_text_config_path="config/scene_description_config.yaml",
        clip_config_path="config/clip_config.yaml",
        blip_config_path="config/blip_config.yaml",
        w_scene=w_scene,
        w_clip=w_clip,
        w_blip=w_blip,
    )

    result_csv_path = "output/results3.csv"
    ground_truth_csv_path = "own_dataset_v2.csv"
    output_score_csv_path = "output/scores3.csv"

    ground_truth_data = pd.read_csv(ground_truth_csv_path)
    all_results = []

    print(f"=== Running with Weights: Scene={w_scene}, Clip={w_clip}, Blip={w_blip} ===")
    for _, row in ground_truth_data.iterrows():
        user_query = row["query"]
        query_nadj_verb, query_verb_nadj = process_query(user_query)
        
        fused_top_k_1 = rank_fusion_system.retrieve_and_fuse(query_nadj_verb, top_k=5)
        fused_top_k_2 = rank_fusion_system.retrieve_and_fuse(query_verb_nadj, top_k=5)

        for item in fused_top_k_1 + fused_top_k_2:
            item["query"] = user_query
            all_results.append(item)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(result_csv_path, index=False, encoding="utf-8-sig")
    correct_ratio = evaluate_results(result_csv_path, ground_truth_csv_path, output_score_csv_path, top_k=5)
    
    optimization_results.append([w_scene, w_clip, w_blip, correct_ratio])
    print(f"=== Results: Correct Ratio: {correct_ratio:.2f}% ===")
    return correct_ratio  

pbounds = {"w_scene": (0, 1), "w_clip": (0, 1), "w_blip": (0, 1)}
optimizer = BayesianOptimization(f=objective, pbounds=pbounds, random_state=42)
optimizer.maximize(init_points=5, n_iter=10)

def save_optimization_results(output_csv_path="output/weights_results3.csv"):
    with open(output_csv_path, mode="w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["w_scene", "w_clip", "w_blip", "correct_ratio"])
        writer.writerows(optimization_results)
    print(f"가중치 최적화 결과 저장 완료: {output_csv_path}")

save_optimization_results()