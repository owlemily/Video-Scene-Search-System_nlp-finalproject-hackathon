import ntpath

# 두 모듈에서 각 클래스 불러오기
from code.basic_retrieval import BGERetrieval
from code.image_retrieval import CLIPRetrieval


def fuse_results(
    frame_results,  # BGERetrieval(frame) topK
    scene_results,  # BGERetrieval(scene) topK
    clip_results,  # CLIPRetrieval topK
    w_frame=0.3,
    w_scene=0.4,
    w_clip=0.3,
    top_k=10,
):
    """
    전체 프레임 메타정보가 없는 상황에서,
    frame_results + scene_results + clip_results를 합쳐
    최종 점수를 계산하는 간단한 방법.

    - 후보 프레임: (frame_results의 filename) ∪ (clip_results의 filename)
    - frame_score: frame_results에 있으면 점수, 없으면 0
    - scene_score: frame_results 통해 얻은 scene_id가 scene_results topK 중 있으면 점수, 없으면 0
    - clip_score: clip_results에 있으면 점수, 없으면 0

    final_score = w_frame * frame_score + w_scene * scene_score + w_clip * clip_score
    """

    # -----------------------------
    # (A) scene_id -> scene_score (scene_results 맵)
    # -----------------------------
    scene_score_map = {}
    for s_item in scene_results:
        sid = s_item.get("scene_id")
        if sid is not None:
            scene_score_map[sid] = s_item.get("score", 0.0)

    # -----------------------------
    # (B) frame_results (filename -> frame_score, scene_id)
    # -----------------------------
    frame_score_map = {}
    frame_scene_map = {}  # filename -> scene_id

    frame_filenames = set()

    for f_item in frame_results:
        raw_path = f_item.get("frame_image_path", "")
        if raw_path.startswith("./"):
            raw_path = raw_path[2:]
        filename_only = ntpath.basename(raw_path)

        frame_filenames.add(filename_only)
        frame_score_map[filename_only] = f_item.get("score", 0.0)
        frame_scene_map[filename_only] = f_item.get("scene_id", None)

    # -----------------------------
    # (C) clip_results (filename -> clip_score)
    # -----------------------------
    clip_score_map = {}
    clip_filenames = set()

    for c_item in clip_results:
        raw_path = c_item.get("image_filename", "")
        if raw_path.startswith("./"):
            raw_path = raw_path[2:]
        filename_only = ntpath.basename(raw_path)

        clip_filenames.add(filename_only)
        clip_score_map[filename_only] = c_item.get("score", 0.0)

    # -----------------------------
    # (D) 최종 후보 = frame_filenames ∪ clip_filenames
    # -----------------------------
    candidate_filenames = frame_filenames.union(clip_filenames)

    # -----------------------------
    # (E) 후보 각각에 대해 점수 계산
    # -----------------------------
    fused_list = []
    for fname in candidate_filenames:
        # frame_score
        frame_s = frame_score_map.get(fname, 0.0)

        # scene_score
        #  - scene_id를 frame_results에서만 알 수 있다고 가정
        scene_s = 0.0
        sid = frame_scene_map.get(fname, None)
        if sid is not None and sid in scene_score_map:
            scene_s = scene_score_map[sid]

        # clip_score
        clip_s = clip_score_map.get(fname, 0.0)

        # 최종 점수
        final_s = w_frame * frame_s + w_scene * scene_s + w_clip * clip_s

        fused_list.append(
            {
                "filename": fname,
                "final_score": final_s,
                "frame_score": frame_s,
                "scene_score": scene_s,
                "clip_score": clip_s,
                "scene_id": sid,  # scene_id가 None일 수도 있음
            }
        )

    # -----------------------------
    # (F) final_score 내림차순 정렬 후 상위 top_k
    # -----------------------------
    fused_list.sort(key=lambda x: x["final_score"], reverse=True)
    return fused_list[:top_k]


if __name__ == "__main__":
    """
    예시 실행:
    - 1) BGE Retrieval (프레임 설명 기반) 객체 생성 후, 사용자 쿼리에 대해 top_k 추출
    - 2) BGE Retrieval (씬 설명 기반) 객체 생성 후, 동일 쿼리에 대해 top_k 추출
    - 3) CLIP Retrieval (이미지) 객체 생성 후, 동일 쿼리에 대해 top_k 추출
    - 4) fuse_results 함수로 세 결과를 랭크퓨전 (w_frame, w_scene, w_clip 가중치 조절)
    """

    # 1) BGE Retrieval (frame)
    frame_text_config_path = "config/frame_description_config.yaml"
    frame_text_retriever = BGERetrieval(config_path=frame_text_config_path)

    # 2) BGE Retrieval (scene)
    scene_text_config_path = "config/scene_description_config.yaml"
    scene_text_retriever = BGERetrieval(config_path=scene_text_config_path)

    # 3) CLIP Retrieval (image)
    clip_config_path = "config/clip_config.yaml"
    image_retriever = CLIPRetrieval(config_path=clip_config_path)

    # 사용자 쿼리
    user_query = "monkey hitting man"

    # (A) BGE 결과 (frame)
    frame_results = frame_text_retriever.retrieve(user_query, top_k=5)
    # print("\n=== Frame Retrieval Results ===")
    # for i, item in enumerate(frame_results, start=1):
    #     print(
    #         f"Rank {i}: timestamp={item['frame_timestamp']}, "
    #         f"image_path={item['frame_image_path']}, "
    #         # f"caption={item['frame_description']}, "
    #         f"scene_id={item['scene_id']}, "
    #         f"score={item['score']:.4f}"
    #     )

    # (B) BGE 결과 (scene)
    scene_results = scene_text_retriever.retrieve(user_query, top_k=5)
    # print("\n=== Scene Retrieval Results ===")
    # for i, item in enumerate(scene_results, start=1):
    #     print(
    #         f"Rank {i}: start={item['scene_start_time']}, "
    #         f"end={item['scene_end_time']}, "
    #         # f"description={item['scene_description']}, "
    #         f"score={item['score']:.4f}"
    #     )

    # (C) CLIP 결과
    clip_results = image_retriever.retrieve(user_query, top_k=5)
    # print("\n=== CLIP Retrieval Results ===")
    # for i, item in enumerate(clip_results, start=1):
    #     print(f"Rank {i}: image={item['image_filename']}, score={item['score']:.4f}")

    # --- Rank Fusion ---
    fused_top_k = fuse_results(
        frame_results,
        scene_results,
        clip_results,
        w_frame=0.3,
        w_scene=0.3,
        w_clip=0.4,
        top_k=5,
    )

    print("\n=== Rank Fusion Results ===")
    for i, item in enumerate(fused_top_k, start=1):
        print(
            f"Rank {i}: filename={item['filename']}, "
            f"Final={item['final_score']:.4f}, "
            f"frame={item['frame_score']:.2f}, "
            f"scene={item['scene_score']:.2f}, "
            f"clip={item['clip_score']:.2f}, "
            f"scene_id={item['scene_id']}"
        )
