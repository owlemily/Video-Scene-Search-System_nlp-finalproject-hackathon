import ntpath

# 두 모듈에서 각 클래스 불러오기
from code.basic_retrieval import BGERetrieval
from code.image_retrieval import CLIPRetrieval


def fuse_results(
    text_results,  # BGERetrieval (text-based) 결과
    image_results,  # CLIPRetrieval (image-based) 결과
    w_bge=0.5,  # BGE 점수 가중치
    w_clip=0.5,  # CLIP 점수 가중치
    top_k=10,
):
    """
    텍스트 검색 결과(text_results)와 이미지 검색 결과(image_results)를 합쳐
    최종 랭킹을 구하는 함수.

    각 결과 예시:
      image_results: [
        {
          'rank': 1,
          'image_filename': 'test_dataset_79/mDUSjBiHYeY_29.750.jpg',
          'score': 0.2800
        }, ...
      ]
      text_results: [
        {
          'rank': 1,
          'frame_timestamp': '50.5',
          'frame_image_path': './test_dataset_79/5qlG1ODkRWw_50.500.jpg',
          'score': 0.8501
        }, ...
      ]

    :param text_results: BGE(Retrieval) 결과 리스트
    :param image_results: CLIP(Retrieval) 결과 리스트
    :param w_bge: BGE 점수에 대한 가중치
    :param w_clip: CLIP 점수에 대한 가중치
    :param top_k: 최종 상위 몇 개 결과를 뽑을지
    :return: rank fusion 후 (최종 점수 기준 내림차순) 상위 top_k 리스트
    """

    # 1) 결과를 dict 형태로 변환하여 관리
    #    key = 파일명 (예: "mDUSjBiHYeY_29.750.jpg")
    fusion_dict = {}

    # --- BGE(Text) 결과 처리 ---
    for item in text_results:
        # 예: "./test_dataset_79/5qlG1ODkRWw_50.500.jpg"
        raw_path = item.get("frame_image_path", "")
        # (1) 앞의 "./" 제거
        if raw_path.startswith("./"):
            raw_path = raw_path[2:]
        # (2) 파일명만 추출
        filename_only = ntpath.basename(raw_path)  # "5qlG1ODkRWw_50.500.jpg"

        fusion_dict[filename_only] = {
            "bge_score": item["score"],
            "clip_score": 0.0,
            "bge_info": item,
            "clip_info": None,
        }

    # --- CLIP(Image) 결과 처리 ---
    for item in image_results:
        # 예: "test_dataset_79/mDUSjBiHYeY_29.750.jpg"
        raw_path = item.get("image_filename", "")
        # (1) 혹시 모를 "./" 제거
        if raw_path.startswith("./"):
            raw_path = raw_path[2:]
        # (2) 파일명만 추출
        filename_only = ntpath.basename(raw_path)  # "mDUSjBiHYeY_29.750.jpg"

        if filename_only in fusion_dict:
            # 이미 BGE에서 들어온 항목이 있는 경우
            fusion_dict[filename_only]["clip_score"] = item["score"]
            fusion_dict[filename_only]["clip_info"] = item
        else:
            # 신규
            fusion_dict[filename_only] = {
                "bge_score": 0.0,
                "clip_score": item["score"],
                "bge_info": None,
                "clip_info": item,
            }

    # 2) 최종 스코어 계산 (가중합)
    fused_list = []
    for filename_key, v in fusion_dict.items():
        fused_score = w_bge * v["bge_score"] + w_clip * v["clip_score"]
        fused_list.append(
            {
                "filename": filename_key,
                "final_score": fused_score,
                "bge_score": v["bge_score"],
                "clip_score": v["clip_score"],
                "bge_info": v["bge_info"],
                "clip_info": v["clip_info"],
            }
        )

    # 3) final_score 내림차순 정렬 후 상위 top_k만
    fused_list.sort(key=lambda x: x["final_score"], reverse=True)

    # 최종 결과 반환
    return fused_list[:top_k]


if __name__ == "__main__":
    """
    예시 실행:
    - 1) BGE Retrieval (텍스트 기반) 객체 생성 후, 사용자 쿼리에 대해 top_k 추출
    - 2) CLIP Retrieval (텍스트->이미지 매칭) 객체 생성 후, 동일 쿼리에 대해 top_k 추출
    - 3) fuse_results 함수로 랭크퓨전 진행 (w_bge, w_clip으로 가중치 조절)
    """

    # 1) BGE Retrieval 생성
    text_config_path = "config/basic_config.yaml"
    text_retriever = BGERetrieval(config_path=text_config_path)

    # 2) CLIP Retrieval 생성
    clip_config_path = "config/clip_config.yaml"
    image_retriever = CLIPRetrieval(config_path=clip_config_path)

    # 사용자 쿼리(텍스트)
    user_query = "Reindeer"  # 예시

    # BGE 결과 (top_k=5)
    bge_results = text_retriever.retrieve(user_query, top_k=5)

    # CLIP 결과 (top_k=5)
    clip_results = image_retriever.retrieve(user_query, top_k=5)

    # --- Rank Fusion ---
    # 예시: BGE 쪽에 좀 더 가중치(0.7)를 주고, CLIP 쪽에는 0.3 부여
    fused_top_k = fuse_results(
        bge_results, clip_results, w_bge=0.7, w_clip=0.3, top_k=5
    )

    print("\n=== Rank Fusion Results (w_bge=0.7, w_clip=0.3) ===")
    for rank, item in enumerate(fused_top_k, start=1):
        print(
            f"Rank {rank}: key={item['key']}, "
            f"FinalScore={item['final_score']:.4f}, "
            f"BGE={item['bge_score']:.4f}, CLIP={item['clip_score']:.4f}"
        )
