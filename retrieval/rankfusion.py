from code.video_retrieval import BGERetrieval, BLIPRetrieval, CLIPRetrieval, Rankfusion

if __name__ == "__main__":
    text_query = "a person is playing guitar"

    scene_retriever = BGERetrieval(config_path="config/video_retrieval_config.yaml")
    clip_retriever = CLIPRetrieval(config_path="config/video_retrieval_config.yaml")
    blip_retriever = BLIPRetrieval(config_path="config/video_retrieval_config.yaml")

    # scene_retriever.retrieve(text_query, top_k=1000)
    # print("\n=== Scene Retrieval 결과 ===")
    # for res in scene_retriever.results[:10]:
    #     print(
    #         f"Rank {res['rank']}: {res['image_filename']} (Score: {res['score']:.4f})"
    #     )

    # Rankfusion 객체 생성 (각 retrieval 결과의 가중치는 상황에 따라 조절)
    rankfusion = Rankfusion(
        scene_retriever=scene_retriever,
        clip_retriever=clip_retriever,
        blip_retriever=blip_retriever,
        weight_clip=0.4,
        weight_blip=0.4,
        weight_scene=0.2,
    )

    # 최종 Rankfusion 결과 (예: 상위 10개 결과)
    fusion_results = rankfusion.retrieve(text_query, top_k=1000, union_top_n=2000)
    print("\n=== Rankfusion 결과 ===")
    for res in fusion_results[:10]:
        print(
            f"Rank {res['rank']}: {res['image_filename']} "
            f"(CLIP: {res['clip_score']:.4f}, BLIP: {res['blip_score']:.4f}, "
            f"Scene: {res['scene_score']:.4f}, Fusion: {res['fusion_score']:.4f})"
        )

    diverse_results = rankfusion.select_diverse_results_by_clustering(
        fusion_results, desired_num=10, top_n=100
    )
    print("=== Diverse Results from Clustering ===")
    for res in diverse_results:
        print(
            f"Rank {res['rank']}: {res['image_filename']} "
            f"(Fusion: {res['fusion_score']:.4f})"
        )
