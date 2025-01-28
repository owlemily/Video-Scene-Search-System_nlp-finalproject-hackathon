import ntpath

# 필요한 모듈에서 각 클래스 불러오기
from code.basic_retrieval import BGERetrieval
from code.image_retrieval import CLIPRetrieval


class RankFusionSystem:
    """
    BGE(frame), BGE(scene), CLIP 세 가지 Retrieval 결과를 합쳐
    최종 점수를 산출하는 시스템을 하나의 클래스로 감싼 예시 코드입니다.

    사용법:
        1) 객체 생성 시 config 경로와 가중치 설정
        2) retrieve_and_fuse(user_query, top_k)를 호출해 결과 얻기
        3) 필요하다면 print_fused_results() 메서드로 가독성 있게 결과 출력
    """

    def __init__(
        self,
        frame_text_config_path: str = "config/frame_description_config.yaml",
        scene_text_config_path: str = "config/scene_description_config.yaml",
        clip_config_path: str = "config/clip_config.yaml",
        w_frame: float = 0.3,
        w_scene: float = 0.4,
        w_clip: float = 0.3,
    ):
        """
        Args:
            frame_text_config_path: BGE 프레임 설명용 config 경로
            scene_text_config_path: BGE 씬 설명용 config 경로
            clip_config_path: CLIP 이미지 검색용 config 경로
            w_frame: frame score 가중치
            w_scene: scene score 가중치
            w_clip: clip score 가중치
        """
        self.frame_retriever = BGERetrieval(config_path=frame_text_config_path)
        self.scene_retriever = BGERetrieval(config_path=scene_text_config_path)
        self.clip_retriever = CLIPRetrieval(config_path=clip_config_path)

        self.w_frame = w_frame
        self.w_scene = w_scene
        self.w_clip = w_clip

    def retrieve_all(self, user_query: str, top_k: int = 100):
        """
        동일한 쿼리에 대해 3가지 Retrieval 결과를 전부 얻어온다.
        Returns:
            tuple: (frame_results, scene_results, clip_results)
        """
        frame_results = self.frame_retriever.retrieve(user_query, top_k=top_k)
        scene_results = self.scene_retriever.retrieve(user_query, top_k=top_k)
        clip_results = self.clip_retriever.retrieve(user_query, top_k=top_k)
        return frame_results, scene_results, clip_results

    def fuse_results(
        self,
        frame_results,
        scene_results,
        clip_results,
        top_k: int = 10,
    ):
        """
        전체 프레임 메타정보가 없는 상황에서,
        frame_results + scene_results + clip_results를 합쳐
        최종 점수를 계산하는 간단한 방법.

        - 후보 프레임: (frame_results의 filename) ∪ (clip_results의 filename)
        - frame_score: frame_results에 있으면 그 점수, 없으면 0
        - scene_score: frame_results 통해 얻은 scene_id가 scene_results topK 중 있으면 점수, 없으면 0
        - clip_score: clip_results에 있으면 그 점수, 없으면 0

        final_score = w_frame * frame_score + w_scene * scene_score + w_clip * clip_score

        Args:
            frame_results: BGE(frame) 결과 리스트
            scene_results: BGE(scene) 결과 리스트
            clip_results: CLIP(image) 결과 리스트
            top_k: 최종 상위 몇 개를 추출할지

        Returns:
            list: final_score 기준 내림차순 정렬된 상위 top_k개의 결과
        """
        # (A) scene_id -> scene_score (scene_results 맵)
        scene_score_map = {}
        for s_item in scene_results:
            sid = s_item.get("scene_id")
            if sid is not None:
                scene_score_map[sid] = s_item.get("score", 0.0)

        # (B) frame_results (filename -> frame_score, scene_id)
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

        # (C) clip_results (filename -> clip_score)
        clip_score_map = {}
        clip_filenames = set()

        for c_item in clip_results:
            raw_path = c_item.get("image_filename", "")
            if raw_path.startswith("./"):
                raw_path = raw_path[2:]
            filename_only = ntpath.basename(raw_path)

            clip_filenames.add(filename_only)
            clip_score_map[filename_only] = c_item.get("score", 0.0)

        # (D) 최종 후보 = frame_filenames ∪ clip_filenames
        candidate_filenames = frame_filenames.union(clip_filenames)

        # (E) 후보 각각에 대해 점수 계산
        fused_list = []
        for fname in candidate_filenames:
            frame_s = frame_score_map.get(fname, 0.0)
            sid = frame_scene_map.get(fname, None)

            scene_s = 0.0
            if sid is not None and sid in scene_score_map:
                scene_s = scene_score_map[sid]

            clip_s = clip_score_map.get(fname, 0.0)

            final_s = (
                self.w_frame * frame_s + self.w_scene * scene_s + self.w_clip * clip_s
            )

            fused_list.append(
                {
                    "filename": fname,
                    "final_score": final_s,
                    "frame_score": frame_s,
                    "scene_score": scene_s,
                    "clip_score": clip_s,
                    "scene_id": sid,
                }
            )

        # (F) final_score 내림차순 정렬 후 상위 top_k
        fused_list.sort(key=lambda x: x["final_score"], reverse=True)
        return fused_list[:top_k]

    def retrieve_and_fuse(self, user_query: str, top_k: int = 5):
        """
        세 가지 Retrieval을 모두 수행한 뒤, fuse_results까지 자동으로 진행한다.
        """
        frame_results, scene_results, clip_results = self.retrieve_all(
            user_query, top_k=100
        )
        fused_top_k = self.fuse_results(
            frame_results, scene_results, clip_results, top_k=top_k
        )
        return fused_top_k

    def print_fused_results(self, fused_list):
        """
        fuse_results로 나온 결과를 보기 좋게 출력한다.
        """
        print("\n=== Rank Fusion Results ===")
        for i, item in enumerate(fused_list, start=1):
            print(
                f"Rank {i}: "
                f"filename={item['filename']}, "
                f"Final={item['final_score']:.4f}, "
                f"frame={item['frame_score']:.2f}, "
                f"scene={item['scene_score']:.2f}, "
                f"clip={item['clip_score']:.2f}, "
                f"scene_id={item['scene_id']}"
            )

    def run_demo(
        self, user_query: str = "Anna and Elsa playing in the snow", top_k: int = 5
    ):
        """
        데모 시나리오: 사용자 쿼리에 대해 Retrieve & Fuse 후 결과를 출력.
        """
        fused_top_k = self.retrieve_and_fuse(user_query=user_query, top_k=top_k)
        self.print_fused_results(fused_top_k)


if __name__ == "__main__":
    # RankFusionSystem 객체 생성 (필요시 가중치, config 경로를 원하는 대로 조절 가능)
    rank_fusion_system = RankFusionSystem(
        frame_text_config_path="config/frame_description_config.yaml",
        scene_text_config_path="config/scene_description_config.yaml",
        clip_config_path="config/clip_config.yaml",
        w_frame=0.2,
        w_scene=0.2,
        w_clip=0.6,
    )

    # 데모 실행
    # 예) "Anna and Elsa playing in the snow" 쿼리에 대해 top_k=5개 결과 표시
    rank_fusion_system.run_demo(user_query="A man on cliff", top_k=5)
