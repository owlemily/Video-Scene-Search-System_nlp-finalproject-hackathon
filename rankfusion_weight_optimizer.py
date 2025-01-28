import torch
from rankfusion_retrieval_pipeline import RankFusionSystem
from sklearn.metrics import ndcg_score
from torch.optim import Adam


class RankFusionWeightOptimizer:
    def __init__(self, rank_fusion_system, learning_rate=0.01):
        """
        Args:
            rank_fusion_system: RankFusionSystem 객체
            learning_rate: 학습률
        """
        self.rank_fusion_system = rank_fusion_system
        self.weights = torch.tensor(
            [
                rank_fusion_system.w_frame,
                rank_fusion_system.w_scene,
                rank_fusion_system.w_clip,
            ],
            requires_grad=True,
            dtype=torch.float32,
        )
        self.optimizer = Adam([self.weights], lr=learning_rate)

    def loss_function(self, fused_results, ground_truth):
        """
        손실 함수: NDCG(Normalized Discounted Cumulative Gain)를 사용하여 순위 기반 평가

        Args:
            fused_results: Rank Fusion 결과 리스트
            ground_truth: 실제 정답 리스트 ([filename, relevance_score])

        Returns:
            loss: 순위 손실 값
        """
        # ground_truth와 fused_results의 relevance 매핑
        relevance_scores = []
        predicted_scores = []

        for result in fused_results:
            filename = result["filename"]
            predicted_scores.append(result["final_score"])
            relevance_scores.append(ground_truth.get(filename, 0))

        # NDCG의 역수 (최소화가 목적이므로)
        ndcg = ndcg_score([relevance_scores], [predicted_scores])
        return 1 - ndcg

    def train(self, training_data, epochs=50, top_k=10):
        """
        가중치 학습

        Args:
            training_data: 학습 데이터 (list of dict)
                [{"query": str, "ground_truth": {filename: relevance_score}}]
            epochs: 학습 반복 횟수
            top_k: 각 쿼리에서 추출할 상위 결과 개수
        """
        for epoch in range(epochs):
            epoch_loss = 0.0

            for data in training_data:
                user_query = data["query"]
                ground_truth = data["ground_truth"]

                # 가중치 업데이트
                with torch.no_grad():
                    self.rank_fusion_system.w_frame = self.weights[0].item()
                    self.rank_fusion_system.w_scene = self.weights[1].item()
                    self.rank_fusion_system.w_clip = self.weights[2].item()

                # 결과 생성 및 손실 계산
                fused_results = self.rank_fusion_system.retrieve_and_fuse(
                    user_query, top_k=top_k
                )
                loss = self.loss_function(fused_results, ground_truth)

                # 역전파 및 최적화
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

        # 최종 가중치 출력
        print("Trained Weights:")
        print(f"w_frame: {self.weights[0].item():.4f}")
        print(f"w_scene: {self.weights[1].item():.4f}")
        print(f"w_clip: {self.weights[2].item():.4f}")


# 사용 예시
if __name__ == "__main__":
    # RankFusionSystem 객체 생성
    rank_fusion_system = RankFusionSystem(
        frame_text_config_path="config/frame_description_config.yaml",
        scene_text_config_path="config/scene_description_config.yaml",
        clip_config_path="config/clip_config.yaml",
        w_frame=1.0,
        w_scene=1.0,
        w_clip=1.0,
    )

    # 학습 데이터 준비 (예시)
    training_data = [
        {
            "query": "Man pushes police officer down with police car",
            "ground_truth": {
                "file1.jpg": 3,  # relevance score
                "file2.jpg": 2,
                "file3.jpg": 1,
            },
        },
        {
            "query": "Dog jumps into the water",
            "ground_truth": {
                "file4.jpg": 3,
                "file5.jpg": 2,
                "file6.jpg": 1,
            },
        },
    ]

    # RankFusionWeightOptimizer 생성 및 학습 실행
    optimizer = RankFusionWeightOptimizer(rank_fusion_system, learning_rate=0.01)
    optimizer.train(training_data, epochs=20, top_k=5)


# 세 가지 가중치(`w_frame`, `w_scene`, `w_clip`)를 학습하기 위해, 최적화 기법(예: Gradient Descent)을 적용하거나 Grid Search, Bayesian Optimization 등의 방법을 사용할 수 있습니다. 여기서는 PyTorch를 활용하여 최적화 기법으로 가중치를 학습하는 코드를 제안합니다.

# 아래는 가중치를 학습하기 위한 코드 예제입니다. **손실 함수**는 `fused 결과의 순위와 실제 레이블(정답 데이터)의 순위 간 차이`를 최소화하는 방식으로 설계됩니다.

# 위 코드는 `RankFusionSystem` 객체의 세 가지 가중치를 학습하도록 설계된 최적화 코드입니다.

# ### 주요 구현 사항
# 1. **NDCG 기반 손실 함수**: 학습된 가중치가 최적화되었는지 평가하기 위해 순위 기반의 NDCG(Normalized Discounted Cumulative Gain)를 사용합니다.
# 2. **PyTorch Optimizer 사용**: `Adam` 옵티마이저를 활용해 가중치를 학습합니다.
# 3. **Training Loop**: 주어진 쿼리와 정답 데이터에 대해 반복적으로 가중치를 업데이트합니다.

# ### 사용법
# 1. `RankFusionSystem` 객체를 생성하여 초기화합니다.
# 2. 학습 데이터(`training_data`)를 준비합니다. 각 쿼리와 관련된 파일의 relevance score를 포함해야 합니다.
# 3. `RankFusionWeightOptimizer` 객체를 생성하고 `train` 메서드를 호출하여 학습을 시작합니다.
