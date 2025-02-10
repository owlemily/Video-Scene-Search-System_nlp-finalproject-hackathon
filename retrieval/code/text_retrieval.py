import json
import os

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


##############################################
# 공통 부모 클래스: BGERetrieval
# - config 파일로부터 모델, 토크나이저, JSON 데이터, 임베딩 파일 경로 등을 로드합니다.
# - 텍스트 리스트(self.texts)에 대해 배치 단위로 임베딩을 수행하고, 코사인 유사도를 계산하는 공통 기능을 제공합니다.
##############################################
class BGERetrieval:
    def __init__(
        self, config_path: str, config_section: str, json_key: str, text_field: str
    ):
        """
        Args:
            config_path (str): 설정 파일 경로.
            config_section (str): 사용하고자 하는 config 내 섹션 이름 (예: "bge-scene" 또는 "bge-script").
            json_key (str): JSON 파일에서 대상 데이터 리스트의 key (예: "scenes" 또는 "scripts").
            text_field (str): 각 항목에서 텍스트로 사용할 필드 이름 (예: "caption" 또는 "summary").
        """
        full_config = self._load_config(config_path)
        if config_section not in full_config:
            raise KeyError(
                f"'{config_section}' 설정이 {config_path}에 존재하지 않습니다."
            )
        self.config = full_config[config_section]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # BGE 모델 및 토크나이저 로드
        self.model_name = self.config["model_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        # JSON 파일 로드: json_key에 해당하는 항목 리스트를 self.data_info로 저장
        self.json_file = self.config["json_file"]
        with open(self.json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.data_info = data.get(json_key, [])
        self.texts = [item[text_field] for item in self.data_info]

        # 임베딩 파일 저장/로딩
        self.embedding_file = self.config["output_file"]
        os.makedirs(os.path.dirname(self.embedding_file), exist_ok=True)
        if os.path.exists(self.embedding_file):
            self._load_embeddings(self.embedding_file)
        else:
            self.embeddings = self._encode_texts(self.texts)
            self._save_embeddings(self.embedding_file)

    def _load_config(self, config_path: str) -> dict:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _encode_texts(
        self, texts: list, batch_size: int = 1024, disable: bool = False
    ) -> torch.Tensor:
        all_embeds = []
        for i in tqdm(
            range(0, len(texts), batch_size), desc="Encoding texts", disable=disable
        ):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                # 일반적으로 CLS 토큰(첫 번째 토큰) 벡터를 사용
                embeds = outputs.last_hidden_state[:, 0, :]
                # L2 Normalization
                embeds = embeds / embeds.norm(dim=-1, keepdim=True)

            all_embeds.append(embeds.cpu())

        return torch.cat(all_embeds, dim=0).to(self.device)

    def _save_embeddings(self, file_path: str) -> None:
        torch.save(
            {"data_info": self.data_info, "features": self.embeddings},
            file_path,
        )
        print(f"[BGE] 임베딩을 {file_path}에 저장했습니다.")

    def _load_embeddings(self, file_path: str) -> None:
        data = torch.load(file_path)
        self.data_info = data["data_info"]
        self.embeddings = data["features"]
        print(f"[BGE] 임베딩을 {file_path}에서 불러왔습니다.")

    def _compute_similarity(
        self, query_vec: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        query_vec = query_vec.to(features.device).float()
        features = features.float()
        return (features @ query_vec.T).squeeze(1)

    # retrieve()는 하위 클래스에서 구체적인 출력 포맷에 맞게 오버라이드합니다.
    def retrieve(self, user_query: str, top_k: int = 5) -> list:
        raise NotImplementedError("하위 클래스에서 retrieve() 메서드를 구현하세요.")


##############################################
# SCENERetrieval: JSON의 scene 정보("scenes" 리스트, 각 항목의 캡션은 "caption") 기반 retrieval
##############################################
class SCENERetrieval(BGERetrieval):
    def __init__(self, config_path: str = "config/merged_config.yaml"):
        # bge-scene 섹션, JSON 내 "scenes" 항목, 텍스트 필드는 "caption" 사용
        super().__init__(
            config_path,
            config_section="bge-scene",
            json_key="scenes",
            text_field="caption",
        )
        self.scene_index = self._build_scene_index_from_json(self.json_file)

    def _build_scene_index_from_json(self, json_file: str) -> dict:
        """
        JSON 파일 내의 scene 정보를 읽어, video_id별로 start_time 기준으로 정렬된 index를 구성합니다.
        """
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        scenes = data.get("scenes", [])
        scene_index = {}
        for scene in scenes:
            video_id = scene.get("video_id")
            if not video_id:
                continue
            try:
                start_time = float(scene["start_time"])
                end_time = float(scene["end_time"])
            except Exception as e:
                print(f"scene 파싱 오류: {e}")
                continue

            # 초기 score는 0.0
            scene_entry = {
                "scene_id": scene.get("scene_id"),
                "start_time": start_time,
                "end_time": end_time,
                "caption": scene.get("caption"),
                "score": 0.0,
            }
            if video_id not in scene_index:
                scene_index[video_id] = {"starts": [], "scenes": []}
            scene_index[video_id]["starts"].append(start_time)
            scene_index[video_id]["scenes"].append(scene_entry)

        # start_time 기준 정렬
        for vid in scene_index:
            combined = list(zip(scene_index[vid]["starts"], scene_index[vid]["scenes"]))
            combined.sort(key=lambda x: x[0])
            scene_index[vid]["starts"] = [item[0] for item in combined]
            scene_index[vid]["scenes"] = [item[1] for item in combined]
        return scene_index

    def retrieve(self, user_query: str, top_k: int = 5) -> list:
        """
        사용자 쿼리에 대해 scene 임베딩 유사도를 계산하여 상위 top_k 결과를 반환합니다.
        """
        query_vec = self._encode_texts([user_query], disable=True)
        scores = self._compute_similarity(query_vec, self.embeddings)
        scores_np = scores.cpu().numpy()
        score_mapping = {
            self.data_info[i]["scene_id"]: float(score)
            for i, score in enumerate(scores_np)
        }
        top_idxs = scores_np.argsort()[-top_k:][::-1]
        results = []
        for rank, idx in enumerate(top_idxs, start=1):
            info = self.data_info[idx]
            scene_id = info["scene_id"]
            results.append(
                {
                    "rank": rank,
                    "video_id": info.get("video_id"),
                    "scene_start_time": info["start_time"],
                    "scene_end_time": info["end_time"],
                    "scene_description": info["caption"],
                    "scene_id": scene_id,
                    "score": score_mapping[scene_id],
                }
            )
        return results


##############################################
# SCRIPTRetrieval: JSON의 script 정보("scripts" 리스트, 각 항목의 요약은 "summary") 기반 retrieval
##############################################
class SCRIPTRetrieval(BGERetrieval):
    def __init__(self, config_path: str = "config/merged_config.yaml"):
        # bge-script 섹션, JSON 내 "scripts" 항목, 텍스트 필드는 "summary" 사용
        super().__init__(
            config_path,
            config_section="bge-script",
            json_key="scripts",
            text_field="summary",
        )
        # script 전용 인덱스 구축 (필요한 경우)
        self.script_index = self._build_script_index_from_json(self.json_file)

    def _build_script_index_from_json(self, json_file: str) -> dict:
        """
        JSON 파일 내의 script 정보를 읽어, video_name과 start 기준으로 정렬된 index를 구성합니다.
        """
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        scripts = data.get("scripts", [])
        script_index = {}
        for script in scripts:
            video_name = script.get("video_name")
            if not video_name:
                continue
            try:
                start_time = float(script["start"])
                end_time = float(script["end"])
            except Exception as e:
                print(f"script 파싱 오류: {e}")
                continue

            # 고유 식별자로 video_name과 start_time 결합
            identifier = f"{video_name}_{start_time}"
            script_entry = {
                "identifier": identifier,
                "video_name": video_name,
                "start": start_time,
                "end": end_time,
                "summary": script.get("summary"),
                "score": 0.0,
            }
            if video_name not in script_index:
                script_index[video_name] = {"start": [], "scripts": []}
            script_index[video_name]["start"].append(start_time)
            script_index[video_name]["scripts"].append(script_entry)
        for vid in script_index:
            combined = list(
                zip(script_index[vid]["start"], script_index[vid]["scripts"])
            )
            combined.sort(key=lambda x: x[0])
            script_index[vid]["start"] = [item[0] for item in combined]
            script_index[vid]["scripts"] = [item[1] for item in combined]
        return script_index

    def retrieve(self, user_query: str, top_k: int = 5) -> list:
        """
        사용자 쿼리에 대해 script 임베딩 유사도를 계산하여 상위 top_k 결과를 반환합니다.
        고유 식별자는 video_name과 start를 결합한 문자열입니다.
        """
        query_vec = self._encode_texts([user_query], disable=True)
        scores = self._compute_similarity(query_vec, self.embeddings)
        scores_np = scores.cpu().numpy()
        # 각 script 항목의 고유 식별자로 score 매핑 생성 (여기서는 start 필드를 사용)
        score_mapping = {
            f"{self.data_info[i]['video_name']}_{self.data_info[i]['start']}": float(
                score
            )
            for i, score in enumerate(scores_np)
        }
        top_idxs = scores_np.argsort()[-top_k:][::-1]
        results = []
        for rank, idx in enumerate(top_idxs, start=1):
            info = self.data_info[idx]
            identifier = f"{info['video_name']}_{info['start']}"
            results.append(
                {
                    "rank": rank,
                    "video_name": info.get("video_name"),
                    "script_start_time": info["start"],
                    "script_end_time": info["end"],
                    "script_summary": info["summary"],
                    "script_identifier": identifier,
                    "score": score_mapping[identifier],
                }
            )
        return results


##############################################
# 사용 예시
##############################################
if __name__ == "__main__":
    # text_query = "A woman grabs a man from behind and holds a knife to his throat, threatening him."
    text_query = "A man is yelling that a rhino is getting too close to his car."
    config_path = "config/video_retrieval_config.yaml"

    # SCENERetrieval 사용 예시 (scene retrieval)
    scene_retriever = SCENERetrieval(config_path=config_path)
    scene_results = scene_retriever.retrieve(text_query, top_k=5)
    print("\n=== SCENERetrieval 결과 ===")
    for res in scene_results:
        print(
            f"Rank {res['rank']}: video_id {res['video_id']} "
            f"({res['scene_start_time']} - {res['scene_end_time']}), "
            f"Description: {res['scene_description']}, Score: {res['score']:.4f}"
        )

    # SCRIPTRetrieval 사용 예시 (script retrieval)
    script_retriever = SCRIPTRetrieval(config_path=config_path)
    script_results = script_retriever.retrieve(text_query, top_k=5)
    print("\n=== SCRIPTRetrieval 결과 ===")
    for res in script_results:
        print(
            f"Rank {res['rank']}: video_id {res['video_name']} "
            f"({res['script_start_time']} - {res['script_end_time']}), "
            f"Summary: {res['script_summary']}, Score: {res['score']:.4f}"
        )
