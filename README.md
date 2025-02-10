## RankFusion 검색 파이프라인

이 프로젝트는 여러 검색 방법(프레임, 씬, 이미지 검색)을 결합하여 사용자 쿼리에 대한 랭킹 결과를 제공하는 **RankFusion 검색 파이프라인**을 구현합니다. 이 파이프라인은 다음을 활용합니다:

1. **BGERetrieval (기본 기하 임베딩 검색)**: 텍스트 기반 프레임 및 씬 검색을 수행합니다.
2. **CLIPRetrieval**: CLIP 임베딩을 사용하여 이미지 기반 검색을 수행합니다.

## 프로젝트 구조

```
.
├── README.md                # 프로젝트 문서
├── code
│   ├── image_retrieval.py  # CLIP 및 BLIP 기반 이미지 검색 구현
│   ├── text_retrieval.py  # BGE 기반 텍스트 검색 (Scene, Script)
│   ├── video_retrieval.py  # RankFusion을 포함한 검색 시스템 구현
|-- config
|   |-- clip_config.yaml               # CLIP 검색 설정 파일
|   |-- frame_description_config.yaml  # 프레임 검색 설정 파일
|   -- scene_description_config.yaml   # 씬 검색 설정 파일
|-- datasets
|   |-- frames_22           # 프레임 검색용 데이터셋
|   -- test_dataset_79      # 테스트 데이터셋
|-- description
|   |-- frame_output_test_dataset_79_v1.json  # 프레임 검색 출력 샘플 (v1)
|   |-- frame_output_test_dataset_79_v2.json  # 프레임 검색 출력 샘플 (v2)
|   |-- frame_output_v3_unsloth_22.json       # 프레임 검색 출력 샘플 (v3)
|   -- scene_output_v22.json                  # 씬 검색 출력 샘플
├── dev
│   ├── benchmark.py  # 벤치마크 및 성능 평가 스크립트
│   ├── benchmark_en.csv  # 평가용 벤치마크 데이터
│   ├── retrieval_results.csv  # 검색 결과 저장 파일
│   ├── eval.csv  # 검색 성능 평가 결과
|-- init_dataset
|   |-- download_test_dataset_79.sh  # 테스트 데이터셋 다운로드 스크립트
|   |-- download_video.sh            # 비디오 데이터 다운로드 스크립트
|   |-- only_extract_frames.py       # 비디오에서 프레임을 추출하는 스크립트
|   -- video                         # 원본 비디오 파일 디렉토리
├── rankfusion.py  # RankFusion 실행 스크립트
├── weight_optimization.py  # 가중치 최적화 스크립트
├── requirements.txt  # 프로젝트 종속성 목록
├── requirements.txt                  # 프로젝트 필수 라이브러리
```

---

## 주요 구성 요소

### 1. **BGERetrieval (기본 기하 임베딩 검색)**

이 모듈은 텍스트 설명을 기반으로 **프레임 및 씬 검색**을 수행합니다.

- **프레임 검색**: 사용자 쿼리에 따라 적절한 프레임을 검색합니다.
- **씬 검색**: 사용자 쿼리에 따라 적절한 씬을 검색합니다.

### 2. **CLIPRetrieval (CLIP 기반 이미지 검색)**

- CLIP 임베딩을 활용하여 쿼리에 가장 적합한 이미지를 검색합니다.

### 3. **RankFusion 랭킹 결합 로직**

`fuse_results` 함수는 프레임, 씬, 이미지 검색 결과를 **가중치를 적용한 점수 기반**으로 결합하여 최종 랭킹을 생성합니다.

- **입력**: 프레임, 씬, 이미지 검색 결과
- **가중치**: 프레임, 씬, CLIP 점수를 조정할 수 있는 파라미터
- **출력**: 최종 랭킹이 적용된 상위 k개 검색 결과

---

## 설정 파일

- `config/frame_description_config.yaml`: 프레임 검색 설정
- `config/scene_description_config.yaml`: 씬 검색 설정
- `config/clip_config.yaml`: 이미지 검색 설정

---

## 사용 방법

### 1. **환경 설정**

Python 3 및 `requirements.txt`에 나열된 필수 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```

---

### 2. **데이터 준비**

- `init_dataset/download_test_dataset_79.sh` 스크립트를 실행하여 **테스트 데이터셋**을 다운로드합니다.

```bash
bash init_dataset/download_test_dataset_79.sh
```

- `init_dataset/only_extract_frames.py`를 실행하여 **비디오에서 프레임을 추출**합니다.

```bash
python init_dataset/only_extract_frames.py
```

---

### 3. **검색 파이프라인 실행**

RankFusion 검색 파이프라인을 실행합니다.

```bash
python rankfusion_retrieval_pipeline.py
```

---

### 4. **쿼리 예제**

`rankfusion_retrieval_pipeline.py`에서 사용자 쿼리를 수정하여 검색을 수행할 수 있습니다.

```python
user_query = "원숭이가 사람을 때리는 장면"
```

---

### 5. **출력 예시**

파이프라인을 실행하면 다음과 같은 **검색 결과 랭킹**이 출력됩니다.

```
=== Rank Fusion 결과 ===
Rank 1: filename=frame_001.jpg, Final=0.8540, frame=0.30, scene=0.40, clip=0.15, scene_id=12
Rank 2: filename=frame_002.jpg, Final=0.7450, frame=0.25, scene=0.35, clip=0.14, scene_id=8
...
```

- `Final`: 최종 점수
- `frame`: 프레임 검색 점수
- `scene`: 씬 검색 점수
- `clip`: 이미지 검색 점수
- `scene_id`: 매칭된 씬 ID

---

## 파일 설명

### **코드 파일**

- `basic_retrieval.py` → `BGERetrieval` (프레임 및 씬 검색) 구현
- `image_retrieval.py` → `CLIPRetrieval` (이미지 검색) 구현

### **데이터셋**

- `frames_22` → 프레임 검색용 데이터셋
- `test_dataset_79` → 검색 성능 테스트용 데이터셋

### **유틸리티 스크립트**

- `rankfusion_retrieval_pipeline.py` → RankFusion 검색 파이프라인 메인 스크립트
- `assign_scene_id.py` → 프레임에 씬 ID를 매핑하는 유틸리티
- `only_extract_frames.py` → 비디오에서 프레임을 추출하는 스크립트

---

## 향후 개선 사항

- 고급 랭킹 결합 기법 추가  
- 실시간 검색 파이프라인 지원  
- 가중치 최적화 알고리즘 적용  

---

질문이나 기여 사항이 있다면 프로젝트 관리자를 통해 문의해 주세요! 
