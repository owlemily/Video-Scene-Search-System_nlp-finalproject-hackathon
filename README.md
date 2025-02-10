# TTV_service 소개

현재 폴더는 평가를 위한 Streamlit 앱과 필수 모듈로 구성되어 있습니다.

기존에는 config를 통해 여러 모델과 프롬프트를 조정할 수 있었지만, 평가 목적에 맞춰 모델, 프롬프트, 기타 설정을 모두 고정했습니다.

코드는 확인하기 쉽도록 기존 코드를 조정하고 단순화하여 `video_retrieval.py`와 `external_videos_preprocess.py`에 추가했습니다.

또한, V100 (RAM 32GB) 환경을 고려해 OOM 문제가 발생하지 않도록 `max_new_tokens`, `num_frames`, 입력 프롬프트 길이를 적절히 조정했습니다.

# Streamlit 앱 소개

이 프로젝트는 텍스트 쿼리를 입력받아 여러 retrieval 모델(**SCENE_Retrieval**, **BLIP_Retrieval**, **CLIP_Retrieval**)을 활용해 해당되는 비디오 프레임과 장면(비디오)를 검색하는 시스템입니다.

다양한 모달리티 기반 retrieval 결과를 **Rankfusion** 기법으로 결합하고, 필요 시 **Qwen-VL 기반 Rerank** 기능으로 재평가하여 최종 결과를 도출합니다.

결과는 **Streamlit** 웹 인터페이스에서 시각화되며, 클러스터링 기법을 활용해 중복을 최소화한 diverse 결과를 제공합니다.

총 1533개의 Youtube8M 동영상들과 추가로 넣어준 동영상들에 대하여 검색이 가능합니다.

![Image](https://github.com/user-attachments/assets/5fa4020e-7c2c-48f8-b785-c8c2a0702f40)

쿼리를 입력했을 때, 가장 유사한 Top-5의 프레임과 영상을 반환하여 보여줍니다.

![Image](https://github.com/user-attachments/assets/1d045365-e751-4f16-b960-3060961d9635)

Advanced Search with Rerank를 진행하여 추가 로직을 적용, Top-5 중에 가장 정확한 프레임과 영상구간을 찾아서 사용자에게 제공합니다.

---

# 주요 기능

- Top5 검색 기능
    - 쿼리를 입력했을 때, 가장 유사한 프레임과 영상구간을 유사도가 높은 순서대로 반환합니다.
    - 입력된 쿼리는 영어로 번역되며, 미리 구축한 임베딩을 기반으로 유사도를 계산합니다.
    - **다중 모달리티 Retrieval & Rankfusion**
        - **SCENE_Retrieval**, **BLIP_Retrieval**, **CLIP_Retrieval** 등 여러 retrieval 모델을 초기화하여 사용합니다.
        - Rankfusion 알고리즘을 통해 각 모델의 결과를 가중 합산하여 최종 검색 결과를 산출합니다.
    - **클러스터링 기반 Diverse 결과 제공**
        - 중복되는 결과를 최소화하기 위해 클러스터링 기법을 사용하여 다양한 검색 결과를 제공합니다. (비슷한 프레임들이 Top-5에 여러개 나오지 않도록 처리)
- **Advanced Search with Rerank**
    - Qwen‑VL 기반의 Rerank 기능을 통해 Top-5의 이미지들을 재평가하여 가장 높은 유사도를 가진 이미지와 영상구간을 반환합니다.
    - 추가 로직을 적용하기 때문에, 기본 Top-5 검색기능보다 검색 정확도가 향상됩니다.

---

## 파일 구조

```
├── exteranl_videos # 외부 동영상 저장 디렉토리 (외부 동영상을 이 폴더에 넣어주세요)
├── original_videos # YouTube 8M 동영상 저장 디렉토리
└── ttv
		├── [README.md](http://readme.md/)                            # 프로젝트 개요 및 사용법 설명
		├── [app.py](http://app.py/)                               # Streamlit 웹 애플리케이션 메인 코드
		├── code
		│   ├── **init**.py
		│   ├── [rerank.py](http://rerank.py/)                        # Advanced Rerank (Qwen‑VL 기반) 관련 함수들
		│   └── video_retrieval.py               # retrieval 관련 클래스 및 Rankfusion 구현
		├── config
		│   ├── optimization_config.yaml         # 임베딩 및 가중치 최적화 관련 설정 파일
		│   └── video_retrieval_config.yaml      # retrieval 모델 설정 파일
		├── description
		│   ├── scene_description.json           # scene에 대한 추가 설명 또는 메타데이터
		├── dev
		│   ├── **init**.py
		│   ├── [benchmark.py](http://benchmark.py/)                     # retrieval 성능 벤치마크 스크립트
		│   └── benchmark_en.csv                 # 벤치마크용 데이터 파일 (영문)
		├── embeddings
		│   ├── blip_image_embeddings_1500.pt    # 미리 계산된 BLIP 이미지 임베딩 파일
		│   ├── clip_image_embeddings_1500.pt    # 미리 계산된 CLIP 이미지 임베딩 파일
		│   └── scene_description_embeddings.pt  # 미리 계산된 Scene BGE 임베딩 파일
		├── external_video_preprocess
		│   ├── external_videos_preprocess.py    # 외부 동영상 전처리 스크립트
		│   └── utils                            # 전처리 관련 유틸리티 모듈들
		├── requirements.txt                     # 설치에 필요한 Python 패키지 목록
		├── run_benchmark.sh                     # 벤치마크 실행용 셸 스크립트
		└── weight_optimization.py               # retrieval 가중치 최적화 관련 스크립트
	
```

---

# 환경 설치

V100 서버 기준으로 작성하였습니다.

평가 시에는 이미 V100 서버에 환경과 1533개 비디오들을 저장해두었으니, 별도의 환경 설치가 필요 없습니다.

### 운영 환경

Linux, cuda 12.2, V100 32GB

## 초기 환경 세팅 (사전 제공한 환경에서는 초기 환경 세팅이 불필요합니다., 아래의 방식을 참고 바랍니다.)

1. 아래 명령어를 실행하여 환경을 설치합니다.
    
    ```bash
    conda remove ffmpeg
    chmod +x init.sh
    ./init.sh
    
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
    
2. **DEEPL_API_KEY 설정 (현재 제공된 V100 서버에는 이미 저희의 DEEPL API 키를 설정해두었습니다.)**
    
    DeepL API 키를 발급 받아 아래와 같이 환경 변수로 추가해줍니다.
    
    ```bash
    export DEEPL_API_KEY="your_api_key_here"
    ```
    
    영구적으로 설정하기 위해서는 '~/.bashrc'에 저장합니다.
    
    ```bash
    echo 'export DEEPL_API_KEY="your_api_key_here"' >> ~/.bashrc
    source ~/.bashrc
    ```
    
3. **설정 파일 수정**
    - `config/video_retrieval_config.yaml` 파일에서 retrieval 모델 관련 설정(파라미터, 가중치 등)을 확인하고 필요에 따라 수정합니다.
4. 외부 동영상을 넣을 external_videos 폴더를 만들고, 그 안에 mp4 파일들을 넣어줍니다. (필수)
5. 외부동영상에 대한 전처리를 진행합니다. (CLIP, BLIP, Scene 임베딩을 생성합니다.)
    
    ```bash
    python external_video_preprocess/external_videos_preprocess.py
    ```
    

# 사전 제공 환경 (ssh -p 30977 [root@10.28.224.97](mailto:root@10.28.224.97))

1. ttv 디렉토리로 이동하여 .venv를 실행합니다. 
    
    ```bash
    cd ttv
    source .venv/bin/activate
    ```
    
2. `/data/ephemeral/home/external_videos` 위치에 외부 동영상 파일들을 넣습니다. 
3. 이후, `external_video_preprocess/external_videos_preprocess.py` 스크립트를 실행하여 외부 동영상 전처리를 진행합니다. (CLIP, BLIP, Scene 임베딩을 생성합니다.)
    
    ```bash
    python external_video_preprocess/external_videos_preprocess.py
    ```
    
4. 스트림릿을 실행합니다. 
    
    ```bash
    streamlit run app.py
    ```
    

---

# 실행 방법

### 1. Streamlit 애플리케이션 실행

- 터미널에서 아래 명령어를 입력하여 애플리케이션을 실행합니다.
    
    ```bash
    streamlit run app.py
    ```
    
- 실행 후 브라우저가 자동으로 열리지 않으면 [http://localhost:8501](http://localhost:8501/) 주소를 직접 입력하여 접속합니다.

---

# 애플리케이션 사용 방법

- 기본 Top-5 검색 기능 (기본적으로 적용)
    - 입력 항목: 쿼리를 입력하고 “검색” 버튼을 누릅니다.
    - 설명: 미리 구축해둔 임베딩을 기반으로 각 Retriever에서 유사도를 계산하고 Rankfusion 기법으로 점수를 계산하여 Top-5의 이미지와 영상 구간을 반환합니다. Top-5에 중복되는 이미지를 제거하기 위해 클러스터링 방법도 도입하였습니다.

- Advabced Searcg with Rerank
    - 입력 항목: 쿼리 위의 체크박스를 클릭합니다.
    - 설명:
        - Qwen‑VL 기반의 재평가 과정을 통해 Top-5 중 가장 유사도가 높은 이미지와 영상 구간을 반환합니다.
        - 페이지 하단의 Expander(펼침 메뉴)를 클릭하면 Fusion 검색 결과 요약(상위 순위 결과)을 확인할 수 있습니다.
        - Rerank된 검색결과를 Top-1의 결과로, 나머지 4개의 결과를 순서대로 Top-2 ~ Top-5의 결과로 확인하실 수 있습니다.

---

# 기타 기능 - 하이퍼파라미터 튜닝을 위해 사용했던 모듈들

## 1. 벤치마크 실행 (옵션)

- retrieval 시스템의 성능 평가를 위해 `dev/benchmark.py` 스크립트를 실행하거나, 제공된 셸 스크립트 `run_benchmark.sh`를 사용합니다.
    
    ```bash
    bash run_benchmark.sh
    ```
    

## 2. 가중치 최적화

- retrieval 가중치 최적화를 위해 `weight_optimization.py` 스크립트를 실행할 수 있습니다.

---

## 추가 정보

- **Embeddings 폴더:**
    - `embeddings` 폴더에는 미리 계산된 BLIP와 CLIP 이미지 임베딩 파일이 저장되어 있어 검색 성능을 향상시키는데 사용됩니다.
- **Description 파일:**
    - `description` 폴더는 장면(scene) 출력 및 전사 요약과 관련된 JSON 파일들을 포함합니다.
- **개발 도구:**
    - `dev` 폴더 내의 스크립트들은 retrieval 시스템의 성능 평가 및 기타 개발 도구들을 제공합니다.
- **외부 동영상 전처리:**
    - `/data/ephemeral/home/external_videos` 위치에 외부 동영상을 넣고,
    - `external_video_preprocess/external_videos_preprocess.py` 스크립트를 실행하여 추가 동영상 파일들을 전처리할 수 있습니다.

---

## 문제 해결 및 지원

- **비디오 파일 미존재**: 입력한 Video ID에 해당하는 비디오 파일이 `external_videos` 나 `original_videos`내에 파일이 존재하는지 확인하세요.
- **시간 형식 오류**: 시작 시간, 종료 시간 또는 timestamp가 숫자 형식인지 확인하세요.
- **DEEPL_API_KEY 오류**: DeepL 번역기를 사용 중이라면 환경 변수에 올바른 API 키가 설정되었는지 확인하세요.
