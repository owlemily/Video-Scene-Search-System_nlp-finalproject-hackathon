# 비디오 검색 애플리케이션

이 애플리케이션은 텍스트 기반 쿼리를 입력받아, 여러 retrieval 모델을 활용하여 관련 비디오 프레임과 장면(scene)을 검색 및 추출하는 시스템입니다. 다양한 모달리티(예: scene, script, BLIP, CLIP) 기반의 retrieval을 Rankfusion 기법으로 결합하여 최종 검색 결과를 도출하며, 결과는 Streamlit 웹 인터페이스를 통해 시각화됩니다.

---

## 주요 기능

- **텍스트 쿼리 번역**  
  입력된 쿼리를 영어로 번역합니다. (기본적으로 [googletrans](https://pypi.org/project/googletrans/) 사용, 필요 시 DeepL 사용 가능)

- **다중 모달리티 Retrieval & Rankfusion**  
  - SCENERetrieval, SCRIPTRetrieval, BLIPRetrieval, CLIPRetrieval 등 여러 retrieval 모델을 초기화하여 사용합니다.
  - Rankfusion 알고리즘을 통해 각 모델의 결과를 가중 합산하여 최종 검색 결과를 산출합니다.

- **비디오 프레임 및 장면(scene) 추출**  
  - OpenCV를 사용하여 비디오의 특정 시간에서 프레임 이미지를 추출합니다.
  - FFmpeg를 이용해 비디오의 지정 구간(장면)을 잘라내어 저장합니다.

- **클러스터링 기반 Diverse 결과 제공**  
  중복되는 결과를 최소화하기 위해 클러스터링 기법을 사용하여 다양한 검색 결과를 제공합니다.

- **Streamlit 웹 인터페이스**  
  사용자 쿼리 입력, 결과 출력 및 시각화를 위한 직관적인 웹 UI를 제공합니다.

---

## 파일 구조

```
|-- README.md
|-- app.py                          # Streamlit 웹 애플리케이션 메인 코드
|-- code
|   |-- __init__.py
|   `-- video_retrieval.py          # retrieval 관련 클래스 및 Rankfusion 구현
|-- config
|   `-- video_retrieval_config.yaml # retrieval 모델 설정 파일
|-- description
|   |-- scene_output_22.json        # 장면(scene) 출력 예시 파일
|   `-- transcription_summary_ver1.json  # 전사 요약 예시 파일
|-- dev
|   |-- __init__.py
|   |-- benchmark.py                # retrieval 성능 벤치마크 스크립트
|   `-- benchmark_en.csv            # 벤치마크용 데이터 파일
|-- embeddings
|   |-- blip_image_embeddings_1500.pt  # 미리 계산된 BLIP 이미지 임베딩 파일
|   `-- clip_image_embeddings_1500.pt  # 미리 계산된 CLIP 이미지 임베딩 파일
|-- requirements.txt                # 설치에 필요한 Python 패키지 목록
|-- run_benchmark.sh                # 벤치마크 실행용 셸 스크립트
```

---

## 설치 방법

1. **Python 환경 준비**  
   Python 3.7 이상이 설치되어 있어야 합니다.

2. **패키지 설치**  
   터미널에서 아래 명령어를 실행하여 `requirements.txt`에 명시된 모든 패키지를 설치합니다.
   ```bash
   pip install -r requirements.txt
   ```

3. **환경 변수 설정 (DeepL 사용 시)**  
   DeepL 번역기를 사용하려면, DeepL API 키를 환경 변수 `DEEPL_API_KEY`에 등록합니다.
   ```bash
   export DEEPL_API_KEY=your_deepl_api_key
   ```

4. **설정 파일 수정**  
   `config/video_retrieval_config.yaml` 파일에서 retrieval 모델 관련 설정을 확인하고, 필요에 따라 수정합니다.

---

## 실행 방법

### 1. Streamlit 애플리케이션 실행

터미널에서 아래 명령어를 입력하여 애플리케이션을 실행합니다.
```bash
streamlit run app.py
```

실행 후, 브라우저가 자동으로 열리지 않는 경우 [http://localhost:8501](http://localhost:8501) 주소를 직접 입력하여 접속할 수 있습니다.

### 2. Streamlit 페이지 내에서 애플리케이션 사용법

Streamlit 웹 페이지가 열리면, 다음과 같이 애플리케이션을 사용할 수 있습니다.

- **쿼리 입력**  
  페이지 상단에 있는 텍스트 입력란에 검색하고자 하는 쿼리를 입력합니다. 기본 예시 쿼리가 미리 입력되어 있으나, 원하는 내용을 수정할 수 있습니다.

- **검색 버튼 클릭**  
  입력한 쿼리에 대해 검색을 수행하려면 **검색** 버튼을 클릭합니다.

- **쿼리 번역 및 검색 진행**  
  내부적으로 쿼리가 영어로 번역된 후, Rankfusion 기법을 사용하여 다양한 retrieval 모델의 결과를 합산합니다.

- **검색 결과 확인**  
  클러스터링을 통해 diverse 결과가 선별되며, 각 결과에 대해 비디오 프레임 이미지와 (장면 정보가 있을 경우) 장면 동영상이 side-by-side 또는 개별적으로 표시됩니다.

- **추가 결과 확인**  
  페이지 하단의 Expander(펼침 메뉴)를 클릭하면, Fusion 검색 결과 요약(상위 순위 결과)을 확인할 수 있습니다.

- **임시 파일 처리**  
  검색이 완료되면, 임시 폴더 내에 생성된 이미지 및 동영상 파일들이 자동으로 삭제되어 저장 공간을 관리합니다.

### 3. 벤치마크 실행 (옵션)

retrieval 시스템의 성능 평가를 위해 `dev/benchmark.py` 스크립트를 실행하거나, 제공된 셸 스크립트 `run_benchmark.sh`를 사용합니다.
```bash
bash run_benchmark.sh
```

---

## 추가 정보

- **Embeddings 폴더:**  
  `embeddings` 폴더에는 미리 계산된 BLIP와 CLIP 이미지 임베딩 파일이 저장되어 있어, 검색 성능을 향상시키기 위해 사용됩니다.

- **Description 파일:**  
  `description` 폴더는 장면 출력 및 전사 요약과 관련된 JSON 파일들을 포함합니다.

- **개발 도구:**  
  `dev` 폴더 내의 스크립트들은 retrieval 시스템의 성능 평가 및 기타 개발 도구들을 제공합니다.

---

## 기여 및 문의

- **기여:**  
  Pull Request를 통한 코드 개선 및 수정 제안을 환영합니다.

- **문의:**  
  버그 신고나 기능 개선 제안은 GitHub 이슈를 통해 남겨주시기 바랍니다.

---

## 라이선스

프로젝트에 적용할 라이선스를 명시하세요. (예: MIT License)

---

이 프로젝트는 다양한 retrieval 모델과 클러스터링 기법을 활용하여 비디오 검색의 효율성과 다양성을 높이는 것을 목표로 합니다. 여러분의 많은 관심과 기여 부탁드립니다.