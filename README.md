## 운영환경
Linux, cuda 12.2, V100 32GB

## 초기 환경 설치
```bash
cd level4-nlp-finalproject-hackathon-nlp-01-lv3
chmod +x init.sh
./init.sh

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## scene_caption_pipeline.py 사용법
1.1. https://huggingface.co/OpenGVLab/InternVideo2-Chat-8B 에서 동의하고 사용권한 확보  
1.2. https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3 에서 동의하고 사용권한 확보
```bash
huggingface-cli login # 특정 모델 사용시 토큰 필요

python scene_caption_pipeline.py # scene_config.yaml 적절히 모델, prompt 수정
```

## EvalSceneCaption Streamlit 실행방법
```bash
cd EvalSceneCaption
streamlit run app.py --server.address 0.0.0.0 --server.port 1111 # 서버포트는 해당 서버에 맞는걸로
```
또는, "아직 주소는 확정 x" 에서 접속 가능 - 필요시 서버 열겠습니다.


파일구조
```bash
level4-nlp-finalproject-hackathon-nlp-01-lv3
|-- README.md
|-- code # pipeline에서 사용되는 모듈들
|   |-- __init__.py
|   |-- audio_utils.py
|   |-- scene_caption_modules.py
|   |-- scene_utils.py
|   |-- utils.py
|   `-- specific_model_utils
|       |-- InternVideo2_utils.py
|       `-- __init__.py
|-- config
|   `-- scene_config.yaml
|-- download_22video.sh # 22개 동영상 다운
|-- evalSceneCaption # Streamlit 서버로 Scene caption 평가하는 코드
|   `-- app.py
|-- init.sh # apt 설치 및 초기 세팅
|-- mono_audio # mono 오디오 저장경로
|-- output # 모든 Scene에 대한 캡션 결과를 담는 저장경로
|   `-- scene_output_v25.json
|-- requirements.txt
|-- scene_caption_pipeline.py # Scene caption 전과정을 적어놓은 코드
|-- scene_info.json # 오디오 정보를 포함한 씬 정보들 모음
|-- scenes # scene(mp4) 파일들이 모여있는 폴더
|-- video # 전체 영상들이 모여있는 폴더
`-- video_timestamps.txt # 모든 비디오를 씬 단위로 나눈 txt
```