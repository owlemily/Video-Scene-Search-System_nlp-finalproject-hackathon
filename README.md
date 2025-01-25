## 운영환경
Linux, cuda 12.1, V100 32GB

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

## EvalCaption 테스트셋 다운 및 모델별 추론하는 법
```bash
chmod +x download_evalCaption_images.sh
./download_evalCaption_images.sh

python evalCaption_pipeline.py # evalCaption.yaml 적절히 모델, prompt 수정
```

## EvalCaption Streamlit 실행방법
```bash
cd Evalcaption
streamlit run app.py --server.address 0.0.0.0 --server.port 1111 # 서버포트는 해당 서버에 맞는걸로
```
또는, http://10.28.224.27:30846/ 에서 접속 가능

## 22개 Retireval 테스트 동영상에 대한 Frame Caption 생성방법
```bash
python frame_caption_pipeline.py # fcs_config.yaml 적절히 수정
```

파일구조
```bash
level4-nlp-finalproject-hackathon-nlp-01-lv3
|-- README.md
|-- code
|   |-- __init__.py
|   |-- frame_caption_modules.py # frame_caption 모듈
|   |-- unsloth_vision_utils.py # Custom_UnslothVisionDataCollator
|   `-- utils.py # 여러 utils (extract_frames_from_folder, create_and_save_dataset 등..)
|-- config # config 저장 경로
|   |-- evalCaption.yaml
|   `-- fcs_config.yaml
|-- datasets # huggingface datasets 저장경로
|   `-- test_dataset_79
|-- download_22video.sh # 22개 영상 다운받는 스크립트
|-- download_evalCaption_images.sh # 79개 이미지 다운받는 스크립트
|-- evalCaption
|   |-- output # streamlit app을 실행시켰을때, save 누를시 Json이 저장되는 경로
|   `-- app.py # streamlit app 코드
|-- evalCaption_pipeline.py # 실행시키면, 79개에 대한 캡셔닝을 진행하는 코드
|-- frame_caption_pipeline.py # 실행시키면, 22개 영상에 대한 캡셔닝을 진행하는 코드
|-- init.sh # 초기 apt 설치 및 22개 영상 다운로드
|-- output # 캡션이 저장되는 경로
|   `-- frame_output_test_dataset_79_v1.json
|-- requirements.txt
`-- test_dataset_79 # 79개 이미지들 다운받았을때 생기는 폴더
```