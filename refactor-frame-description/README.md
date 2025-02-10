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

## DeepL 사용을 위한 API 키 설정
DeepL API 키를 발급받아 아래와 같이 환경변수로 추가해줍니다.
```bash
export DEEPL_API_KEY="your_api_key_here"
```
영구적으로 설정하기 위해서는 '~/.bashrc'에 저장합니다.
```bash
echo 'export DEEPL_API_KEY="your_api_key_here"' >> ~/.bashrc
source ~/.bashrc
```

## 사용할 수 있는 모델 종류
- OpenGVLab/InternVL2_5-4B
- unsloth/Qwen2-VL-7B-Instruct-bnb-4bit

## 22개 동영상에 대한 Frame Caption 생성하는 Pipeline 함수 실행법
비디오들에 대해 Frame으로 모두 나눠 저장하고, 데이터셋을 만들어, 추론까지 진행하는 과정입니다.
base_config.yaml에서 model, prompt를 적절히 수정하고 파이썬 파일을 실행합니다.  
config 파일에서 'frame_caption' 부분만 조정하시면 됩니다.
```bash
python frame_caption_pipeline.py
```

## EvalCaption 테스트셋 다운하는 법 + 해당 테스트셋에 대해 추론하는 Pipeline 함수 실행법
1. 쉘스크립트를 이용해 79개 이미지를 다운받습니다.
2. 파이썬 파일을 실행해 Pipeline을 진행합니다. (여기서는 프레임 추출 후 저장 과정은 포함되어있지 않습니다.)  
이 때, evalCaption.yaml에서 적절히 모델, 프롬프트, max_new_tokens를 조절해주세요.  
config 파일에서 'frame_caption' 부분만 조정하시면 됩니다.
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



파일구조
```bash
level4-nlp-finalproject-hackathon-nlp-01-lv3
|-- README.md
|-- modules
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