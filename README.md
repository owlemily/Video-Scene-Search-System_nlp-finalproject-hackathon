## 운영환경
Linux, cuda 12.2, V100 32GB

## 초기 환경 설치
저희 코드에서는 동영상 처리를 위해 FFmpeg를 사용합니다.  
저희 V100 서버의 경우, Conda의 FFmpeg를 사용하도록 설정되어 있어서 지워야 됩니다.  
아래와 같이 경로가 "/opt/conda/bin/ffmpeg"로 나온다면 Conda에서 먼저 ffmpeg를 지워주세요.
```bash
which ffmpeg
/opt/conda/bin/ffmpeg
```
### Conda에서 지우기
```bash
conda remove ffmpeg
which ffmpeg
/usr/bin/ffmpeg
```
### 그 후, 아래 명령어를 실행하여 환경을 설치합니다.
```bash
cd level4-nlp-finalproject-hackathon-nlp-01-lv3
chmod +x init.sh
./init.sh

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 사용할 수 있는 모델 종류
- OpenGVLab/InternVide/usr/bin/ffmpego2-Chat-8B
- lmms-lab/LLaVA-Video-7B-Qwen2
- OpenGVLab/InternVideo2_5_Chat_8B

### 특정 모델 사용시 사용권한을 얻어야 합니다.
1. https://huggingface.co/OpenGVLab/InternVideo2-Chat-8B 에서 동의하고 사용권한 확보  
2. https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3 에서 동의하고 사용권한 확보
3. 그 후에, Hugging Face Token으로 로그인을 합니다.
```bash
huggingface-cli login
```

## 22개의 비디오에 대해 Scene을 추출하고 Caption을 다는 Pipeline 함수 실행법
base_config.yaml에서 model, prompt, use_audio 등을 적절히 수정하고 파이썬 파일을 실행합니다.  
config에서는 'scene_caption' 부분만 설정하시면 됩니다.
```bash
python scene_caption_pipeline.py
```

## 이미 구축된 Scene 모음(mp4)들에 대해 Caption을 다는 Pipeline 함수 실행법
evalSceneCaption.yaml에서 model, prompt, use_audio 등을 적절히 수정하고 파이썬 파일을 실행합니다.  
config에서는 'scene_caption' 부분만 설정하시면 됩니다.
```bash
python evalSceneCaption_pipeline.py
```

## EvalSceneCaption Streamlit 실행방법
```bash
cd EvalSceneCaption
streamlit run app.py --server.address 0.0.0.0 --server.port 1111 # 서버포트는 해당 서버에 맞는걸로
```
또는, "아직 주소는 확정 x" 에서 접속 가능 - 필요시 서버 열겠습니다.

# Config 형식
```bash
# 공통적으로 사용되는 설정
# device: 사용할 디바이스 ("cuda" 또는 "cpu")
# video_folder: 원본 비디오 파일 저장 경로 (예: "./video")
general:
  video_folder: "./video"

# Frame Captioning을 할 때 사용되는 설정
frame_caption:
  # frames_folder: 프레임 이미지 저장 경로 (예: "./frames") - Video 폴더에서 프레임 이미지를 추출하여 저장하는 폴더이자, 이 폴더의 이미지를 이용하여 캡셔닝을 수행
  # key_frames_folder: 키프레임 이미지를 저장할 폴더경로 (예: "./key_frames")
  # frame_rate: 초당 추출할 프레임 수 (FPS) (예: 4)
  # frame_rate는 비디오 폴더에서 frames_folder에 프레임 이미지를 추출할 때 사용되는 프레임 수입니다.
  # frame_rate를 사용하지 않고, 모든 프레임을 전후로 비교해 키프레임을 추출하여 key_frames_folder에 저장할 수도 있습니다.
  frames_folder: "./frames"
  key_frames_folder: "./key_frames"
  frame_rate: 4

  # output_folder: 캡셔닝 진행 후 결과파일이 저장될 폴더경로 (예: "./frames_output")
  # frame_output_filename: 캡셔닝 진행 후 결과파일명 (예: "frame_output_test_p1_v1.json")
  output_folder: "./frames_output"
  frame_output_filename: "frame_output_p1_v1.json"

  # model: 사용할 캡셔닝 모델 ("OpenGVLab/InternVL2_5-4B" 또는 "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit")
  model: "OpenGVLab/InternVL2_5-4B"

  # use_datasets: 데이터셋 사용 여부 (True 또는 False) - False일 시, 밑의 설정 모두 무시
  # datasets_folder: 데이터셋 저장 경로 (예: "./frames_datasets") - frames 폴더에서 이미지 1개씩 가져와서 추론하지 않고, 속도향상을 위해 데이터셋으로 만들어 저장해둘 폴더경로 (옵션임)
  # datasets_name: 데이터셋 이름 (예: "dataset_v1") - datasets 폴더에 저장할 데이터셋 이름 ("datasets_folder/datasets_name" 경로에 저장됨)
  use_datasets: True
  datasets_folder: "./frames_datasets"
  datasets_name: "dataset_v1"

  # prompt: 캡셔닝 모델 프롬프트 (예: "<image>\nDescribe this image in detail.") - 앞에 무조건 "<image>\n"을 붙여주시고, 뒤에 프롬프트 작성해주세요.
  # max_new_tokens: 생성할 최대 토큰 수 (예: 128)
  # batch_size: 배치 크기 (예: 1) - unsloth 모델은 6~8, OpenGVLab 모델은 1로 설정하면 됩니다. (확실히 실험은 안해봐서 한번 조정해서 해보세요)
  # batch_size는 use_datasets가 True일 때만 사용됩니다.
  prompt: "<image>\nDescribe this image in detail."
  max_new_tokens: 128 
  batch_size: 1


# Scene Captioning을 할 때 사용되는 설정
scene_caption:
  # fps_adjusted_video_folder: FPS 조정된 비디오 파일 저장 경로 (예: "./fps_adjusted_video") - Video 폴더에서 각 비디오의 FPS를 조정하여 저장하는 폴더
  fps_adjusted_video_folder: "./fps_adjusted_video"

  # timestamp_file: 씬 타임스탬프 파일 경로 (예: "./video_timestamps.txt") - 비디오 폴더에서 모든 비디오에 대해 PySceneDetect를 이용하여 추출한 씬들의 타임스탬프가 적힌 파일
  PySceneDetect_threshold: 30.0
  PySceneDetect_min_scene_len: 2
  timestamp_file: "./timestamps.txt"

  # scene_folder: 추출된 씬 이미지 저장 경로 (예: "./scenes") - Video 폴더에서 각 비디오의 Scene들을 timestamp_txt_file에 따라 쪼개서 저장하는 경로
  # output_folder: 캡셔닝 진행 후 결과파일이 저장될 폴더경로 (예: "./scenes_output")
  # scene_output_filename: 씬 캡셔닝 결과 파일명 (예: "scene_output_p1_v1.json")
  scene_folder: "./scenes"
  output_folder: "./scenes_output"
  scene_output_filename: "scene_output_p1_v1.json"

  # model: 사용할 씬 캡셔닝 모델 ("OpenGVLab/InternVideo2_5_Chat_8B" 또는 "lmms-lab/LLaVA-Video-7B-Qwen2" 또는 "OpenGVLab/InternVideo2-Chat-8B")
  model: "OpenGVLab/InternVideo2_5_Chat_8B"

  # prompt: 씬 캡셔닝 모델 프롬프트 (예: "Describe the video scene in detail.")
  # max_new_tokens: 생성할 최대 토큰 수 (예: 256)
  # max_num_frames: 씬 캡셔닝 시 사용할 최대 프레임 수 (예: 64)
  prompt: "Describe the video scene in detail."
  max_new_tokens: 256
  max_num_frames: 64
  
  audio:
    # use_audio: 오디오 사용 여부 (True 또는 False) - False일 시, 밑의 설정 모두 무시
    # mono_audio_folder: 모노 오디오 파일 저장 경로 (예: "./mono_audio") - Scenes 폴더의 모든 Scene의 모노 오디오를 저장하는 폴더
    # scene_info_with_audio_scripts_file: 오디오 스크립트가 포함된 Scene 정보 JSON 파일 경로 (예: "./scene_info_with_audio_scripts.json" 또는 null)
    # scene_info_with_audio_scripts_file에 파일이 지정될 경우, 이미 번역된 대사들이 담긴 JSON 파일을 사용하여 캡셔닝을 수행합니다.
    # scene_info_with_audio_scripts_file이 null일 경우, 즉석에서 저장된 mono 오디오를 사용하여 whisper 모델로 대사를 추출하고 캡셔닝을 수행합니다.
    use_audio: False
    mono_audio_folder: "./mono_audio"
    scene_info_with_audio_scripts_file: "./scene_info_with_audio_scripts.json"
```

# 파일구조
```bash
level4-nlp-finalproject-hackathon-nlp-01-lv3
|-- README.md
|-- modules # pipeline에서 사용되는 모듈들
|   |-- __init__.py
|   |-- audio_utils.py
|   |-- scene_caption_modules.py
|   |-- scene_utils.py
|   |-- utils.py
|   |-- extra_utils.py
|   `-- specific_model_utils
|   |   |-- InternVideo2_5_Chat_utils.py
|   |   |-- InternVideo2_utils.py
|   |   |-- LlavaVideo_utils.py
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