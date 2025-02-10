# VTT_service 소개

현재 폴더는 평가를 위한 Streamlit 앱과 필수 모듈로 구성되어 있습니다.

기존에는 config를 통해 여러 모델과 프롬프트를 조정할 수 있었지만, 평가 목적에 맞춰 모델, 프롬프트, 기타 설정을 모두 고정했습니다.

코드는 확인하기 쉽도록 기존 코드를 조정하고 단순화하여 `vtt_service_utils.py`와 `captioning.py`에 추가했습니다.

또한, V100 (RAM 32GB) 환경을 고려해 OOM 문제가 발생하지 않도록 `max_new_tokens`, `num_frames`, 입력 프롬프트 길이를 적절히 조정했습니다.

# Streamlit 앱 소개

이 프로젝트는 특정 영상 씬 또는 프레임에 캡션을 생성하는 Streamlit 기반 애플리케이션입니다.

Video_id와 Timestamp를 직접 입력하여 실시간으로 캡션 확인이 가능하고, Txt 파일을 넣어 배치 처리방식으로 한번에 결과를 확인할 수도 있습니다.

<img width="1567" alt="Image" src="https://github.com/user-attachments/assets/de5ce3f1-e177-4915-ab67-cbed0751860b" />

---

# 주요 기능

- **Scene Captioning (씬 캡셔닝)**
    - Video_ID(파일명)과 시작 시간(Start Time), 끝 시간(End Time)을 넣으면 해당 구간의 캡션을 보여줍니다.
    - Single Scene Input 모드에서는 사용자가 하나씩 입력을 하여 캡션을 확인할 수 있습니다.
    - Batch Input from TXT File 모드에서는 사용자가 Video_id, Start, End를 tab으로 구분하여 1줄씩 적은 txt를 첨부하면 전체에 대해 배치로 추론을 진행한 후, 전체 결과를 보여줍니다.
- **Frame Captioning (프레임 캡셔닝)**
    - Video_ID(파일명)과 Timestamp(초)를 넣으면 해당 시간의 프레임에 대한 캡션을 보여줍니다.
    - Single Frame Input 모드에서는 사용자가 Video_id, Timestamp를 하나씩 입력하여 캡션을 확인할 수 있습니다.
    - Batch Input from TXT File 모드에서는 사용자가 Video_id, Timestamp를 공백으로 구분하여 1줄씩 적은 txt를 첨부하면 전체에 대해 배치로 추론을 진행한 후, 전체 결과를 보여줍니다.

---

# 파일 및 폴더 구조

```
├── external_videos               # 외부 동영상 파일을 추가하는 폴더 (외부 동영상을 이 폴더에 넣어주세요)
├── original_videos               # youtube 8M 동영상이 저장된 폴더 (미리 1533개에 대해 저장해둔 폴더 - 건드리지 않으셔도 됩니다)
└── vtt
    ├── app.py                    # Streamlit 메인 애플리케이션 스크립트
    ├── init.sh
    ├── config
    |   └── base_config.yaml      # 캡셔닝, 번역, 모델 설정 파일 
    ├── requirements.txt
    ├── temp_save_folder          # 임시 파일(동영상 구간, 프레임 이미지 등) 저장 폴더 - (처리 과정중에 잠시 파일이 저장되는 dummy 폴더입니다.)
    └── utils
		    ├── __init__.py
        ├── LlavaVideo_utils.py
        ├── captioning.py         # 캡셔닝 모델 초기화 및 캡션 생성 함수
        └── vtt_service_utils.py  # 비디오 처리 및 오디오 변환 함수           
```

> 외부 동영상을 사용할 경우,  `external_videos` 폴더에 비디오들을 미리 넣어주어야 캡션 생성이 가능합니다.
> 

---

# 환경 설치

V100 서버 기준으로 작성하였습니다.

평가 시에는 이미 V100 서버에 환경과 1533개 비디오들을 저장해두었으니, 별도의 환경 설치가 필요 없습니다.

## 운영 환경

Linux, cuda 12.2, V100 32GB

## 초기 환경 세팅 (사전 제공한 환경에서는 초기 환경 세팅이 불필요합니다. 아래의 방식을 참고 바랍니다.)

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
    
3. 외부 동영상을 넣을 external_videos 폴더를 만들고, 그 안에 mp4 파일들을 넣어줍니다. (필수)

# 사전 제공 환경 (ssh -p 31347 [root@10.28.224.64](mailto:root@10.28.224.64)에서)

1. vtt 디렉토리로 이동하여 .venv를 실행합니다. 
    
    ```bash
    cd vtt
    source .venv/bin/activate
    ```
    
2. 외부 동영상을 넣을 external_videos 폴더를 만들고, 그 안에 mp4 파일들을 넣어줍니다. (필수)
3. 스트림릿을 실행합니다. 
    
    ```bash
    streamlit run app.py
    ```
    
### **또는 [video to text](http://10.28.224.64:30806/) 에서 확인할 수 있습니다.** (VPN 연결 필수)
---

# 실행 방법

1. 프로젝트 루트 디렉토리에서 터미널(또는 CMD)을 실행합니다.
2. 아래 명령어를 입력하여 Streamlit 애플리케이션을 시작합니다:
    
    ```bash
    streamlit run app.py
    ```
    
3. 기본 웹 브라우저가 열리면서 캡셔닝 인터페이스에 접속할 수 있습니다.

---

# 사용 방법

### 1. Scene Captioning (씬 캡셔닝)

- **단일 입력 모드**
    - **입력 항목**: Video ID(확장자를 제외한 동영상 파일명), 시작 시간(`start`), 종료 시간(`end`)
    - **설명**: 사용자가 입력한 비디오 ID에 해당하는 동영상 파일이 `original_videos` 나 `external_videos`에 존재해야 하며, 지정한 시간 구간의 영상이 캡션 생성 대상이 됩니다.
    - **실행**: "Generate Caption for Single Scene" 버튼 클릭 시, 해당 구간의 캡션을 생성합니다.
- **배치 처리 (TXT 파일 업로드)**
    - **파일 형식**: 각 줄에 `video_id start end` 형식으로 작성합니다.
    - **설명**: 여러 개의 비디오 구간에 대해 일괄적으로 캡셔닝 작업을 진행할 수 있습니다.
    - **주의**: 텍스트 파일에 공백으로 구분된 정확한 형식의 정보가 있어야 올바른 배치 처리가 가능합니다.
        
        **TXT file Example**
        
        ```
        qwkd2lnjnd 1 23
        clqkn13_4lw 24.6 46
        
        ```
        

---

### 2. Frame Captioning (프레임 캡셔닝)

- **단일 입력 모드**
    - **입력 항목**: Video ID, 원하는 timestamp (초 단위)
    - **설명**: 사용자가 입력한 timestamp에 해당하는 프레임을 추출한 후, 캡션을 생성합니다.
    - **실행**: "Generate Frame Caption for Single Frame" 버튼 클릭 시, 해당 프레임의 캡션을 생성합니다.
- **배치 처리 (TXT 파일 업로드)**
    - **파일 형식**: 각 줄에 `video_id timestamp` 형식으로 작성합니다.
    - **설명**: 여러 비디오의 다양한 timestamp에 대해 일괄 캡셔닝 작업을 진행할 수 있습니다.
    - **주의**: 각 줄의 정보가 공백으로 구분된 `video_id timestamp` 형식이어야 정상적으로 처리됩니다.
        
        **TXT file Example**
        
        ```
        qwkd2lnjnd 1
        clqkn13_4lw 24.6
        
        ```
        

> TXT 배치 처리 관련:
> 
> - 씬 캡셔닝의 경우, 텍스트 파일의 각 줄은 `video_id start end` 형식을 따라야 합니다.
> - 프레임 캡셔닝의 경우, 텍스트 파일의 각 줄은 `video_id timestamp` 형식을 따라야 합니다.
> - 입력 형식이 올바르지 않을 경우 오류 메시지가 출력 되며 해당 항목은 건너뛰게 됩니다.

---

## 참고 사항

- **외부 동영상 파일**: 외부 동영상을 사용할 경우,  `external_videos` 폴더에 비디오들을 미리 넣어주어야 합니다. (예: `video_id.mp4` 형태)
- **임시 파일 관리**: 캡셔닝 작업 중 생성된 임시 동영상 구간, 오디오 파일, 프레임 이미지 등은 작업 완료 후 자동으로 삭제됩니다.
- **모델 로딩 및 GPU 메모리**: 페이지 전환 시 사용하지 않는 모델 및 GPU 메모리 캐시가 자동으로 삭제됩니다.
- **번역기 선택**: config 파일(`config/base_config.yaml`)에서 `translator_name` 값을 통해 사용할 번역기를 선택할 수 있으며, DeepL을 선택한 경우 반드시 `DEEPL_API_KEY`를 환경 변수로 입력해야 합니다. (저희 v100 서버에서 번역은 deepl로 설정되어 있습니다.)

---

## 문제 해결 및 지원

- **비디오 파일 미존재**: 입력한 Video ID에 해당하는 비디오 파일이 `external_videos` 나 `original_videos`내에 파일이 존재하는지 확인하세요.
- **시간 형식 오류**: 시작 시간, 종료 시간 또는 timestamp가 숫자 형식인지 확인하세요.
- **DEEPL_API_KEY 오류**: DeepL 번역기를 사용 중이라면 환경 변수에 올바른 API 키가 설정되었는지 확인하세요.

---
