# VTT_service 소개
현재 폴더는 평가를 용이하게 하기위한 Streamlit 어플과 필요한 코드들로 구성되어있습니다.  

기존에 config로 여러 모델, 프롬프트를 조정하여 실행할 수 있었던 것과 달리, 평가만을 위해 모델과 프롬프트, 기타 설정들을 모두 고정시켜두었습니다.  

코드도 용이하게 확인할 수 있도록, 기존의 코드를 조정하고 단순화하여 vtt_service_utils.py와 captioning.py에 추가해두었습니다.

V100 환경(램 32GB)라는 제약조건을 지키기 위해, OOM 문제가 발생하지 않도록 max_new_tokens, num_frames, 입력 프롬프트 길이를 모두 적절히 조정하였습니다.

# 비디오 캡셔닝 애플리케이션

이 프로젝트는 영상의 특정 씬(Scene) 또는 프레임(Frame)에 대해 캡션(자막)을 생성하는 Streamlit 기반 애플리케이션입니다. 사용자는 단일 입력 또는 텍스트 파일을 통한 배치 처리 방식으로 캡셔닝을 진행할 수 있습니다.

---

## 주요 기능

- **Scene Captioning (씬 캡셔닝)**
  - 사용자가 입력한 비디오의 시작 시간과 종료 시간을 기준으로 해당 구간의 캡션을 생성합니다.
  - 단일 입력과 배치 처리(txt 파일 업로드) 방식을 지원합니다.

- **Frame Captioning (프레임 캡셔닝)**
  - 지정된 timestamp에서 프레임을 추출하여 해당 프레임의 캡션을 생성합니다.
  - 단일 입력과 배치 처리(txt 파일 업로드) 방식을 지원합니다.

- **모델 로드 및 GPU 메모리 관리**
  - 필요한 모델만 로드하며, 페이지 전환 시 GPU 메모리 캐시를 해제하여 효율적인 자원 관리를 합니다.

- **번역 기능**
  - 캡션 결과를 다른 언어로 번역할 수 있으며, `googletrans` 또는 `deepl` 번역기를 선택하여 사용할 수 있습니다.

---

## 파일 및 폴더 구조

```
├── video_input_folder       # 외부 동영상 파일을 추가하는 폴더
├── temp_save_folder         # 임시 파일(동영상 구간, 프레임 이미지 등) 저장 폴더
├── config
│   └── base_config.yaml     # 캡셔닝, 번역, 모델 설정 파일
├── utils
│   ├── vtt_service_utils.py # 비디오 처리 및 오디오 변환 함수
│   └── captioning.py        # 캡셔닝 모델 초기화 및 캡션 생성 함수
└── app.py                   # Streamlit 메인 애플리케이션 스크립트
```

> **주의**: **외부 동영상을 사용할 경우, 반드시 `video_input_folder` 폴더에 비디오 파일을 추가해야 합니다.**

---

## 환경 설정

1. **Python 버전**: Python 3.8 이상을 권장합니다.
2. **필수 패키지 설치**: 필요한 패키지는 `requirements.txt` 파일에 명시되어 있습니다. 아래 명령어를 통해 설치할 수 있습니다. (가상환경 사용을 추천합니다.)
   ```bash
   pip install -r requirements.txt
   ```
3. **DEEPL_API_KEY 설정**: DeepL 번역기를 사용하려면, API 키를 환경 변수로 등록해야 합니다.
   - **Linux/Mac**:
     ```bash
     export DEEPL_API_KEY=your_deepl_api_key_here
     ```
   - **Windows (CMD)**:
     ```cmd
     set DEEPL_API_KEY=your_deepl_api_key_here
     ```


vtt_service 폴더 내부의 init.sh와 requirements.txt를 이용하여 환경을 설치합니다.
```bash

bash init.sh 

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### 만약, conda를 사용할 경우에는 ffmpeg 경로를 확인해주세요!
```bash
which ffmpeg
/opt/conda/bin/ffmpeg
```
위와 같이 ffmpeg 경로가 conda라면, conda에서 삭제해줍니다. (저희 V100 초기환경에서는 삭제가 필요합니다.)
### Conda에서 지우기
```bash
conda remove ffmpeg
```


---

## 실행 방법

1. 프로젝트 루트 디렉토리에서 터미널(또는 CMD)을 실행합니다.
2. 아래 명령어를 입력하여 Streamlit 애플리케이션을 시작합니다:
   ```bash
   streamlit run app.py
   ```
3. 기본 웹 브라우저가 열리면서 캡셔닝 인터페이스에 접속할 수 있습니다.

---

## 사용 방법

### 1. Scene Captioning (씬 캡셔닝)

- **단일 입력 모드**
  - **입력 항목**: Video ID, 시작 시간(`start`), 종료 시간(`end`)
  - **설명**: 사용자가 입력한 비디오 ID에 해당하는 동영상 파일이 `video_input_folder`에 존재해야 하며, 지정한 시간 구간의 영상이 캡션 생성 대상이 됩니다.
  - **실행**: "Generate Caption for Single Scene" 버튼 클릭 시, 해당 구간의 캡션을 생성합니다.

- **배치 처리 (TXT 파일 업로드)**
  - **파일 형식**: 각 줄에 `video_id start end` 형식으로 작성합니다.
  - **설명**: 여러 개의 비디오 구간에 대해 일괄적으로 캡셔닝 작업을 진행할 수 있습니다.
  - **주의**: 텍스트 파일에 공백으로 구분된 정확한 형식의 정보가 있어야 올바른 배치 처리가 가능합니다.

**TXT file Example**
```txt
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

> **TXT 배치 처리 관련**:  
> - 씬 캡셔닝의 경우, 텍스트 파일의 각 줄은 `video_id start end` 형식을 따라야 합니다.  
> - 프레임 캡셔닝의 경우, 텍스트 파일의 각 줄은 `video_id timestamp` 형식을 따라야 합니다.  
> - 입력 형식이 올바르지 않을 경우 오류 메시지가 출력되며 해당 항목은 건너뛰게 됩니다.

**TXT file Example**
```txt
qwkd2lnjnd 1
clqkn13_4lw 24.6
```

---

## 참고 사항

- **외부 동영상 파일**: 캡셔닝에 사용할 모든 외부 동영상 파일은 반드시 `video_input_folder` 폴더 내에 추가되어 있어야 합니다. (예: `video_id.mp4` 형태)
- **임시 파일 관리**: 캡셔닝 작업 중 생성된 임시 동영상 구간, 오디오 파일, 프레임 이미지 등은 작업 완료 후 자동으로 삭제됩니다.
- **모델 로딩 및 GPU 메모리**: 페이지 전환 시 사용하지 않는 모델 및 GPU 메모리 캐시가 자동으로 해제되므로, 제한된 메모리 환경에서 동작합니다.
- **번역기 선택**: config 파일(`config/base_config.yaml`)에서 `translator_name` 값을 통해 사용할 번역기를 선택할 수 있으며, DeepL을 선택한 경우 반드시 `DEEPL_API_KEY`를 환경 변수로 입력해야 합니다.

---

## 문제 해결 및 지원

- **비디오 파일 미존재**: 입력한 Video ID에 해당하는 비디오 파일이 `video_input_folder` 내에 존재하는지 확인하세요.
- **시간 형식 오류**: 시작 시간, 종료 시간 또는 timestamp가 숫자 형식인지 확인하세요.
- **DEEPL_API_KEY 오류**: DeepL 번역기를 사용 중이라면 환경 변수에 올바른 API 키가 설정되었는지 확인하세요.

---



## 라이선스

이 프로젝트는 [LICENSE](LICENSE) 파일의 조건에 따라 배포됩니다.






