# VTT_service 소개
현재 폴더는 평가를 용이하게 하기위한 Streamlit 어플과 필요한 코드들로 구성되어있습니다.  

기존에 config로 여러 모델, 프롬프트를 조정하여 실행할 수 있었던 것과 달리, 평가만을 위해 모델과 프롬프트, 기타 설정들을 모두 고정시켜두었습니다.  

코드도 용이하게 확인할 수 있도록, 기존의 코드를 조정하고 단순화하여 vtt_service_utils.py와 captioning.py에 추가해두었습니다.

V100 환경(램 32GB)라는 제약조건을 지키기 위해, OOM 문제가 발생하지 않도록 max_new_tokens, num_frames, 입력 프롬프트 길이를 모두 적절히 조정하였습니다.

# 환경 설치
vtt_service 폴더 내부의 init.sh와 requirements.txt를 이용하여 환경을 설치합니다.
```bash
cd level4-nlp-finalproject-hackathon-nlp-01-lv3/vtt_service

chmod +x init.sh

python -m venv .venv
source .venb/bin/activate

pip install -r requirements.txt
```

## 만약, conda를 사용할 경우에는 ffmpeg 경로를 확인해주세요!
```bash
which ffmpeg
/opt/conda/bin/ffmpeg
```
위와 같이 ffmpeg 경로가 conda라면, conda에서 삭제해줍니다. (저희 V100 초기환경에서는 삭제가 필요합니다.)
### Conda에서 지우기
```bash
conda remove ffmpeg
```