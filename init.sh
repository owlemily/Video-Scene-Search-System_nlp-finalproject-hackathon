#!/bin/bash
apt-get update
apt-get install curl unzip git build-essential -y

pip install --upgrade pip
pip install -r requirements.txt

# Retrieve 테스트 데이터셋 영상 22개 다운로드
chmod +x download_22video.sh
./download_22video.sh