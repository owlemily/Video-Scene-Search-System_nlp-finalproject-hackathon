#!/bin/bash
apt-get update
apt-get install curl unzip git build-essential -y

apt-get install ffmpeg libx264-dev -y

# Retrieve 테스트 데이터셋 영상 22개 다운로드
chmod +x download_22video.sh
./download_22video.sh