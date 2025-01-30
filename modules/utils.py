"""
utils.py

함수 목록:
1. load_config
"""

import os

import yaml


def load_config(config_path):
    """
    YAML 설정 파일을 로드하는 함수
    Args:
        config_path (str): 설정 파일 경로 (ex. "../config/config.yaml")

    Returns:
        config (dict): 로드된 설정 파일
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"[ERROR] config_path가 존재하지 않습니다: {config_path}"
        )
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"[ERROR] 유효하지 않은 config 파일입니다: {config_path}\n{e}")


if __name__ == "__main__":
    # load_config 함수 테스트
    config = load_config("../config/fcs_config.yaml")
    print(config)
