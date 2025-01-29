"""
utils.py

함수 목록:
1. load_config
"""

import yaml


def load_config(config_path):
    """
    YAML 설정 파일을 로드하는 함수
    Args:
        config_path (str): 설정 파일 경로 (ex. "../config/config.yaml")

    Returns:
        config (dict): 로드된 설정 파일
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    # load_config 함수 테스트
    config = load_config("../config/fcs_config.yaml")
    print(config)
