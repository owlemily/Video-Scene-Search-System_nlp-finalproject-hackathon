"""
utils.py

함수 목록:
1. load_config
2. translate_caption
"""

import yaml
import deepl
from googletrans import Translator


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

def translate_caption(caption, translator, target_lang="ko"):
    """
    caption을 입력으로 받아 번역된 caption_ko를 반환하는 함수

    Args:
        caption (str): 번역할 캡션
        translator (googletrans.Translator or deepl.Translator): 번역기 객체
        target_lang (str): 번역할 언어 코드 (default: "ko")

    Returns:
        str: 번역된 캡션
    """
    try:
        if isinstance(translator, Translator):  # googletrans 사용
            return translator.translate(caption, dest=target_lang).text
        elif isinstance(translator, deepl.Translator):  # DeepL 사용
            return translator.translate_text(caption, target_lang=target_lang).text
        else:
            raise ValueError("지원되지 않는 번역기 객체입니다.")
    except Exception as e:
        print(f"번역 실패: {caption}. 오류: {e}")
        return ""


if __name__ == "__main__":
    # load_config 함수 테스트
    config = load_config("../config/fcs_config.yaml")
    print(config)

    # translate_caption 함수 테스트 - DeepL 사용
    auth_key = os.environ.get("DEEPL_API_KEY")
    translator = deepl.Translator(auth_key)
    print(translate_caption("Hello, world!", translator, target_lang="ko"))

    # translate_caption 함수 테스트 - Googletrans 사용
    translator = Translator()
    print(translate_caption("Hello, world!", translator, target_lang="ko"))