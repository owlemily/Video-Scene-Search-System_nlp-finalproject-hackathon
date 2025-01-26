from transformers import AutoModel, AutoTokenizer
import torch
import os
token = os.environ['HF_TOKEN']
def initialize_model(model_path='OpenGVLab/InternVideo2-Chat-8B'):
    """
    Initializes the model and tokenizer with given configurations.

    Args:
        model_path (str): The path to the model.
        mm_llm_compress (bool): Whether to enable LLM compression.

    Returns:
        tuple: The initialized model, tokenizer, and image processor.
    """
    tokenizer =  AutoTokenizer.from_pretrained('OpenGVLab/InternVideo2-Chat-8B', trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()


    return model, tokenizer
