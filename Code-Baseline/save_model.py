import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from constants.data import (hf_auth, model_id, model_path)
from constants.config import device

def save_model():
    '''Download and save the LLM to disk'''
    print(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, token = hf_auth)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map='cpu', torch_dtype=torch.float16, token = hf_auth)
    
    os.makedirs(model_path, exist_ok=True)

    model.save_pretrained(model_path, from_pt=True) 
    tokenizer.save_pretrained(model_path, from_pt=True)


if __name__ == "__main__":
    save_model()
