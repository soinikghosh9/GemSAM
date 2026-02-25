import torch
from transformers import AutoModelForImageTextToText
import os

def find_param(m, target, path=''):
    for name, child in m.named_children():
        current_path = f"{path}.{name}" if path else name
        if name == target:
            print(f"FOUND: {current_path} ({child.__class__.__name__})")
        find_param(child, target, current_path)

model_id = "google/medgemma-1.5-4b-it"
print(f"Checking model: {model_id}")
model = AutoModelForImageTextToText.from_pretrained(model_id, device_map='auto', torch_dtype=torch.float16)

print("\nSearching for 'embed_tokens':")
find_param(model, 'embed_tokens')

print("\nSearching for 'language_model':")
find_param(model, 'language_model')
