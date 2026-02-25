import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
import os

model_id = "google/medgemma-1.5-4b-it"
print(f"Loading model: {model_id}")

kwargs = {
    "device_map": "auto",
    "attn_implementation": "eager",
    "torch_dtype": torch.float16
}

model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
processor = AutoProcessor.from_pretrained(model_id)

# Minimal test prompt
prompt = "<start_of_turn>user\nWhat is in this image?<end_of_turn>\n<start_of_turn>model\n"
inputs = processor(text=prompt, images=None, return_tensors="pt").to(model.device)

print("Running minimal generation (text-only)...")
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=20)

print("Generated text:")
print(processor.decode(output[0], skip_special_tokens=True))
print("\nSUCCESS: Base model generation works.")
