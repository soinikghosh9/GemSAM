import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
import sys
import os

# Add src to path
sys.path.append('d:/MedGamma/src')
from config import Config

def inspect_model():
    print(f"Loading base model: {Config.MEDGEMMA_ID}")
    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    kwargs = {
        "device_map": "auto",
        "attn_implementation": "eager",
        "low_cpu_mem_usage": True,
        "quantization_config": quantization_config
    }
    try:
        model = AutoModelForImageTextToText.from_pretrained(Config.MEDGEMMA_ID, **kwargs)
        print("\nBase Model Structure:")
        for name, _ in model.named_children():
            print(f" - {name}")
        
        if hasattr(model, "model"):
            print("\nmodel.model Structure:")
            for name, _ in model.model.named_children():
                print(f"    - {name}")
                
        adapter_path = "d:/MedGamma/checkpoints/production/final"
        print(f"\nAttempting to load adapter from: {adapter_path}")
        peft_model = PeftModel.from_pretrained(model, adapter_path)
        print("Success!")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_model()
