import torch
from transformers import AutoModelForImageTextToText
import sys
import os

# Add src to path
sys.path.append('d:/MedGamma/src')
from config import Config

def test_load_adapter():
    print(f"Loading base model: {Config.MEDGEMMA_ID}")
    kwargs = {
        "device_map": "auto",
        "attn_implementation": "eager",
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.float16
    }
    model = AutoModelForImageTextToText.from_pretrained(Config.MEDGEMMA_ID, **kwargs)
    
    adapter_path = "d:/MedGamma/checkpoints/production/final"
    print(f"\nAttempting to load adapter via model.load_adapter: {adapter_path}")
    
    if hasattr(model, "load_adapter"):
        try:
            model.load_adapter(adapter_path)
            print("Successfully loaded adapter via model.load_adapter!")
            return
        except Exception as e:
            print(f"model.load_adapter FAILED: {e}")
    else:
        print("Model does not have load_adapter method.")

    # Try PeftModel but with manual target mapping if needed
    print("\nAttempting PeftModel.from_pretrained (the original failing method):")
    from peft import PeftModel
    try:
        peft_model = PeftModel.from_pretrained(model, adapter_path)
        print("PeftModel.from_pretrained SUCCESS!")
    except Exception as e:
        print(f"PeftModel.from_pretrained FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_load_adapter()
