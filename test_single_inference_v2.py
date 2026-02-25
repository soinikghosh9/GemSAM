import torch
from PIL import Image
import sys
import os

# Set environment variable to avoid relative import issues during standalone run
sys.path.append('d:/MedGamma/src')

# Patch config if needed
import config
from medgemma_wrapper import MedGemmaWrapper

def test_single_inference():
    wrapper = MedGemmaWrapper()
    print("Loading model...")
    wrapper.load()
    
    # Try to load adapter
    adapter_path = "d:/MedGamma/checkpoints/production/final"
    if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        print(f"Loading adapter: {adapter_path}")
        if hasattr(wrapper.model, "load_adapter"):
            wrapper.model.load_adapter(adapter_path)
            print("Adapter loaded via load_adapter")
        else:
            from peft import PeftModel
            wrapper.model = PeftModel.from_pretrained(wrapper.model, adapter_path)
            print("Adapter loaded via PeftModel")
    
    wrapper.model.eval()
    
    # Path to a known image
    image_path = "d:/MedGamma/medical_data/vinbigdata-chest-xray/test/002a20b22a074094e43f3f2d250f283d.jpg"
    if not os.path.exists(image_path):
        # Fallback to any found image
        import glob
        images = glob.glob("d:/MedGamma/medical_data/vinbigdata-chest-xray/test/*.jpg")
        if images:
            image_path = images[0]
        else:
            print("No images found!")
            return

    print(f"Testing on image: {image_path}")
    
    # Run inference
    print("Running inference...")
    try:
        result = wrapper.analyze_image(image_path, task="detection")
        print("\n--- Output ---")
        print(result)
    except Exception as e:
        print(f"Inference FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_inference()
