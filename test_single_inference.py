import torch
from PIL import Image
import sys
import os

# Add src to path
sys.path.append('d:/MedGamma/src')
from medgemma_wrapper import MedGemmaWrapper

def test_single_inference():
    wrapper = MedGemmaWrapper()
    print("Loading model...")
    wrapper.load()
    
    # DO NOT load adapter for now, let's see baseline behavior with refined prompt
    # Since adapter loading is failing with KeyError anyway
    
    image_path = "d:/MedGamma/data/images/000434271f63a053c34568a09042057f.png" # Just a guess, let's find a real one
    if not os.path.exists(image_path):
        # Find first image in data dir
        import glob
        images = glob.glob("d:/MedGamma/**/*.png", recursive=True)
        if images:
            image_path = images[0]
        else:
            print("No images found!")
            return

    print(f"Testing on image: {image_path}")
    
    # Test with baseline refined prompt (no reasoning requested yet)
    result = wrapper.analyze_image(image_path, task="detection")
    print("\n--- Baseline Refined Prompt Output ---")
    print(result)
    
    # Test with reasoning requested
    result_reasoning = wrapper.analyze_image(image_path, task="detection", reasoning=True)
    print("\n--- Reasoning Prompt Output ---")
    print(result_reasoning)

if __name__ == "__main__":
    test_single_inference()
