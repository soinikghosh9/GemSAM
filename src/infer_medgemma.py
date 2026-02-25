
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from PIL import Image
import os
import random
import glob
import numpy as np
import pydicom
from src.config import Config

def load_dicom_as_pil(dicom_path):
    """
    Reads a DICOM file and converts it to a PIL Image (RGB).
    """
    ds = pydicom.dcmread(dicom_path)
    pixel_array = ds.pixel_array.astype(float)
    
    # Normalize to 0-255
    # Handle MONOCHROME1 (inverted) if present
    if hasattr(ds, "PhotometricInterpretation") and ds.PhotometricInterpretation == "MONOCHROME1":
        pixel_array = np.max(pixel_array) - pixel_array
        
    pixel_array = (np.maximum(pixel_array, 0) / pixel_array.max()) * 255.0
    pixel_array = np.uint8(pixel_array)
    
    image = Image.fromarray(pixel_array)
    return image.convert("RGB")

def infer():
    print("Loading Fine-Tuned MedGemma...")
    
    # 1. Load Base Model (Quantized same as training)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model_id = Config.MEDGEMMA_ID
    adapter_path = "outputs/medgemma_lora_vindr/final"
    
    try:
        processor = AutoProcessor.from_pretrained(model_id)
    except Exception:
        processor = AutoProcessor.from_pretrained(model_id, use_fast=False)

    
    # Configure Image Token (Same as Training)
    # The training script logic was robust, let's replicate it to be sure
    image_token = "<image>"
    if "<image>" in processor.tokenizer.get_vocab():
        image_token = "<image>"
    elif "<image_soft_token>" in processor.tokenizer.get_vocab():
        image_token = "<image_soft_token>"
    elif hasattr(processor.tokenizer, "boi_token") and processor.tokenizer.boi_token:
        image_token = processor.tokenizer.boi_token
    
    print(f"DEBUG: Using Image Token: '{image_token}'")
    
    # Sync Processor
    processor.boi_token = image_token
    processor.tokenizer.boi_token = image_token
    processor.image_token = image_token
    
    # Check if we need to add token to vocab (if base model didn't have it)
    if image_token not in processor.tokenizer.get_vocab():
         processor.tokenizer.add_tokens([image_token], special_tokens=True)

    print("Loading Base Model...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    
    # Resize embeddings if we added tokens
    model.resize_token_embeddings(len(processor.tokenizer))
    
    print(f"Loading LoRA Adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval() # Ensure eval mode
    
    # 2. Select Test Image (DICOM)
    data_dir = "medical_data/vinbigdata-chest-xray"
    
    # Look for DICOM files
    test_images = glob.glob(f"{data_dir}/test/*.dicom")
    if not test_images:
         test_images = glob.glob(f"{data_dir}/train/*.dicom")
    
    if not test_images:
        print("Error: No images found to test.")
        return

    selected_image_path = random.choice(test_images)
    print(f"Testing on Image: {selected_image_path}")
    
    try:
        if selected_image_path.lower().endswith(".png") or selected_image_path.lower().endswith(".jpg"):
            image = Image.open(selected_image_path).convert("RGB")
        else:
            # Assume DICOM
            image = load_dicom_as_pil(selected_image_path)
    except Exception as e:
        print(f"Failed to load image: {e}")
        return

    # 3. Inference
    user_prompt = "Locate all abnormalities."
    # Ensure strict format: <image>\nUser: ...\nModel: (no space after colon usually preferred by some models, but we trained with it?)
    # Training was: f"{processor.boi_token}\nUser: {prompt}\nModel:"
    prompt = f"{processor.boi_token}\nUser: {user_prompt}\nModel:"
    
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
        padding=True,
        max_length=512,
        truncation=True
    ).to(model.device)
    
    print("Generating response...")
    with torch.inference_mode():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=200,
            do_sample=True, # Allow some creativity to break loops
            temperature=0.2, # Low temp for precision
            repetition_penalty=1.2, # Strong penalty for loops
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id
        )
        
    decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Post-processing to remove prompt and "User:" hallucinations
    print("\n--- Raw Model Output ---")
    print(decoded)
    
    print("\n--- Cleaned Output ---")
    # Split by the prompt end if possible, or just look for "Model:"
    try:
        # The decoded string often contains the full prompt
        # We want everything AFTER "Model:"
        if "Model:" in decoded:
            response = decoded.split("Model:")[-1].strip()
        else:
            response = decoded
            
        # Stop at next "User:" if generated
        if "User:" in response:
            response = response.split("User:")[0].strip()
            
        print(response)
    except Exception as e:
        print(f"Error cleaning output: {e}")
        print(decoded)
    print("--------------------")

if __name__ == "__main__":
    infer()
