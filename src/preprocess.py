from PIL import Image
import os
import torch
import torchvision.transforms as T
import numpy as np
import pydicom

# --- 1. Custom Transform for Training (Square Padding) ---
class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = (max_wh - w) // 2
        vp = (max_wh - h) // 2
        padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)
        return T.functional.pad(image, padding, 0, 'constant')

# --- 2. Training Transform Factory ---
def get_training_transform(size=1024):
    """
    Standard transform for MedGemma/SigLIP training.
    Uses Square Padding to preserve aspect ratio, matching inference logic.
    """
    return T.Compose([
        SquarePad(),
        T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(), # [0, 255] -> [0.0, 1.0]
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # [-1, 1]
    ])

# --- 3. Inference Preprocessing (Robust Loader) ---
def load_and_preprocess(image_path, target_size=1024, pad_to_square=False):
    """
    Loads an image (support PNG, JPG, DICOM), ensures RGB, and strictly resizes/pads to target_size.
    Handles 16-bit PNGs and DICOMs by normalizing to 8-bit.
    Args:
        pad_to_square: If True, pads the resized image to be exactly target_size x target_size.
    Returns: PIL.Image
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
        
    try:
        # Check for DICOM
        if image_path.lower().endswith('.dicom') or image_path.lower().endswith('.dcm'):
            ds = pydicom.dcmread(image_path)
            pixel_array = ds.pixel_array.astype(float)
            
            # Handle MONOCHROME1 (Inverted)
            if hasattr(ds, "PhotometricInterpretation") and ds.PhotometricInterpretation == "MONOCHROME1":
                pixel_array = np.max(pixel_array) - pixel_array
            
            # Normalize to 0-255
            pixel_array = (np.maximum(pixel_array, 0) / pixel_array.max()) * 255.0
            pixel_array = np.uint8(pixel_array)
            img = Image.fromarray(pixel_array).convert("RGB")
        else:
            # Standard Image Load
            img = Image.open(image_path)
            
            # Handle 16-bit Grayscale (I;16)
            if img.mode == 'I;16':
                img_np = np.array(img)
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
                img_np = (img_np * 255).astype(np.uint8)
                img = Image.fromarray(img_np).convert("RGB")
            else:
                img = img.convert("RGB")
        
        # Resize Logic: Scale longest edge to target_size
        w, h = img.size
        # Handle cases where image is smaller than target (upscale) or larger (downscale)
        scale = target_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # High-quality resize
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        if pad_to_square:
            new_img = Image.new("RGB", (target_size, target_size), (0, 0, 0))
            # Center output
            paste_x = (target_size - new_w) // 2
            paste_y = (target_size - new_h) // 2
            new_img.paste(img, (paste_x, paste_y))
            return new_img
        
        return img
        
    except Exception as e:
        print(f"ERROR: Failed to preprocess image: {e}")
        return None 
