import os
import torch

class Config:
    # Hardware Settings
    # Hardware Settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LOAD_IN_4BIT = True if DEVICE == "cuda" else False
    USE_BF16 = True if DEVICE == "cuda" and torch.cuda.is_bf16_supported() else False
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

    # Model Paths / IDs
    MEDGEMMA_ID = "google/medgemma-1.5-4b-it" 
    SAM2_CHECKPOINT = os.path.join(BASE_DIR, "checkpoints", "sam2_hiera_tiny.pt")
    
    # Inference Settings
    CONFIDENCE_THRESHOLD = 0.5
    MAX_NEW_TOKENS = 512

    @staticmethod
    def ensure_dirs():
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
