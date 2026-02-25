"""
MedSAM Wrapper - SAM2 Integration for Medical Image Segmentation

Provides segmentation capabilities using SAM2 with optional LoRA adapter
from medical fine-tuning.
"""

import torch
import numpy as np
from PIL import Image
from .config import Config
import os
from peft import PeftModel


class MedSAMWrapper:
    """SAM2 wrapper for medical image segmentation with LoRA adapter support."""

    def __init__(self, adapter_path: str = None):
        """
        Initialize MedSAM wrapper.

        Args:
            adapter_path: Optional path to trained LoRA adapter. If None,
                         will search default locations.
        """
        self.predictor = None
        self.inference_state = None
        self.model = None
        self.adapter_path = adapter_path

    def load(self, adapter_path: str = None):
        """
        Load SAM2 model with optional LoRA adapter.

        Args:
            adapter_path: Override adapter path (takes precedence over __init__ path)
        """
        if self.model is not None:
            return
        
        # Use provided path or fall back to instance path
        adapter_path = adapter_path or self.adapter_path

        print(f"Loading SAM2 ({Config.SAM2_CHECKPOINT})...")
        try:
            from hydra.utils import instantiate
            from omegaconf import OmegaConf
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            # 1. Load Config
            if os.path.exists("sam2_hiera_t.yaml"):
                cfg = OmegaConf.load("sam2_hiera_t.yaml")
                self.model = instantiate(cfg.model, _recursive_=True)
            else:
                # Fallback to build_sam2 from package
                from sam2.build_sam import build_sam2
                self.model = build_sam2("sam2_hiera_t.yaml", Config.SAM2_CHECKPOINT, device=Config.DEVICE)

            # 2. Load Checkpoint
            if os.path.exists(Config.SAM2_CHECKPOINT):
                sd = torch.load(Config.SAM2_CHECKPOINT, map_location="cpu")['model']
                self.model.load_state_dict(sd)
                self.model.to(Config.DEVICE)

            # 3. Load LoRA Adapter if available
            self._load_adapter(adapter_path)

            self.predictor = SAM2ImagePredictor(self.model)
            print("SAM2 Loaded.")
            
        except ImportError:
            print("(!) SAM2 library not found. Running in SIMULATION mode.")
            self.predictor = "SIMULATION"
        except Exception as e:
            print(f"Error loading SAM2: {e}. Running in SIMULATION mode.")
            self.predictor = "SIMULATION"

    def _load_adapter(self, adapter_path: str = None):
        """Load LoRA adapter for SAM2 image encoder."""
        if not self.model:
            return

        # Search paths in order of preference
        search_paths = []
        if adapter_path:
            search_paths.append(adapter_path)

        # Default adapter locations
        search_paths.extend([
            "checkpoints/production/sam2/final",
            "checkpoints/production/sam2/best",
            "outputs/sam2_medical/final",
            "outputs/sam2_medical/best",
            "outputs/sam2_adapter"
        ])

        for path in search_paths:
            if os.path.exists(path) and os.path.exists(os.path.join(path, "adapter_config.json")):
                print(f"  > Loading SAM2 LoRA Adapter from {path}...")
                try:
                    self.model.image_encoder = PeftModel.from_pretrained(
                        self.model.image_encoder,
                        path
                    )
                    print("  > SAM2 Adapter Loaded Successfully.")
                    return
                except Exception as e:
                    print(f"  (!) Failed to load SAM2 adapter: {e}")

        print("  > No SAM2 adapter found. Using base model.")

    def set_image(self, image_path):
        if self.predictor == "SIMULATION":
            return

        if self.predictor:
            image = Image.open(image_path).convert("RGB")
            self.predictor.set_image(np.array(image))

    def predict_mask(self, box):
        """
        box: [x1, y1, x2, y2]
        """
        if not self.predictor:
            return None, None
            
        if self.predictor == "SIMULATION":
            x1, y1, x2, y2 = map(int, box)
            mask = np.zeros((1024, 1024), dtype=bool) 
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(1024, x2), min(1024, y2)
            mask[y1:y2, x1:x2] = True
            # Return 3 dummy masks to match real API
            return np.array([mask, mask, mask]), np.array([0.99, 0.8, 0.7])
        
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array([box]),
            multimask_output=True, 
        )
        return masks, scores
