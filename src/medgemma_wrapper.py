import os
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from PIL import Image
import json
import re
import gc
try:
    from .config import Config
    from .preprocess import load_and_preprocess
except (ImportError, ValueError):
    from config import Config
    from preprocess import load_and_preprocess

# Import modality-aware prompts (centralized prompt management)
try:
    try:
        from .prompts.modality_prompts import (
            get_detection_prompt,
            get_screening_prompt,
            get_vqa_context,
            normalize_modality,
            get_normal_findings,
            MODALITY_DETECTION_CLASSIFICATION_PROMPT,
        )
    except (ImportError, ValueError):
        from prompts.modality_prompts import (
            get_detection_prompt,
            get_screening_prompt,
            get_vqa_context,
            normalize_modality,
            get_normal_findings,
            MODALITY_DETECTION_CLASSIFICATION_PROMPT,
        )
    MODALITY_PROMPTS_AVAILABLE = True
except ImportError:
    MODALITY_PROMPTS_AVAILABLE = False
    print("Warning: Modality prompts module not found. Using legacy prompts.")

class MedGemmaWrapper:
    def __init__(self):
        self.processor = None
        self.model = None
        
    def load(self):
        """
        Loads MedGemma. 
        Supports CPU fallback if GPU is incompatible.
        """
        print(f"Loading MedGemma ({Config.MEDGEMMA_ID}) on {Config.DEVICE}...")
        
        # Clear memory before loading
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        quantization_config = None
        if Config.LOAD_IN_4BIT and Config.DEVICE == "cuda":
            print("Mode: 4-bit Quantization (GPU)")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if Config.USE_BF16 else torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:
            print("Mode: Standard Precision (CPU/GPU)")

        try:
             self.processor = AutoProcessor.from_pretrained(Config.MEDGEMMA_ID, local_files_only=True)
        except OSError:
             self.processor = AutoProcessor.from_pretrained(Config.MEDGEMMA_ID)
        
        # Load Model
        try:
             self.model = self._load_model_safe(quantization_config)
        except RuntimeError as e:
             if "out of memory" in str(e):
                 print("(!) Loading OOM Detected. Retrying with aggressive cleanup...")
                 gc.collect()
                 torch.cuda.empty_cache()
                 self.model = self._load_model_safe(quantization_config)
             else:
                 raise e

        # CRITICAL: Setup processor and tokens AFTER model is loaded
        # This was previously dead code (after return statements)
        self._setup_processor_and_tokens()
        print("MedGemma loaded successfully.")

    def _load_model_safe(self, quantization_config):
         # Robust loading with local fallback
         try:
             # Try local first
             return self._load_from_hf(quantization_config, local_only=True)
         except OSError:
             print("  > Local load failed. Downloading from Hub...")
             return self._load_from_hf(quantization_config, local_only=False)
             
    def _load_from_hf(self, quantization_config, local_only):
        """Load model from HuggingFace - returns model only."""
        kwargs = {
            "device_map": {"": "cuda:0"} if torch.cuda.is_available() else "auto",
            "attn_implementation": None,
  # Let transformers pick (SDPA/Flash)
            "low_cpu_mem_usage": True,
            "local_files_only": local_only
        }
        if quantization_config:
            kwargs["quantization_config"] = quantization_config
            return AutoModelForImageTextToText.from_pretrained(Config.MEDGEMMA_ID, **kwargs)
        else:
            model = AutoModelForImageTextToText.from_pretrained(Config.MEDGEMMA_ID, **kwargs)
            # CRITICAL: Only move to device if NOT using auto device_map
            if kwargs.get("device_map") is None:
                model = model.to(Config.DEVICE)
            return model

        def _setup_processor(self, model_path: str, use_base_model: bool = False):
        print("Setting up processor and tokens for inference...")
        base_model = "google/gemma-3-4b-it" 
        try:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(model_path)
            # Patch processor if not default
            target_image_size = 448 # Default to 448
            if target_image_size:
                if hasattr(self.processor, "image_processor"):
                     self.processor.image_processor.size = {"height": target_image_size, "width": target_image_size}

            # Setup tokenizer
            self.tokenizer = self.processor.tokenizer

            # Monkey-patch image token validation
            def _check_special_mm_tokens(sequence): pass
            self.processor._check_special_mm_tokens = _check_special_mm_tokens

        except Exception as e:
            print(f"Error setting up processor: {e}")
            raise

        if use_base_model:
            print("  > Base Model Requested (Zero-Shot Mode).")
            self.model = base_model
        else:
            lora_path = "checkpoints/production/medgemma/final"
            if os.path.exists(lora_path):
                print(f"  > Loading MedGemma LoRA Adapter from {lora_path}...")
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(
                    base_model,
                    lora_path,
                    is_trainable=False
                )
                print("  > LoRA Adapter Loaded Successfully. Model in eval mode.")
            else:
                self.model = base_model
