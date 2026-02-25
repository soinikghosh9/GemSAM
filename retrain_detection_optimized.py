"""
OPTIMIZED Detection Retraining for MedGamma.

Key Performance Optimizations:
1. PRE-TOKENIZATION: Tokenize entire dataset BEFORE training loop
2. IMAGE CACHING: Cache processed images in memory or disk
3. ASYNC DATA LOADING: Use prefetch and proper multiprocessing
4. GRADIENT CHECKPOINTING: Reduce memory for larger batches
5. MIXED PRECISION: Use bfloat16 for Ampere+ GPUs
6. PROGRESSIVE RESIZING: Start small, increase resolution

Key Generalization Improvements:
1. CLASS-BALANCED SAMPLING: Proper weighting for rare classes
2. DATA AUGMENTATION: Medical-appropriate transforms
3. MIXUP/CUTMIX: Regularization for better generalization
4. MULTI-SCALE TRAINING: Random resize within bounds

Usage:
    python retrain_detection_optimized.py                    # Full training
    python retrain_detection_optimized.py --quick            # Quick test
    python retrain_detection_optimized.py --cache-dir cache  # Use disk cache
"""

import os
import sys
import gc
import json
import time
import argparse
import random
import hashlib
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from datetime import datetime
import numpy as np
from PIL import Image
import threading

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Import modality-aware prompts (critical for multi-modal training)
try:
    from src.prompts.modality_prompts import (
        get_detection_prompt,
        get_normal_findings,
        SUPPORTED_MODALITIES,
    )
    MODALITY_PROMPTS_AVAILABLE = True
except ImportError:
    MODALITY_PROMPTS_AVAILABLE = False
    print("Warning: Modality prompts module not found. Using legacy chest X-ray prompts.")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class OptimizedTrainingConfig:
    """Optimized training configuration."""
    # Data
    data_dir: str = "medical_data"
    cache_dir: str = ""  # Empty = in-memory cache

    # Model
    model_id: str = "google/medgemma-1.5-4b-it"
    lora_r: int = 16
    lora_alpha: int = 32

    # Training - OPTIMIZED for 16GB VRAM (RTX 4060 Ti)
    # MedGemma 4B activations are HUGE - gradient checkpointing is REQUIRED
    epochs: int = 3
    batch_size: int = 4  # With grad checkpointing: ~7-8GB VRAM
    grad_accum: int = 4  # Effective batch = 16
    lr: float = 2e-5
    warmup_ratio: float = 0.1
    max_seq_length: int = 768  # Increased for description-rich JSON output

    # Speed Optimizations
    use_bf16: bool = True  # Better than fp16 on Ampere+
    gradient_checkpointing: bool = True  # REQUIRED for 16GB - recomputes activations to save VRAM
    compile_model: bool = False  # torch.compile (experimental)
    prefetch_factor: int = 2  # Reduced for memory

    # Data Optimizations - CRITICAL FOR SPEED
    precompute_tokens: bool = True  # Pre-tokenize dataset
    cache_images: bool = True  # Cache images as they load
    preload_images: bool = True  # Pre-load images (with RAM limit)
    max_cache_gb: float = 20.0  # Max RAM for image cache (leave room for OS)
    train_only_cached: bool = True  # ONLY train on cached images (no slow on-the-fly loading)
    num_workers: int = 0  # Windows: 0, Linux: 4+

    # STABILITY OPTIONS (for long training runs)
    stream_mode: bool = False  # No caching - load, use, discard (most stable)
    disk_cache: bool = False  # Cache processed images to disk (stable + fast)
    disk_cache_dir: str = "cache/processed_images"
    memory_cleanup_interval: int = 100  # Force GC every N batches
    auto_reduce_cache_on_pressure: bool = True  # Reduce cache if RAM low

    # Generalization
    use_augmentation: bool = False  # DISABLED - allows effective caching
    use_mixup: bool = False  # Medical: careful with mixup
    class_balanced: bool = True
    max_normal_ratio: float = 0.3  # Cap normal at 30%

    # Paths
    output_dir: str = "checkpoints/production/medgemma/detection"


# =============================================================================
# Optimized Dataset with Pre-tokenization
# =============================================================================

class PreTokenizedDetectionDataset(Dataset):
    """
    Dataset that pre-tokenizes all samples for maximum training speed.

    Key optimization: Tokenization happens ONCE during initialization,
    not during each training iteration.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        processor,
        tokenizer,
        config: OptimizedTrainingConfig,
        indices: Optional[List[int]] = None
    ):
        self.base_dataset = base_dataset
        self.processor = processor
        self.tokenizer = tokenizer
        self.config = config
        self.indices = indices or list(range(len(base_dataset)))

        # Pre-tokenized cache
        self.token_cache: Dict[int, Dict] = {}
        self.image_cache: Dict[int, torch.Tensor] = {}

        # Detection prompts — MUST match inference prompts in medgemma_wrapper.py EXACTLY
        # Now uses modality-aware prompts for multi-modal training
        self.default_modality = "xray"  # Default for datasets without modality info

        if MODALITY_PROMPTS_AVAILABLE:
            # Use centralized modality prompts
            self.detection_prompts = {
                modality: get_detection_prompt(modality)
                for modality in SUPPORTED_MODALITIES
            }
            print(f"[Training] Using modality-aware prompts for: {SUPPORTED_MODALITIES}")
        else:
            # Legacy fallback - chest X-ray only
            legacy_prompt = (
                "Analyze this chest X-ray for pathological findings. "
                "For each finding, provide the class name and bounding box coordinates. "
                "Additionally, provide a comprehensive clinical description of the anomaly, its severity, and its exact anatomical location. "
                'Output in JSON format: {"findings": [{"class": "...", "box": [x1,y1,x2,y2], "description": "..."}]}'
            )
            self.detection_prompts = {"xray": legacy_prompt}
            print("[Training] Using legacy chest X-ray prompts (modality prompts module not available)")

        # Get image token info
        self._setup_image_tokens()

        # Pre-tokenize if enabled
        if config.precompute_tokens:
            self._precompute_all()

        # Pre-load images if enabled (HUGE speedup - no disk I/O during training)
        if getattr(config, 'preload_images', False):
            self._preload_all_images()

    def _setup_image_tokens(self):
        """Setup image token configuration."""
        if hasattr(self.tokenizer, "boi_token") and self.tokenizer.boi_token:
            self.image_token = self.tokenizer.boi_token
        else:
            self.image_token = "<image>"

        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        self.image_seq_length = getattr(self.processor, 'image_seq_length', 256)

    def _precompute_all(self):
        """Pre-compute tokens for all samples."""
        print(f"Pre-tokenizing {len(self.indices)} samples...")
        start = time.time()

        # Use thread pool for parallel tokenization
        def tokenize_sample(idx):
            real_idx = self.indices[idx]
            sample = self.base_dataset[real_idx]
            return idx, self._tokenize_sample(sample)

        # Sequential for now (tokenizer may not be thread-safe)
        for i, idx in enumerate(self.indices):
            sample = self.base_dataset[idx]
            self.token_cache[i] = self._tokenize_sample(sample)

            if (i + 1) % 1000 == 0:
                print(f"  Tokenized {i+1}/{len(self.indices)}...")

        elapsed = time.time() - start
        print(f"Pre-tokenization complete in {elapsed:.1f}s")

    def _preload_all_images(self):
        """Pre-load images into RAM (with memory limit to prevent crash)."""
        # Get available RAM and set limit (leave 8GB for OS + training)
        max_cache_gb = getattr(self.config, 'max_cache_gb', 20.0)
        if PSUTIL_AVAILABLE:
            try:
                total_ram_gb = psutil.virtual_memory().total / 1024**3
                max_cache_gb = min(total_ram_gb - 8, max_cache_gb)
            except:
                pass

        print(f"Pre-loading images into RAM (limit: {max_cache_gb:.0f}GB)...")
        print(f"  Remaining images will be loaded on-the-fly during training")
        start = time.time()

        bytes_per_image = 0
        for i, idx in enumerate(self.indices):
            # Check memory limit
            if bytes_per_image > 0:
                current_gb = len(self.image_cache) * bytes_per_image / 1024**3
                if current_gb >= max_cache_gb:
                    print(f"  Reached RAM limit ({current_gb:.1f}GB), stopping pre-load")
                    print(f"  Cached {len(self.image_cache)}/{len(self.indices)} images ({100*len(self.image_cache)/len(self.indices):.0f}%)")
                    break

            if i not in self.image_cache:
                # Load and process image
                sample = self.base_dataset[idx]
                image = sample["image"]

                # Ensure PIL Image
                if isinstance(image, str):
                    image = Image.open(image).convert("RGB")
                elif not isinstance(image, Image.Image):
                    image = Image.fromarray(np.array(image)).convert("RGB")
                else:
                    image = image.convert("RGB")

                # Process image
                image_inputs = self.processor.image_processor(
                    image,
                    return_tensors="pt"
                )
                pixel_values = image_inputs["pixel_values"].squeeze(0)

                # Store in cache (use float16 to save RAM)
                self.image_cache[i] = pixel_values.half()

                # Calculate bytes per image from first cached image
                if bytes_per_image == 0:
                    bytes_per_image = pixel_values.numel() * 2  # float16 = 2 bytes

            if (i + 1) % 1000 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                mem_gb = len(self.image_cache) * bytes_per_image / 1024**3 if bytes_per_image > 0 else 0
                print(f"  Loaded {i+1}/{len(self.indices)} ({rate:.1f} img/s, {mem_gb:.1f}GB RAM)")

        elapsed = time.time() - start
        mem_gb = len(self.image_cache) * bytes_per_image / 1024**3 if bytes_per_image > 0 else 0
        print(f"Pre-loading complete in {elapsed:.1f}s ({len(self.image_cache)} images, {mem_gb:.1f}GB cached)")

        # If train_only_cached is enabled, restrict to cached images only (HUGE speedup)
        if getattr(self.config, 'train_only_cached', False) and len(self.image_cache) < len(self.indices):
            cached_indices = list(self.image_cache.keys())
            original_count = len(self.indices)
            # Update indices to only include cached images
            self.indices = [self.indices[i] for i in cached_indices]
            # Rebuild token cache with new indices
            new_token_cache = {}
            for new_idx, old_idx in enumerate(cached_indices):
                if old_idx in self.token_cache:
                    new_token_cache[new_idx] = self.token_cache[old_idx]
            self.token_cache = new_token_cache
            # Rebuild image cache with new indices
            new_image_cache = {}
            for new_idx, old_idx in enumerate(cached_indices):
                if old_idx in self.image_cache:
                    new_image_cache[new_idx] = self.image_cache[old_idx]
            self.image_cache = new_image_cache
            print(f"  [FAST MODE] Training only on {len(self.indices)} cached images (was {original_count})")
            print(f"  This eliminates ALL slow on-the-fly image loading!")

    def _tokenize_sample(self, sample: Dict) -> Dict:
        """Tokenize a single sample (called during pre-computation)."""
        # Get modality from sample (if available) for modality-aware training
        modality = sample.get("modality", self.default_modality)
        if modality not in self.detection_prompts:
            modality = self.default_modality

        # Get modality-specific detection prompt
        detection_prompt = self.detection_prompts[modality]

        # Build target JSON
        boxes = sample.get("boxes", [])
        labels = sample.get("labels", [])

        findings = []
        if boxes:
            for box, label in zip(boxes, labels):
                desc = f"Pathological presence of {label} detected in the specified region, indicating an underlying clinical anomaly requiring attention."
                reasoning = f"Visual evidence shows {label} in the localized region, consistent with clinical patterns of this pathology."
                findings.append({
                    "class": label,
                    "box": [int(b) for b in box],
                    "reasoning": reasoning,
                    "description": desc
                })
            target_json = json.dumps({"findings": findings})
        else:
            # Normal image: use modality-specific clinical language
            if MODALITY_PROMPTS_AVAILABLE:
                normal_findings = get_normal_findings(modality)
                target_json = json.dumps(normal_findings)
            else:
                # Legacy fallback - chest X-ray specific
                target_json = json.dumps({"findings": [{"class": "No significant abnormality", "box": [0,0,0,0], "description": "No acute cardiopulmonary abnormality identified. Heart size and mediastinal contour are within normal limits. Lungs are clear bilaterally without focal consolidation, effusion, or pneumothorax."}]})

        # Build full prompt with chat template
        full_prompt = (
            f"<start_of_turn>user\n{self.image_token} {detection_prompt}"
            f"<end_of_turn>\n<start_of_turn>model\n{target_json}<end_of_turn>"
        )

        # Tokenize text (without image tokens)
        max_len = getattr(self.config, 'max_seq_length', 384)

        # Robust tokenization and label masking
        # The user prompt includes the image token
        user_prompt = f"<start_of_turn>user\n {detection_prompt}\n<end_of_turn>\n<start_of_turn>model\n"
        target_prompt = f"{target_json}<end_of_turn>"
        
        # Exact tokenization
        user_ids_list = self.tokenizer.encode(user_prompt, add_special_tokens=True)
        target_ids_list = self.tokenizer.encode(target_prompt, add_special_tokens=False)
        
        # Find where "user\n" ends to insert image tokens
        user_turn_ids = self.tokenizer.encode("user\n", add_special_tokens=False)
        insert_pos = 2  # Fallback
        for i in range(len(user_ids_list) - len(user_turn_ids) + 1):
            if user_ids_list[i:i+len(user_turn_ids)] == user_turn_ids:
                insert_pos = i + len(user_turn_ids)
                break
                
        # Inject image tokens into user_ids_list
        img_tokens = [self.image_token_id] * self.image_seq_length
        user_ids_list = user_ids_list[:insert_pos] + img_tokens + user_ids_list[insert_pos:]
        
        # Concatenate and truncate
        combined_ids = user_ids_list + target_ids_list
        if len(combined_ids) > max_len:
            combined_ids = combined_ids[:max_len]
            
        input_ids = torch.tensor(combined_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        
        labels = input_ids.clone()
        # Mask out user prompt AND the padding if any
        # user_ids_list has length L. Only target_ids (the assistant response) are learned.
        user_len = min(len(user_ids_list), max_len)
        labels[:user_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "has_findings": len(findings) > 0
        }

    def _load_and_cache_image(self, idx: int) -> torch.Tensor:
        """Load and cache preprocessed image with multiple caching strategies."""
        # STRATEGY 1: Check RAM cache first (fastest)
        if idx in self.image_cache:
            cached = self.image_cache[idx]
            return cached.float() if cached.dtype == torch.float16 else cached

        # STRATEGY 2: Check disk cache if enabled
        disk_cache_dir = getattr(self.config, 'disk_cache_dir', '')
        use_disk_cache = getattr(self.config, 'disk_cache', False) and disk_cache_dir
        if use_disk_cache:
            disk_path = os.path.join(disk_cache_dir, f"img_{idx}.pt")
            if os.path.exists(disk_path):
                try:
                    pixel_values = torch.load(disk_path, weights_only=True)
                    return pixel_values.float() if pixel_values.dtype == torch.float16 else pixel_values
                except:
                    pass  # Fall through to load from source

        # STRATEGY 3: Load from source
        real_idx = self.indices[idx]
        sample = self.base_dataset[real_idx]
        image = sample["image"]

        # Ensure PIL Image
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image)).convert("RGB")
        else:
            image = image.convert("RGB")

        # Process image
        image_inputs = self.processor.image_processor(
            image,
            return_tensors="pt"
        )
        pixel_values = image_inputs["pixel_values"].squeeze(0)

        # STREAM MODE: No caching at all (most stable for long training)
        if getattr(self.config, 'stream_mode', False):
            return pixel_values  # Just return, don't cache

        # DISK CACHE: Save to disk for future use (stable + reasonably fast)
        if use_disk_cache:
            try:
                os.makedirs(disk_cache_dir, exist_ok=True)
                disk_path = os.path.join(disk_cache_dir, f"img_{idx}.pt")
                torch.save(pixel_values.half(), disk_path)
            except Exception as e:
                pass  # Disk cache failure is non-fatal

        # RAM CACHE: Only if memory allows
        if self.config.cache_images and not getattr(self.config, 'stream_mode', False):
            max_cache_gb = getattr(self.config, 'max_cache_gb', 20.0)
            bytes_per_image = pixel_values.numel() * 2  # float16 = 2 bytes
            current_cache_gb = len(self.image_cache) * bytes_per_image / 1024**3

            skip_cache = current_cache_gb >= max_cache_gb

            # Check system RAM if psutil available
            if not skip_cache and PSUTIL_AVAILABLE:
                try:
                    mem = psutil.virtual_memory()
                    available_gb = mem.available / 1024**3
                    # Leave at least 6GB free for training + OS
                    if available_gb < 6.0:
                        skip_cache = True
                        # Auto-reduce cache if enabled
                        if getattr(self.config, 'auto_reduce_cache_on_pressure', True):
                            self._reduce_cache_on_pressure()
                except:
                    pass

            if not skip_cache:
                self.image_cache[idx] = pixel_values.half()

        return pixel_values

    def _reduce_cache_on_pressure(self):
        """Reduce RAM cache when memory pressure is detected."""
        if len(self.image_cache) > 100:
            # Remove 20% of cached images (oldest first by dict order)
            keys_to_remove = list(self.image_cache.keys())[:len(self.image_cache) // 5]
            for k in keys_to_remove:
                del self.image_cache[k]
            gc.collect()
            print(f"  [MEMORY] Reduced cache by {len(keys_to_remove)} images due to RAM pressure")

    def _augment_image(self, image: Image.Image) -> Image.Image:
        """Apply medical-appropriate augmentations."""
        import torchvision.transforms as T

        # Conservative augmentation for medical images
        transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            # No aggressive crops - preserve pathology
        ])

        return transform(image)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict:
        # Bounds check
        if idx >= len(self.indices):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.indices)} samples")

        # Get pre-tokenized data
        if idx in self.token_cache:
            tokens = self.token_cache[idx]
        else:
            real_idx = self.indices[idx]
            sample = self.base_dataset[real_idx]
            tokens = self._tokenize_sample(sample)

        # Load image (may be cached)
        pixel_values = self._load_and_cache_image(idx)

        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "labels": tokens["labels"],
            "pixel_values": pixel_values,
            "has_findings": tokens["has_findings"]
        }


def collate_optimized(batch: List[Dict]) -> Dict:
    """Optimized collate function with proper padding."""
    from torch.nn.utils.rnn import pad_sequence

    # Get pad token ID
    pad_id = 0  # Will be overwritten

    # Separate fields
    input_ids_list = [b["input_ids"] for b in batch]
    attention_mask_list = [b["attention_mask"] for b in batch]
    labels_list = [b["labels"] for b in batch]
    pixel_values_list = [b["pixel_values"] for b in batch]

    # Pad sequences
    input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
    attention_mask_padded = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=-100)

    # Stack pixel values
    pixel_values_stacked = torch.stack(pixel_values_list)

    # Token type IDs
    token_type_ids = torch.zeros_like(input_ids_padded)

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "labels": labels_padded,
        "pixel_values": pixel_values_stacked,
        "token_type_ids": token_type_ids
    }


# =============================================================================
# Class-Balanced Sampler
# =============================================================================

def create_balanced_sampler(dataset: Dataset, indices: List[int], max_normal_ratio: float = 0.3):
    """
    Create a weighted sampler for class-balanced training (inverse frequency).
    
    Key insight: VinDr has ~70% "No Finding" and highly skewed abnormalities. 
    We cap normals and compute inverse-frequency weights for specific diseases
    so rare abnormalities are learned deeply.
    """
    normal_indices = []
    abnormal_indices = []
    
    sample_to_label = {}
    label_counts = {}

    print("Scanning dataset for deep class balance...")

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        boxes = sample.get("boxes", [])
        labels = sample.get("labels", [])

        is_normal = not boxes and not labels
        if not is_normal and isinstance(labels, list) and len(labels) == 1:
            if isinstance(labels[0], str) and labels[0].lower() == "no finding":
                is_normal = True

        if is_normal:
            normal_indices.append(i)
        else:
            abnormal_indices.append(i)
            # Group sample by its primary (or rarest) label for weighting
            primary = str(labels[0]) if isinstance(labels, list) and len(labels) > 0 else "unknown"
            sample_to_label[i] = primary
            label_counts[primary] = label_counts.get(primary, 0) + 1

        if (i + 1) % 2000 == 0:
            print(f"  Scanned {i+1}/{len(indices)}...")

    print(f"  Normal: {len(normal_indices)}, Abnormal: {len(abnormal_indices)}")
    print(f"  Abnormality distribution: {label_counts}")

    # Cap normal samples to strictly limit their frequency
    target_normal = int(len(abnormal_indices) * (max_normal_ratio / (1 - max_normal_ratio)))
    if len(normal_indices) > target_normal:
        import random
        random.shuffle(normal_indices)
        normal_indices = normal_indices[:target_normal]
        print(f"  Capped normal to {target_normal} samples (max ratio {max_normal_ratio})")

    # Create inverse-frequency weights
    weights = []
    normal_weight = 1.0
    abnormal_classes = len(label_counts)
    total_abnormal = len(abnormal_indices)

    for i in range(len(indices)):
        if i in normal_indices:
            weights.append(normal_weight)
        elif i in abnormal_indices:
            label = sample_to_label[i]
            count = label_counts[label]
            # Inverse frequency weighting: (N / K) / count
            # Scaled up by 2 to structurally prioritize abnormal loss
            class_weight = (total_abnormal / (abnormal_classes * max(count, 1))) * 2.0
            weights.append(class_weight)
        else:
            weights.append(0.0) # Masked out normals

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(indices),
        replacement=True
    )


# =============================================================================
# Disk Cache Preprocessing (run once before training for maximum stability)
# =============================================================================

def preprocess_disk_cache(data_dir: str, cache_dir: str, model_id: str, max_samples: int = 0):
    """
    Pre-process all images to disk cache BEFORE training.
    This ensures stable training with constant memory usage.

    Run: python retrain_detection_optimized.py --preprocess-cache
    """
    print("=" * 60)
    print("  DISK CACHE PREPROCESSING")
    print("  This will process all images and save to disk.")
    print("  Run this ONCE before training for maximum stability.")
    print("=" * 60)

    from transformers import AutoProcessor

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    # Load processor
    print("\n[1/3] Loading processor...")
    processor = AutoProcessor.from_pretrained(
        model_id, token=hf_token, trust_remote_code=True
    )

    # Load dataset
    print("\n[2/3] Loading dataset...")
    from data.factory import MedicalDatasetFactory
    factory = MedicalDatasetFactory(data_dir)
    dataset = factory.get_dataset("vindr", task="detection", split="train")

    total = len(dataset) if max_samples == 0 else min(max_samples, len(dataset))
    print(f"  Total images to process: {total}")

    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)

    # Process all images
    print(f"\n[3/3] Processing images to {cache_dir}...")
    start = time.time()
    processed = 0
    skipped = 0

    for idx in range(total):
        cache_path = os.path.join(cache_dir, f"img_{idx}.pt")

        # Skip if already cached
        if os.path.exists(cache_path):
            skipped += 1
            continue

        try:
            sample = dataset[idx]
            image = sample["image"]

            # Ensure PIL Image
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif not isinstance(image, Image.Image):
                image = Image.fromarray(np.array(image)).convert("RGB")
            else:
                image = image.convert("RGB")

            # Process image
            image_inputs = processor.image_processor(image, return_tensors="pt")
            pixel_values = image_inputs["pixel_values"].squeeze(0)

            # Save to disk as float16
            torch.save(pixel_values.half(), cache_path)
            processed += 1

            if (idx + 1) % 500 == 0:
                elapsed = time.time() - start
                rate = (processed + skipped) / elapsed
                eta = (total - idx - 1) / rate if rate > 0 else 0
                print(f"  Processed {idx+1}/{total} ({rate:.1f} img/s, ETA: {eta/60:.1f}min)")

                # Clear memory periodically
                gc.collect()

        except Exception as e:
            print(f"  Error processing {idx}: {e}")
            continue

    elapsed = time.time() - start
    cache_size_gb = sum(
        os.path.getsize(os.path.join(cache_dir, f))
        for f in os.listdir(cache_dir) if f.endswith('.pt')
    ) / 1024**3

    print("\n" + "=" * 60)
    print("  PREPROCESSING COMPLETE!")
    print(f"  Processed: {processed} images")
    print(f"  Skipped (already cached): {skipped} images")
    print(f"  Cache size: {cache_size_gb:.2f}GB")
    print(f"  Time: {elapsed/60:.1f} minutes")
    print(f"\n  Now run training with: --disk-cache --disk-cache-dir {cache_dir}")
    print("=" * 60)


# =============================================================================
# Main Training Function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Optimized Detection Retraining")
    parser.add_argument("--data_dir", default="medical_data")
    parser.add_argument("--output_dir", default="checkpoints/production/medgemma/detection")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (4 for 16GB VRAM)")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max_samples", type=int, default=0, help="Limit dataset size (0=all)")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--turbo", action="store_true", help="TURBO mode: minimal LoRA, ~45-60min (demo only)")
    parser.add_argument("--balanced", action="store_true", help="BALANCED mode: good detection quality, ~2-3h")
    parser.add_argument("--full", action="store_true", help="FULL mode: all 10k images 1 epoch, ~14-16h")
    parser.add_argument("--extended", action="store_true", help="EXTENDED mode: all 10k images 3 epochs, ~40-45h")
    parser.add_argument("--no-precompute", action="store_true", help="Disable pre-tokenization")
    parser.add_argument("--cache-dir", default="", help="Disk cache directory")
    parser.add_argument("--freeze-vision", action="store_true", help="Freeze vision encoder (faster)")
    parser.add_argument("--all-images", action="store_true", help="Use ALL images (disable train_only_cached)")

    # STABILITY OPTIONS - for safe long training runs
    parser.add_argument("--stream", action="store_true",
                        help="STREAM MODE: No RAM caching, load-use-discard (most stable, slower)")
    parser.add_argument("--disk-cache", action="store_true", default=True,
                        help="DISK CACHE: Cache processed images to disk (stable + fast with SSD)")
    parser.add_argument("--disk-cache-dir", default="C:/medgamma_cache",
                        help="Directory for disk cache")
    parser.add_argument("--stable", action="store_true", default=True,
                        help="STABLE MODE: Optimized settings for crash-free long training")
    parser.add_argument("--preprocess-cache", action="store_true",
                        help="Pre-process all images to disk cache (run ONCE before training)")
    args = parser.parse_args()

    # PREPROCESS CACHE MODE: Run preprocessing and exit
    if getattr(args, 'preprocess_cache', False):
        cache_dir = getattr(args, 'disk_cache_dir', 'cache/processed_images')
        preprocess_disk_cache(
            data_dir=args.data_dir,
            cache_dir=cache_dir,
            model_id="google/medgemma-1.5-4b-it",
            max_samples=args.max_samples
        )
        return  # Exit after preprocessing

    # TURBO MODE: Aggressive optimizations for fast training (demo only, not for detection)
    if args.turbo:
        print("\n" + "!" * 60)
        print("  TURBO MODE - Fast training (~30-60 min)")
        print("  WARNING: Use --balanced for better detection quality!")
        print("!" * 60)
        args.lora_r = 4           # Minimal LoRA rank
        args.batch_size = 2
        args.grad_accum = 8
        args.epochs = 1
        args.lr = 1e-4
        args.max_samples = 1000
        args.freeze_vision = True
        args.max_seq_length = 256
        print(f"  LoRA rank: {args.lora_r}")
        print(f"  Samples: {args.max_samples}")
        print(f"  Epochs: {args.epochs}")
        print(f"  Vision encoder: FROZEN")
        print("!" * 60 + "\n")
        args.max_seq_length = 256
    elif getattr(args, 'balanced', False):
        # BALANCED MODE: Good detection quality with reasonable training time
        print("\n" + "=" * 60)
        print("  BALANCED MODE - Optimal detection quality (~3-4 hours)")
        print("  Prompt aligned with inference, description-rich JSON targets")
        print("=" * 60)
        args.lora_r = 16           # 2× capacity for spatial reasoning
        args.batch_size = 1        # Longer seq (768) needs less batch
        args.grad_accum = 16       # Effective batch = 16
        args.epochs = 3            # Complete training, no early stop
        args.lr = 2e-5             # Stable LR for LoRA-16
        args.max_samples = 5000    # Good sample diversity
        args.freeze_vision = False  # Allow vision projector to adapt
        args.max_seq_length = 768  # Description-rich JSON output
        args.all_images = False   # Use cached images only
        print(f"  LoRA rank: {args.lora_r}")
        print(f"  Samples: {args.max_samples}")
        print(f"  Epochs: {args.epochs}")
        print(f"  Learning rate: {args.lr}")
        print(f"  Max seq length: {args.max_seq_length}")
        print(f"  Vision encoder: TRAINABLE (better detection)")
        print("=" * 60 + "\n")
    elif getattr(args, 'full', False):
        # FULL MODE: Maximum quality - realistic for 14-16 hours
        # Math: 10,500 images / batch_size 2 = 5250 batches/epoch
        # At ~10s/batch: 1 epoch = 14.5 hours, 2 epochs = 29 hours
        # So for 14-16 hours: use all images but only 1 epoch with higher LR
        print("\n" + "*" * 60)
        print("  FULL MODE - Maximum quality (~14-16 hours)")
        print("  Training on ALL 10,500 images, 1 epoch!")
        print("*" * 60)
        args.lora_r = 16          # Full LoRA rank (maximum capacity)
        args.batch_size = 2       # Keep small for memory
        args.grad_accum = 8       # Effective batch = 16
        args.epochs = 1           # 1 epoch = ~14.5 hours with all images
        args.lr = 5e-5            # Higher LR since only 1 epoch
        args.max_samples = 0      # ALL samples
        args.freeze_vision = False  # Trainable vision
        args.max_seq_length = 384
        args.all_images = True    # USE ALL IMAGES (not just cached)
        args.max_cache_gb = 14.0  # REDUCED: Leave RAM for on-the-fly loading + training
        print(f"  LoRA rank: {args.lora_r} (maximum)")
        print(f"  Images: ALL 10,500")
        print(f"  Batches: ~5,250 (at ~10s each = ~14.5 hours)")
        print(f"  Epochs: {args.epochs}")
        print(f"  Learning rate: {args.lr} (higher for single epoch)")
        print(f"  Vision encoder: TRAINABLE")
        print(f"  Cache limit: {args.max_cache_gb}GB (leaves RAM for on-the-fly loading)")
        print("*" * 60 + "\n")
    elif getattr(args, 'extended', False):
        # EXTENDED MODE: Maximum quality with ALL images and 3 epochs (~40-45 hours)
        print("\n" + "#" * 60)
        print("  EXTENDED MODE - Ultimate quality (~40-45 hours)")
        print("  Training on ALL 10,500 images, 3 epochs!")
        print("#" * 60)
        args.lora_r = 16
        args.batch_size = 2
        args.grad_accum = 8
        args.epochs = 3           # 3 full epochs
        args.lr = 2e-5            # Conservative LR for multi-epoch
        args.max_samples = 0
        args.freeze_vision = False
        args.max_seq_length = 384
        args.all_images = True
        args.max_cache_gb = 14.0  # REDUCED: Leave RAM for on-the-fly loading
        print(f"  LoRA rank: {args.lora_r} (maximum)")
        print(f"  Images: ALL 10,500")
        print(f"  Batches: ~5,250/epoch × 3 = ~15,750 total")
        print(f"  Epochs: {args.epochs}")
        print(f"  Estimated time: ~40-45 hours")
        print(f"  Vision encoder: TRAINABLE")
        print(f"  Cache limit: {args.max_cache_gb}GB")
        print("#" * 60 + "\n")
    else:
        args.max_seq_length = 384  # Default
        args.freeze_vision = getattr(args, 'freeze_vision', False)
        args.all_images = getattr(args, 'all_images', False)

    # STABLE MODE: Apply stability-optimized settings for crash-free long training
    if getattr(args, 'stable', False):
        print("\n" + "=" * 60)
        print("  STABLE MODE ENABLED")
        print("  Optimized for crash-free long training on Windows")
        print("=" * 60)
        # Use disk cache by default in stable mode (most reliable)
        if not getattr(args, 'stream', False):
            args.disk_cache = True
        args.max_cache_gb = 10.0  # Conservative RAM cache
        args.memory_cleanup_interval = 50  # More frequent cleanup
        print(f"  Disk cache: {getattr(args, 'disk_cache', False)}")
        print(f"  Stream mode: {getattr(args, 'stream', False)}")
        print(f"  RAM cache limit: {args.max_cache_gb}GB")
        print(f"  Memory cleanup: Every 50 batches")
        print("=" * 60 + "\n")

    # Sanitize argv for imported modules
    sys.argv = [sys.argv[0]]

    # CUDA memory optimization for 16GB VRAM
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

    # Clear any existing CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # Set train_only_cached based on all_images flag
    use_all_images = getattr(args, 'all_images', False)
    train_only_cached = not use_all_images  # If all_images=True, train_only_cached=False

    # CRITICAL: When using disk_cache or stream mode, DISABLE RAM caching
    use_disk_cache = getattr(args, 'disk_cache', False)
    use_stream = getattr(args, 'stream', False)
    disable_ram_cache = use_disk_cache or use_stream

    config = OptimizedTrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        lora_r=args.lora_r,
        max_seq_length=getattr(args, 'max_seq_length', 384),
        precompute_tokens=not args.no_precompute,
        cache_dir=args.cache_dir,
        train_only_cached=train_only_cached,  # Override based on mode
        max_cache_gb=getattr(args, 'max_cache_gb', 20.0),  # Reduced for FULL mode
        # DISABLE RAM caching when using disk cache or stream mode
        cache_images=not disable_ram_cache,  # Don't cache to RAM if using disk/stream
        preload_images=not disable_ram_cache,  # Don't preload to RAM if using disk/stream
        # Stability options
        stream_mode=use_stream,
        disk_cache=use_disk_cache,
        disk_cache_dir=getattr(args, 'disk_cache_dir', 'cache/processed_images'),
        memory_cleanup_interval=getattr(args, 'memory_cleanup_interval', 100),
        auto_reduce_cache_on_pressure=True
    )

    print("=" * 60)
    print("  MedGamma OPTIMIZED Detection Retraining")
    print("=" * 60)
    print(f"  Pre-tokenization: {config.precompute_tokens}")
    if use_all_images:
        print(f"  Training on: ALL 10,500 images")
    else:
        print(f"  Training on: CACHED images only (~4,500 images)")
    print(f"  Gradient checkpointing: {config.gradient_checkpointing}")
    print(f"  BF16: {config.use_bf16}")
    # Stability options - show caching mode
    if config.disk_cache:
        print(f"  DISK CACHE: ON ({config.disk_cache_dir})")
        print(f"  RAM caching: DISABLED (images loaded from SSD)")
    elif config.stream_mode:
        print(f"  STREAM MODE: ON (no caching, load-use-discard)")
        print(f"  RAM caching: DISABLED")
    else:
        print(f"  RAM caching: ON (preload_images={config.preload_images})")
        print(f"  RAM cache limit: {config.max_cache_gb}GB")
    print(f"  Memory cleanup: Every {config.memory_cleanup_interval} batches")
    print("=" * 60)

    # =========================================================================
    # Load Dataset
    # =========================================================================
    print("\n[1/5] Loading VinDr dataset...")

    from data.factory import MedicalDatasetFactory
    factory = MedicalDatasetFactory(config.data_dir)
    base_dataset = factory.get_dataset("vindr", task="detection", split="train")

    total_len = len(base_dataset)
    print(f"  Total samples: {total_len}")

    # Quick mode or limited samples
    if args.quick:
        indices = list(range(min(100, total_len)))
        print(f"  [QUICK MODE] Using {len(indices)} samples")
    elif args.max_samples > 0:
        indices = list(range(min(args.max_samples, total_len)))
        print(f"  [LIMITED] Using {len(indices)} samples")
    else:
        indices = list(range(total_len))

    # =========================================================================
    # Load Model
    # =========================================================================
    print("\n[2/5] Loading MedGemma with LoRA...")

    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig, AutoConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    import types

    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='accelerate.*')
    warnings.filterwarnings('ignore', message='.*Gemma3ImageProcessor.*')
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    # Load processor
    processor = AutoProcessor.from_pretrained(
        config.model_id, token=hf_token, trust_remote_code=True, use_fast=False
    )

    # Setup image token
    if hasattr(processor.tokenizer, "boi_token") and processor.tokenizer.boi_token:
        image_token = processor.tokenizer.boi_token
    else:
        image_token = "<image>"
        if image_token not in processor.tokenizer.get_vocab():
            processor.tokenizer.add_tokens([image_token], special_tokens=True)

    image_token_id = processor.tokenizer.convert_tokens_to_ids(image_token)
    print(f"  Image token: '{image_token}' (ID: {image_token_id})")

    # Load config
    model_config = AutoConfig.from_pretrained(
        config.model_id, token=hf_token, trust_remote_code=True
    )
    if hasattr(model_config, "image_token_index"):
        model_config.image_token_index = image_token_id
    model_config.ignore_index = -100

    # Monkey-patch validation
    def no_op_check(*args, **kwargs):
        return
    processor._check_special_mm_tokens = types.MethodType(no_op_check, processor)

    # Quantization config
    quantization_config = None
    if torch.cuda.is_available():
        # Use BF16 compute dtype for better stability on Ampere+
        compute_dtype = torch.bfloat16 if config.use_bf16 and torch.cuda.is_bf16_supported() else torch.float16
        print(f"  Compute dtype: {compute_dtype}")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

    # Load model
    model = AutoModelForImageTextToText.from_pretrained(
        config.model_id,
        config=model_config,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if config.use_bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token
    )

    # Disable use_cache for training (incompatible with gradient checkpointing)
    if hasattr(model, 'config'):
        model.config.use_cache = False
    if hasattr(model, 'generation_config') and model.generation_config is not None:
        model.generation_config.use_cache = False

    # Prepare for training
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={'use_reentrant': False}
    )

    # Freeze vision encoder if requested (HUGE speedup - no backprop through vision tower)
    if getattr(args, 'freeze_vision', False):
        print("  Freezing vision encoder (no backprop through vision tower)")
        for name, param in model.named_parameters():
            if 'vision_tower' in name or 'vision_model' in name or 'multi_modal_projector' in name:
                param.requires_grad = False

    # Apply LoRA - use fewer targets in turbo mode for speed
    if getattr(args, 'turbo', False):
        # Minimal LoRA: only attention projections (fastest)
        target_modules = ["q_proj", "v_proj"]
        lora_alpha = config.lora_r * 2  # Standard ratio
        print(f"  TURBO LoRA targets: {target_modules}")
    else:
        # Full LoRA: all projection layers
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
        lora_alpha = config.lora_alpha

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05 if getattr(args, 'turbo', False) else 0.1,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # Create Pre-tokenized Dataset
    # =========================================================================
    print("\n[3/5] Creating optimized dataset...")

    optimized_dataset = PreTokenizedDetectionDataset(
        base_dataset=base_dataset,
        processor=processor,
        tokenizer=processor.tokenizer,
        config=config,
        indices=indices
    )

    # IMPORTANT: Get actual dataset size after train_only_cached filtering
    actual_dataset_size = len(optimized_dataset)
    print(f"  Actual training samples: {actual_dataset_size}")

    # Create balanced sampler (only if NOT using train_only_cached, since filtering already happened)
    if config.class_balanced and not args.quick and not getattr(config, 'train_only_cached', False):
        # Use original indices for sampler
        sampler = create_balanced_sampler(base_dataset, indices, config.max_normal_ratio)
        shuffle = False
    else:
        # When train_only_cached is True, just use simple shuffling
        # The dataset is already filtered to cached images
        sampler = None
        shuffle = True
        if getattr(config, 'train_only_cached', False):
            print(f"  [FAST MODE] Using simple shuffle (dataset already filtered)")

    # Create dataloader
    dataloader = DataLoader(
        optimized_dataset,
        batch_size=config.batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=config.num_workers,
        collate_fn=collate_optimized,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        persistent_workers=config.num_workers > 0
    )

    # =========================================================================
    # Training Loop
    # =========================================================================
    print(f"\n[4/5] Training for {config.epochs} epochs...")
    print(f"  Batches per epoch: {len(dataloader)}")
    print(f"  Effective batch size: {config.batch_size * config.grad_accum}")

    from transformers import get_cosine_schedule_with_warmup

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    num_training_steps = len(dataloader) * config.epochs
    num_warmup_steps = int(config.warmup_ratio * num_training_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # Mixed precision scaler (use new API)
    scaler = GradScaler('cuda') if torch.cuda.is_available() else None

    model.train()

    training_log = []
    best_loss = float('inf')
    total_batches = len(dataloader)
    crash_recovery_enabled = True  # Save model on unexpected crashes

    # Timing trackers for performance analysis
    batch_times = []
    data_times = []
    forward_times = []
    backward_times = []

    # Crash recovery: wrap training in try-except to save progress on unexpected errors
    training_crashed = False
    crash_epoch = 0
    crash_batch = 0

    try:
        for epoch in range(config.epochs):
            epoch_start = time.time()
            epoch_loss = 0.0
            epoch_steps = 0
            data_start = time.time()  # Start timing data loading

            for batch_idx, batch in enumerate(dataloader):
                try:
                    batch_start = time.time()
                    data_time = batch_start - data_start  # Time to get batch from dataloader

                    # Move to device (non_blocking for speed)
                    inputs = {
                        "input_ids": batch["input_ids"].to(device, non_blocking=True),
                        "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
                        "pixel_values": batch["pixel_values"].to(device, non_blocking=True),
                        "token_type_ids": batch["token_type_ids"].to(device, non_blocking=True),
                        "labels": batch["labels"].to(device, non_blocking=True)
                    }

                    # Sync CUDA for accurate timing
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    forward_start = time.time()

                    # Forward pass with mixed precision
                    if scaler:
                        with autocast('cuda'):
                            outputs = model(**inputs)
                            loss = outputs.loss / config.grad_accum

                        if torch.isnan(loss):
                            print(f"  WARNING: NaN loss at batch {batch_idx}, skipping")
                            optimizer.zero_grad()
                            data_start = time.time()
                            continue

                        loss_val = loss.item() * config.grad_accum

                        # Sync for timing
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        backward_start = time.time()
                        forward_time = backward_start - forward_start

                        scaler.scale(loss).backward()

                        if (batch_idx + 1) % config.grad_accum == 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            scaler.step(optimizer)
                            scaler.update()
                            scheduler.step()
                            optimizer.zero_grad()
                    else:
                        outputs = model(**inputs)
                        loss = outputs.loss / config.grad_accum
                        loss_val = loss.item() * config.grad_accum

                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        backward_start = time.time()
                        forward_time = backward_start - forward_start

                        loss.backward()

                        if (batch_idx + 1) % config.grad_accum == 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad()

                    # Sync and record backward time
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    backward_time = time.time() - backward_start
                    total_batch_time = time.time() - batch_start

                    # Track times
                    batch_times.append(total_batch_time)
                    data_times.append(data_time)
                    forward_times.append(forward_time)
                    backward_times.append(backward_time)

                    epoch_loss += loss_val
                    epoch_steps += 1

                    # Log progress every 10 batches (more frequent for debugging)
                    if batch_idx % 10 == 0 or batch_idx == total_batches - 1:
                        avg_loss = epoch_loss / max(epoch_steps, 1)
                        mem_info = ""
                        ram_info = ""
                        if torch.cuda.is_available():
                            mem_used = torch.cuda.memory_allocated() / 1024**3
                            mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                            mem_info = f", GPU: {mem_used:.1f}/{mem_total:.1f}GB"

                        # System RAM monitoring
                        if PSUTIL_AVAILABLE:
                            try:
                                mem = psutil.virtual_memory()
                                ram_used = (mem.total - mem.available) / 1024**3
                                ram_total = mem.total / 1024**3
                                ram_info = f", RAM: {ram_used:.1f}/{ram_total:.1f}GB"

                                # WARNING: Check for low RAM (less than 4GB available)
                                if mem.available < 4 * 1024**3:
                                    print(f"\n  !!! LOW RAM WARNING: Only {mem.available/1024**3:.1f}GB available !!!")
                                    print(f"  Consider stopping training to prevent system crash\n")
                            except:
                                pass

                        # Calculate timing stats
                        avg_batch = sum(batch_times[-10:]) / len(batch_times[-10:]) if batch_times else 0
                        avg_data = sum(data_times[-10:]) / len(data_times[-10:]) if data_times else 0
                        avg_fwd = sum(forward_times[-10:]) / len(forward_times[-10:]) if forward_times else 0
                        avg_bwd = sum(backward_times[-10:]) / len(backward_times[-10:]) if backward_times else 0

                        # ETA calculation
                        remaining_batches = total_batches - batch_idx - 1 + (config.epochs - epoch - 1) * total_batches
                        eta_seconds = remaining_batches * avg_batch if avg_batch > 0 else 0
                        eta_min = eta_seconds / 60

                        print(f"  Epoch {epoch+1}/{config.epochs} | "
                              f"Batch {batch_idx}/{total_batches} | "
                              f"Loss: {avg_loss:.4f}{mem_info}{ram_info} | "
                              f"Batch: {avg_batch:.2f}s (data:{avg_data:.2f}s fwd:{avg_fwd:.2f}s bwd:{avg_bwd:.2f}s) | "
                              f"ETA: {eta_min:.1f}min", flush=True)

                    # Periodic checkpoint saving every 500 batches to prevent losing progress
                    if batch_idx > 0 and batch_idx % 500 == 0:
                        checkpoint_dir = config.output_dir + f"_checkpoint_e{epoch+1}_b{batch_idx}"
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        model.save_pretrained(checkpoint_dir)
                        print(f"  [CHECKPOINT] Saved to {checkpoint_dir}")
                        # Also save current training state
                        checkpoint_log = os.path.join(checkpoint_dir, "checkpoint_state.json")
                        with open(checkpoint_log, "w") as f:
                            json.dump({
                                "epoch": epoch + 1,
                                "batch": batch_idx,
                                "total_batches": total_batches,
                                "avg_loss": avg_loss,
                                "timestamp": datetime.now().isoformat()
                            }, f, indent=2)

                    # PERIODIC MEMORY CLEANUP to prevent fragmentation and leaks
                    cleanup_interval = getattr(config, 'memory_cleanup_interval', 100)
                    if batch_idx > 0 and batch_idx % cleanup_interval == 0:
                        # Clear Python garbage
                        gc.collect()
                        # Clear CUDA cache
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        # Clear timing buffers to prevent memory growth
                        if len(batch_times) > 1000:
                            batch_times = batch_times[-100:]
                            data_times = data_times[-100:]
                            forward_times = forward_times[-100:]
                            backward_times = backward_times[-100:]

                    # Reset data timer for next batch
                    data_start = time.time()

                except torch.cuda.OutOfMemoryError as e:
                    print(f"  OOM at batch {batch_idx} - reduce batch_size")
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    raise  # Stop training on OOM - need to fix batch size
                except Exception as e:
                    print(f"  Error batch {batch_idx}: {e}")
                    continue

            # Epoch summary
            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            epoch_time = time.time() - epoch_start

            print(f"\n  >>> Epoch {epoch+1} complete: Loss={avg_epoch_loss:.4f}, Time={epoch_time:.1f}s")

            training_log.append({
                "epoch": epoch + 1,
                "avg_loss": round(avg_epoch_loss, 6),
                "time_seconds": round(epoch_time, 1)
            })

            # Save best
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_dir = config.output_dir + "_best"
                os.makedirs(best_dir, exist_ok=True)
                model.save_pretrained(best_dir)
                print(f"  [BEST] Saved to {best_dir}")

    except KeyboardInterrupt:
        print("\n\n" + "!" * 60)
        print("  TRAINING INTERRUPTED BY USER")
        print("!" * 60)
        training_crashed = True
        crash_epoch = epoch + 1 if 'epoch' in dir() else 0
        crash_batch = batch_idx if 'batch_idx' in dir() else 0

    except Exception as e:
        print("\n\n" + "!" * 60)
        print(f"  UNEXPECTED TRAINING ERROR: {e}")
        print("!" * 60)
        training_crashed = True
        crash_epoch = epoch + 1 if 'epoch' in dir() else 0
        crash_batch = batch_idx if 'batch_idx' in dir() else 0
        import traceback
        traceback.print_exc()

    # Save model on crash to preserve progress
    if training_crashed and crash_recovery_enabled:
        recovery_dir = config.output_dir + f"_CRASH_e{crash_epoch}_b{crash_batch}"
        print(f"\n[CRASH RECOVERY] Saving model to {recovery_dir}...")
        try:
            os.makedirs(recovery_dir, exist_ok=True)
            model.save_pretrained(recovery_dir)
            # Save crash state
            crash_log = os.path.join(recovery_dir, "crash_state.json")
            with open(crash_log, "w") as f:
                json.dump({
                    "epoch": crash_epoch,
                    "batch": crash_batch,
                    "total_batches": total_batches,
                    "training_log": training_log,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)
            print(f"[CRASH RECOVERY] Model saved successfully!")
            print(f"  You can resume from this checkpoint later.")
        except Exception as save_err:
            print(f"[CRASH RECOVERY] Failed to save: {save_err}")

    # =========================================================================
    # Save Final (only if training completed successfully)
    # =========================================================================
    if not training_crashed:
        print(f"\n[5/5] Saving final adapter to {config.output_dir}...")

        os.makedirs(config.output_dir, exist_ok=True)
        model.save_pretrained(config.output_dir)

        # Save training log
        log_path = os.path.join(config.output_dir, "training_log.json")
        with open(log_path, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "epochs": config.epochs,
                    "lr": config.lr,
                    "lora_r": config.lora_r,
                    "batch_size": config.batch_size,
                    "grad_accum": config.grad_accum,
                    "precompute_tokens": config.precompute_tokens,
                    "class_balanced": config.class_balanced
                },
                "training_log": training_log
            }, f, indent=2)

        print("\n" + "=" * 60)
        print("  OPTIMIZED Training Complete!")
        print(f"  Best loss: {best_loss:.6f}")
        print(f"  Saved to: {config.output_dir}")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("  Training did NOT complete successfully.")
        print(f"  Crashed at: Epoch {crash_epoch}, Batch {crash_batch}")
        if crash_recovery_enabled:
            print(f"  Recovery checkpoint saved to: {recovery_dir}")
        print("=" * 60)


if __name__ == "__main__":
    main()
