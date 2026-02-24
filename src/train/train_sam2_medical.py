"""
Enhanced SAM2 Medical Segmentation Training

This module provides robust training for SAM2 on medical imaging datasets
with LoRA fine-tuning, mixed precision, and comprehensive logging.

Supports:
- SLAKE dataset (medical image segmentation masks)
- Multiple prompt types (box, point, mask)
- Validation with Dice/IoU metrics
- Gradient accumulation for memory efficiency
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torch.amp import GradScaler, autocast
from PIL import Image
from tqdm import tqdm

# SAM2 imports
from hydra.utils import instantiate
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model


@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration for SAM2 training."""
    enabled: bool = True
    patience: int = 5  # Number of evaluation windows without improvement
    min_delta: float = 0.005  # Minimum improvement to count as progress
    loss_threshold: float = 0.1  # Stop if loss goes below this
    dice_threshold: float = 0.9  # Stop if dice goes above this
    eval_interval: int = 50  # Evaluate every N batches


@dataclass
class SAM2TrainingConfig:
    """Configuration for SAM2 medical training."""
    # Data
    data_dir: str = "medical_data"
    dataset: str = "slake"  # slake, brats, or custom
    image_size: int = 1024  # SAM2 REQUIRES 1024 - architecture constraint (64x64 feature maps)

    # Model
    sam2_config: str = "sam2_hiera_t.yaml"
    checkpoint: str = "checkpoints/sam2_hiera_tiny.pt"

    # LoRA - minimal for 16GB VRAM with 1024x1024 images
    lora_r: int = 4  # Small rank to save memory
    lora_alpha: int = 8
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["qkv", "proj"])

    # Training - aggressively optimized for 16GB VRAM
    epochs: int = 5
    batch_size: int = 1  # Must be 1 for 1024x1024 on 16GB GPU
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 16  # Higher accumulation for effective batch size
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # Mixed precision
    use_amp: bool = True

    # Validation
    val_split: float = 0.1
    val_every_n_epochs: int = 1

    # Output
    output_dir: str = "outputs/sam2_medical"
    save_every_n_epochs: int = 1

    # Debug
    max_samples: int = -1  # -1 for all

    # Early stopping
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)


class MedicalSegmentationDataset(Dataset):
    """
    Medical image segmentation dataset supporting SLAKE format.

    Expected structure:
    - imgs/case_id/source.jpg (or .png)
    - imgs/case_id/mask.png
    """

    def __init__(self, data_dir: str, split: str = "train", image_size: int = 1024):
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        self.samples = []

        # Find SLAKE dataset
        slake_dir = os.path.join(data_dir, "Slake1.0", "imgs")
        if not os.path.exists(slake_dir):
            slake_dir = os.path.join(data_dir, "slake", "imgs")

        if os.path.exists(slake_dir):
            self._load_slake(slake_dir)
        else:
            print(f"[WARN] SLAKE directory not found at {slake_dir}")

        print(f"[SAM2 Dataset] Loaded {len(self.samples)} samples for {split}")

    def _load_slake(self, imgs_dir: str):
        """Load SLAKE dataset with masks."""
        for case_id in os.listdir(imgs_dir):
            case_path = os.path.join(imgs_dir, case_id)
            if not os.path.isdir(case_path):
                continue

            # Check for required files
            mask_path = os.path.join(case_path, "mask.png")

            # Find source image (can be jpg or png)
            source_path = None
            for ext in ["source.jpg", "source.png", "image.jpg", "image.png"]:
                candidate = os.path.join(case_path, ext)
                if os.path.exists(candidate):
                    source_path = candidate
                    break

            if source_path and os.path.exists(mask_path):
                # Verify mask is not empty
                try:
                    mask = Image.open(mask_path).convert("L")
                    mask_arr = np.array(mask)
                    if mask_arr.max() > 0:  # Has actual mask content
                        self.samples.append({
                            "image": source_path,
                            "mask": mask_path,
                            "case_id": case_id
                        })
                except Exception as e:
                    pass  # Skip invalid masks

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample["image"]).convert("RGB")
        image = image.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        image_array = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)  # [3, H, W]

        # Normalize for SAM2 (ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std

        # Load mask
        mask = Image.open(sample["mask"]).convert("L")
        mask = mask.resize((self.image_size, self.image_size), resample=Image.NEAREST)
        mask_array = np.array(mask).astype(np.float32)
        mask_array = (mask_array > 0).astype(np.float32)  # Binary mask
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)  # [1, H, W]

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "case_id": sample["case_id"]
        }


class SAM2MedicalTrainer:
    """
    Enhanced SAM2 trainer for medical segmentation.

    Features:
    - LoRA fine-tuning of image encoder
    - Mixed precision training
    - Gradient accumulation
    - Dice + BCE loss
    - Validation with IoU metrics
    """

    def __init__(self, config: SAM2TrainingConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler('cuda') if config.use_amp else None

        # Training state
        self.global_step = 0
        self.best_val_dice = 0.0
        self.training_log = []

        # Load model
        self._load_model()

    def _load_model(self):
        """Load SAM2 model with LoRA adapters."""
        print(f"Loading SAM2 model...")

        # Check paths
        config_path = self.config.sam2_config
        checkpoint_path = self.config.checkpoint

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"SAM2 config not found: {config_path}")

        if not os.path.exists(checkpoint_path):
            # Try alternative paths
            alt_paths = [
                "sam2_hiera_tiny.pt",
                "checkpoints/sam2_hiera_tiny.pt",
                os.path.join(self.config.data_dir, "sam2_hiera_tiny.pt")
            ]
            for alt in alt_paths:
                if os.path.exists(alt):
                    checkpoint_path = alt
                    break
            else:
                raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint_path}")

        # Load config and instantiate model
        cfg = OmegaConf.load(config_path)
        self.model = instantiate(cfg.model, _recursive_=True)

        # Load weights
        print(f"Loading weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if "model" in state_dict:
            state_dict = state_dict["model"]

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)} (expected for LoRA)")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")

        self.model.to(self.device)

        # Enable gradient checkpointing for memory efficiency (critical for 1024x1024)
        self._enable_gradient_checkpointing()

        # Apply LoRA to image encoder
        self._apply_lora()

    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory-efficient training with 1024x1024 images."""
        try:
            # Enable for image encoder (biggest memory consumer)
            if hasattr(self.model.image_encoder, 'gradient_checkpointing_enable'):
                self.model.image_encoder.gradient_checkpointing_enable()
                print("Gradient checkpointing enabled for image encoder")
            elif hasattr(self.model.image_encoder, 'set_grad_checkpointing'):
                self.model.image_encoder.set_grad_checkpointing(True)
                print("Gradient checkpointing enabled for image encoder")
            else:
                # Manual approach - enable for transformer blocks
                for name, module in self.model.image_encoder.named_modules():
                    if 'block' in name.lower() or 'layer' in name.lower():
                        if hasattr(module, 'gradient_checkpointing'):
                            module.gradient_checkpointing = True
                print("Manual gradient checkpointing applied to encoder blocks")
        except Exception as e:
            print(f"Warning: Could not enable gradient checkpointing: {e}")

    def _apply_lora(self):
        """Apply LoRA adapters to SAM2 image encoder."""
        peft_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none"
        )

        try:
            self.model.image_encoder = get_peft_model(
                self.model.image_encoder,
                peft_config
            )
            print("LoRA adapters attached to image encoder:")
            self.model.image_encoder.print_trainable_parameters()
        except Exception as e:
            print(f"Warning: Failed to apply LoRA: {e}")
            print("Proceeding with standard fine-tuning")

    def compute_loss(
        self,
        pred_masks: torch.Tensor,
        gt_masks: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Enhanced loss for medical segmentation with boundary awareness.

        Combines:
        1. Dice loss - Region overlap
        2. Focal BCE - Hard pixel mining
        3. Boundary loss - Edge precision (critical for medical)

        Args:
            pred_masks: [B, 1, 256, 256] logits
            gt_masks: [B, 1, H, H] ground truth

        Returns:
            loss: Combined loss tensor
            metrics: Dict with individual loss components
        """
        # Resize GT to match prediction resolution
        gt_low = F.interpolate(gt_masks, size=pred_masks.shape[-2:], mode='nearest')

        # Sigmoid for probabilities
        pred_probs = torch.sigmoid(pred_masks)
        smooth = 1e-6

        # 1. DICE LOSS (soft dice with log for stability)
        intersection = (pred_probs * gt_low).sum(dim=(2, 3))
        union = pred_probs.sum(dim=(2, 3)) + gt_low.sum(dim=(2, 3))
        dice_score = (2 * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice_score.mean()

        # 2. FOCAL BCE LOSS (focuses on hard pixels - important for medical boundaries)
        # Focal loss: FL = -alpha * (1-p)^gamma * log(p)
        gamma = 2.0  # Focus more on hard examples
        alpha = 0.75  # Weight for positive class (lesions are usually minority)

        bce = F.binary_cross_entropy_with_logits(pred_masks, gt_low, reduction='none')
        pt = torch.exp(-bce)  # Probability of correct classification
        focal_weight = alpha * (1 - pt) ** gamma
        focal_loss = (focal_weight * bce).mean()

        # 3. BOUNDARY LOSS (enhances edge precision for medical ROIs)
        # Extract boundaries using Sobel-like gradient
        boundary_loss = self._compute_boundary_loss(pred_probs, gt_low)

        # Combined loss (weighted for medical imaging)
        # Higher weight on dice (region accuracy) and boundary (edge precision)
        total_loss = 0.4 * dice_loss + 0.3 * focal_loss + 0.3 * boundary_loss

        metrics = {
            "dice_loss": dice_loss.item(),
            "focal_loss": focal_loss.item(),
            "boundary_loss": boundary_loss.item(),
            "dice_score": dice_score.mean().item()
        }

        return total_loss, metrics

    def _compute_boundary_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary-aware loss for precise edge segmentation.
        Uses Sobel-like gradient to extract boundaries and compute loss.
        """
        # Sobel kernels for boundary extraction
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)

        # Extract GT boundaries
        gt_grad_x = F.conv2d(gt, sobel_x, padding=1)
        gt_grad_y = F.conv2d(gt, sobel_y, padding=1)
        gt_boundary = torch.sqrt(gt_grad_x**2 + gt_grad_y**2 + 1e-8)
        gt_boundary = (gt_boundary > 0.1).float()  # Binary boundary

        # Extract predicted boundaries
        pred_grad_x = F.conv2d(pred, sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred, sobel_y, padding=1)
        pred_boundary = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-8)

        # Boundary-weighted loss (AMP-safe using MSE instead of BCE)
        # BCE is unsafe with autocast, MSE works well for boundary matching
        boundary_weight = 1.0 + 4.0 * gt_boundary  # 5x weight on boundaries
        pred_boundary_clamped = pred_boundary.clamp(0, 1)

        # Weighted MSE loss - AMP safe
        boundary_mse = F.mse_loss(pred_boundary_clamped, gt_boundary, reduction='none')
        weighted_boundary_loss = (boundary_weight * boundary_mse).mean()

        return weighted_boundary_loss

    def _extract_box_prompt(self, mask: torch.Tensor) -> torch.Tensor:
        """Extract bounding box from mask for SAM2 prompt."""
        B = mask.shape[0]
        boxes = []

        for i in range(B):
            m = mask[i, 0]  # [H, W]
            rows, cols = torch.where(m > 0.5)

            if len(rows) > 0:
                y1, x1 = rows.min().item(), cols.min().item()
                y2, x2 = rows.max().item(), cols.max().item()

                # Add small noise for robustness (simulates imperfect detection boxes)
                noise = np.random.randint(-5, 6, 4)
                box = [
                    max(0, x1 + noise[0]),
                    max(0, y1 + noise[1]),
                    min(m.shape[1], x2 + noise[2]),
                    min(m.shape[0], y2 + noise[3])
                ]
            else:
                # Fallback for empty masks
                box = [0, 0, 100, 100]

            boxes.append(box)

        return torch.tensor(boxes, device=self.device, dtype=torch.float32).unsqueeze(1)

    def _extract_point_prompts(self, mask: torch.Tensor, num_points: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract point prompts from mask for multi-prompt training.
        Returns both foreground (positive) and background (negative) points.

        This improves the model's ability to segment with point-based interaction
        (important for medical "magic mask" functionality).
        """
        B = mask.shape[0]
        all_points = []
        all_labels = []

        for i in range(B):
            m = mask[i, 0]  # [H, W]
            H, W = m.shape

            points = []
            labels = []

            # Find foreground (positive) points
            fg_rows, fg_cols = torch.where(m > 0.5)

            if len(fg_rows) > 0:
                # Sample random foreground points
                num_fg = min(num_points, len(fg_rows))
                indices = np.random.choice(len(fg_rows), num_fg, replace=False)

                for idx in indices:
                    y, x = fg_rows[idx].item(), fg_cols[idx].item()
                    # Add slight noise
                    x = max(0, min(W-1, x + np.random.randint(-3, 4)))
                    y = max(0, min(H-1, y + np.random.randint(-3, 4)))
                    points.append([x, y])
                    labels.append(1)  # Positive/foreground

                # Add center point (most reliable)
                center_y = fg_rows.float().mean().int().item()
                center_x = fg_cols.float().mean().int().item()
                points.append([center_x, center_y])
                labels.append(1)

            # Find background (negative) points - outside the mask
            bg_rows, bg_cols = torch.where(m < 0.5)

            if len(bg_rows) > 0 and len(points) > 0:
                # Sample 1-2 background points near the boundary
                num_bg = min(2, len(bg_rows))
                indices = np.random.choice(len(bg_rows), num_bg, replace=False)

                for idx in indices:
                    y, x = bg_rows[idx].item(), bg_cols[idx].item()
                    points.append([x, y])
                    labels.append(0)  # Negative/background

            # Fallback if no valid points
            if len(points) == 0:
                points = [[W//2, H//2]]
                labels = [1]

            all_points.append(points)
            all_labels.append(labels)

        # Pad to same length and convert to tensors
        max_points = max(len(p) for p in all_points)
        padded_points = []
        padded_labels = []

        for pts, lbls in zip(all_points, all_labels):
            # Pad with dummy points (will be masked)
            while len(pts) < max_points:
                pts.append([0, 0])
                lbls.append(-1)  # Invalid label
            padded_points.append(pts)
            padded_labels.append(lbls)

        points_tensor = torch.tensor(padded_points, device=self.device, dtype=torch.float32)
        labels_tensor = torch.tensor(padded_labels, device=self.device, dtype=torch.int32)

        return points_tensor, labels_tensor

    def _get_prompt_strategy(self) -> str:
        """
        Randomly select prompt strategy for training.
        This makes the model robust to different input types (magic mask support).
        """
        strategies = ["box", "points", "box_and_points", "center_point"]
        weights = [0.4, 0.25, 0.25, 0.1]  # Box most common (matches MedGemma output)
        return np.random.choice(strategies, p=weights)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, Dict[str, float]]:
        """
        Single training step with multi-prompt strategy.

        Randomly selects between:
        - Box prompts (most common - matches MedGemma detection output)
        - Point prompts (for interactive segmentation / magic mask)
        - Combined box + points (most accurate)
        - Center point only (simple click segmentation)
        """
        images = batch["image"].to(self.device)
        masks = batch["mask"].to(self.device)
        B = images.shape[0]

        # Select prompt strategy for this batch
        prompt_strategy = self._get_prompt_strategy()

        # Forward pass with mixed precision
        with autocast('cuda', enabled=self.config.use_amp):
            # Image encoder
            backbone_out = self.model.forward_image(images)
            _, vision_feats, _, feat_sizes = self.model._prepare_backbone_features(backbone_out)

            # High-res features for decoder
            high_res_features = None
            if self.model.use_high_res_features_in_sam and len(vision_feats) > 1:
                high_res_features = [
                    x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                    for x, s in zip(vision_feats[:-1], feat_sizes[:-1])
                ]

            # Image embeddings
            image_embeddings = vision_feats[-1].permute(1, 2, 0).view(B, -1, *feat_sizes[-1])

            # Prepare prompts based on strategy
            boxes = None
            points = None
            point_labels = None

            if prompt_strategy in ["box", "box_and_points"]:
                boxes = self._extract_box_prompt(masks)

            if prompt_strategy in ["points", "box_and_points", "center_point"]:
                num_pts = 1 if prompt_strategy == "center_point" else 3
                points, point_labels = self._extract_point_prompts(masks, num_points=num_pts)

            # Prompt encoder
            sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
                points=(points, point_labels) if points is not None else None,
                boxes=boxes,
                masks=None
            )

            # Mask decoder
            low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_res_features
            )

            # Compute loss (now with boundary awareness)
            loss, metrics = self.compute_loss(low_res_masks, masks)
            loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        if self.config.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Add prompt strategy to metrics for logging
        metrics["prompt_strategy"] = prompt_strategy

        return loss.item() * self.config.gradient_accumulation_steps, metrics

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Run validation and compute metrics."""
        self.model.eval()

        total_dice = 0.0
        total_iou = 0.0
        num_samples = 0

        for batch in tqdm(val_loader, desc="Validating", leave=False):
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            B = images.shape[0]

            with autocast('cuda', enabled=self.config.use_amp):
                # Forward pass
                backbone_out = self.model.forward_image(images)
                _, vision_feats, _, feat_sizes = self.model._prepare_backbone_features(backbone_out)

                high_res_features = None
                if self.model.use_high_res_features_in_sam and len(vision_feats) > 1:
                    high_res_features = [
                        x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                        for x, s in zip(vision_feats[:-1], feat_sizes[:-1])
                    ]

                image_embeddings = vision_feats[-1].permute(1, 2, 0).view(B, -1, *feat_sizes[-1])
                boxes = self._extract_box_prompt(masks)

                sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
                    points=None, boxes=boxes, masks=None
                )

                low_res_masks, _, _, _ = self.model.sam_mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    repeat_image=False,
                    high_res_features=high_res_features
                )

            # Compute metrics
            pred_masks = torch.sigmoid(low_res_masks) > 0.5
            gt_masks = F.interpolate(masks, size=pred_masks.shape[-2:], mode='nearest') > 0.5

            for i in range(B):
                pred = pred_masks[i, 0]
                gt = gt_masks[i, 0]

                intersection = (pred & gt).sum().float()
                union = (pred | gt).sum().float()

                if union > 0:
                    iou = intersection / union
                    dice = 2 * intersection / (pred.sum() + gt.sum())
                    total_iou += iou.item()
                    total_dice += dice.item()
                    num_samples += 1

        self.model.train()

        return {
            "val_dice": total_dice / max(num_samples, 1),
            "val_iou": total_iou / max(num_samples, 1),
            "val_samples": num_samples
        }

    def train(self):
        """Main training loop."""
        print(f"\n{'='*60}")
        print("SAM2 MEDICAL SEGMENTATION TRAINING")
        print(f"{'='*60}")

        # Create dataset
        dataset = MedicalSegmentationDataset(
            data_dir=self.config.data_dir,
            split="train",
            image_size=self.config.image_size
        )

        if len(dataset) == 0:
            print("ERROR: No training samples found!")
            return

        # Apply sample limit if specified
        if self.config.max_samples > 0:
            indices = list(range(min(len(dataset), self.config.max_samples)))
            dataset = Subset(dataset, indices)
            print(f"  Limited to {len(indices)} samples")

        # Split into train/val
        total_samples = len(dataset)
        val_size = int(total_samples * self.config.val_split)
        train_size = total_samples - val_size

        indices = list(range(total_samples))
        np.random.shuffle(indices)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices) if val_size > 0 else None

        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset) if val_dataset else 0}")

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        ) if val_dataset else None

        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Learning rate scheduler
        total_steps = len(train_loader) * self.config.epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
            pct_start=self.config.warmup_ratio
        )

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Training loop
        self.model.train()
        print(f"\nStarting training for {self.config.epochs} epochs...")
        print(f"  Batches per epoch: {len(train_loader)}")
        print(f"  Gradient accumulation: {self.config.gradient_accumulation_steps}")
        print(f"  Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")

        # Early stopping configuration
        es_config = self.config.early_stopping
        if es_config.enabled:
            print(f"  Early Stopping: enabled (patience={es_config.patience}, "
                  f"loss_threshold={es_config.loss_threshold}, dice_threshold={es_config.dice_threshold})")
        else:
            print("  Early Stopping: disabled")

        # Early stopping state
        best_loss = float('inf')
        best_dice = 0.0
        patience_counter = 0
        early_stop_triggered = False
        window_loss_sum = 0.0
        window_dice_sum = 0.0
        window_steps = 0
        global_batch_idx = 0
        first_loss_check_done = False  # For immediate convergence check

        for epoch in range(self.config.epochs):
            if early_stop_triggered:
                break

            epoch_start = time.time()
            epoch_loss = 0.0
            epoch_dice = 0.0
            epoch_steps = 0

            self.optimizer.zero_grad()

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}")
            for batch_idx, batch in enumerate(pbar):
                try:
                    loss, metrics = self.train_step(batch)
                    epoch_loss += loss
                    epoch_dice += metrics["dice_score"]
                    epoch_steps += 1

                    # Track for early stopping window
                    window_loss_sum += loss
                    window_dice_sum += metrics["dice_score"]
                    window_steps += 1
                    global_batch_idx += 1

                    # Gradient accumulation step
                    if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                        if self.config.use_amp:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.config.max_grad_norm
                            )
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.config.max_grad_norm
                            )
                            self.optimizer.step()

                        self.optimizer.zero_grad()
                        self.scheduler.step()
                        self.global_step += 1

                        # Aggressive memory clearing (critical for 1024x1024 on 16GB)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()

                    # Update progress bar
                    avg_loss = epoch_loss / epoch_steps
                    avg_dice = epoch_dice / epoch_steps
                    pbar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "dice": f"{avg_dice:.4f}",
                        "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
                    })

                    # Early stopping check
                    if es_config.enabled:
                        # Immediate convergence check after first 10 batches
                        if not first_loss_check_done and global_batch_idx >= 10:
                            first_loss_check_done = True
                            current_loss = epoch_loss / max(epoch_steps, 1)
                            current_dice = epoch_dice / max(epoch_steps, 1)
                            if current_loss < es_config.loss_threshold:
                                print(f"\n  [EARLY STOP] Initial loss {current_loss:.6f} already below threshold")
                                early_stop_triggered = True
                                break
                            if current_dice > es_config.dice_threshold:
                                print(f"\n  [EARLY STOP] Initial dice {current_dice:.4f} already above threshold")
                                early_stop_triggered = True
                                break

                        # Regular interval check
                        if global_batch_idx % es_config.eval_interval == 0 and window_steps > 0:
                            window_avg_loss = window_loss_sum / window_steps
                            window_avg_dice = window_dice_sum / window_steps

                            # Check absolute thresholds (model has converged)
                            if window_avg_loss < es_config.loss_threshold:
                                print(f"\n  [EARLY STOP] Loss {window_avg_loss:.6f} < threshold {es_config.loss_threshold}")
                                early_stop_triggered = True
                                break

                            if window_avg_dice > es_config.dice_threshold:
                                print(f"\n  [EARLY STOP] Dice {window_avg_dice:.4f} > threshold {es_config.dice_threshold}")
                                early_stop_triggered = True
                                break

                            # Check for improvement (using loss as primary metric)
                            improvement = best_loss - window_avg_loss
                            if improvement > es_config.min_delta:
                                best_loss = window_avg_loss
                                best_dice = max(best_dice, window_avg_dice)
                                patience_counter = 0
                                print(f"\n  [ES] New best: loss={best_loss:.6f}, dice={best_dice:.4f}")
                            else:
                                patience_counter += 1
                                if patience_counter % 2 == 0:  # Log every other check
                                    print(f"\n  [ES] No improvement. Patience: {patience_counter}/{es_config.patience}")

                                if patience_counter >= es_config.patience:
                                    print(f"\n  [EARLY STOP] No improvement for {es_config.patience} evaluations")
                                    early_stop_triggered = True
                                    break

                            # Reset window
                            window_loss_sum = 0.0
                            window_dice_sum = 0.0
                            window_steps = 0

                except Exception as e:
                    print(f"\n  Error in batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            # Epoch summary
            epoch_time = time.time() - epoch_start
            avg_loss = epoch_loss / max(epoch_steps, 1)
            avg_dice = epoch_dice / max(epoch_steps, 1)

            if early_stop_triggered:
                print(f"\n  Epoch {epoch+1} stopped early: Loss={avg_loss:.4f}, Dice={avg_dice:.4f}")
            else:
                print(f"\n  Epoch {epoch+1} complete: Loss={avg_loss:.4f}, Dice={avg_dice:.4f}, Time={epoch_time:.1f}s")

            # Validation
            val_metrics = {}
            if val_loader and (epoch + 1) % self.config.val_every_n_epochs == 0:
                val_metrics = self.validate(val_loader)
                print(f"  Validation: Dice={val_metrics['val_dice']:.4f}, IoU={val_metrics['val_iou']:.4f}")

                # Save best model
                if val_metrics["val_dice"] > self.best_val_dice:
                    self.best_val_dice = val_metrics["val_dice"]
                    self._save_checkpoint("best")

            # Log epoch
            self.training_log.append({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "train_dice": avg_dice,
                "early_stopped": early_stop_triggered,
                **val_metrics,
                "time": epoch_time
            })

            # Save checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self._save_checkpoint(f"epoch_{epoch+1}")

            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # Break outer loop if early stopped
            if early_stop_triggered:
                break

        # Save final model
        self._save_checkpoint("final")
        self._save_training_log()

        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        if early_stop_triggered:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            print(f"Best training loss: {best_loss:.6f}, Best dice: {best_dice:.4f}")
        print(f"Best validation Dice: {self.best_val_dice:.4f}")
        print(f"Output: {self.config.output_dir}")
        print(f"{'='*60}")

    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        save_dir = os.path.join(self.config.output_dir, name)
        os.makedirs(save_dir, exist_ok=True)

        # Save LoRA adapters
        self.model.image_encoder.save_pretrained(save_dir)

        # Save config
        config_path = os.path.join(save_dir, "training_config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.__dict__, f, indent=2, default=str)

        print(f"  Checkpoint saved: {save_dir}")

    def _save_training_log(self):
        """Save training log to JSON."""
        log_path = os.path.join(self.config.output_dir, "training_log.json")
        with open(log_path, "w") as f:
            json.dump(self.training_log, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="SAM2 Medical Segmentation Training")
    parser.add_argument("--data-dir", default="medical_data", help="Data directory")
    parser.add_argument("--output-dir", default="outputs/sam2_medical", help="Output directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max-samples", type=int, default=-1, help="Limit samples (-1 for all)")
    parser.add_argument("--quick", action="store_true", help="Quick run with 100 samples, 1 epoch")

    args = parser.parse_args()

    # Create config
    config = SAM2TrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs if not args.quick else 1,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_samples=args.max_samples if not args.quick else 100
    )

    # Run training
    trainer = SAM2MedicalTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
