"""
Medical Image Augmentation Pipeline for MedGamma.

Provides domain-appropriate augmentations for medical images that:
1. Preserve diagnostic features (no aggressive crops/distortions)
2. Simulate real-world variations (positioning, contrast, noise)
3. Handle different modalities appropriately (X-ray vs CT vs MRI vs Ultrasound)

Key techniques:
- Geometric: rotation, flip, scale (conservative)
- Intensity: brightness, contrast, gamma
- Noise: Gaussian, salt-pepper (simulate detector noise)
- Medical-specific: CLAHE, histogram equalization
"""

import numpy as np
import random
from PIL import Image, ImageEnhance, ImageFilter
from typing import Optional, Tuple, Dict, List, Callable
import torch
from dataclasses import dataclass


@dataclass
class AugmentationConfig:
    """Configuration for medical image augmentation."""
    # Geometric
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.0  # Usually not appropriate for medical
    rotation_range: Tuple[float, float] = (-15, 15)  # degrees
    rotation_prob: float = 0.5
    scale_range: Tuple[float, float] = (0.9, 1.1)
    scale_prob: float = 0.3

    # Intensity
    brightness_range: Tuple[float, float] = (0.85, 1.15)
    brightness_prob: float = 0.5
    contrast_range: Tuple[float, float] = (0.85, 1.15)
    contrast_prob: float = 0.5
    gamma_range: Tuple[float, float] = (0.85, 1.15)
    gamma_prob: float = 0.3

    # Noise
    gaussian_noise_prob: float = 0.2
    gaussian_noise_std: float = 0.02
    salt_pepper_prob: float = 0.1
    salt_pepper_amount: float = 0.01

    # Medical-specific
    clahe_prob: float = 0.3
    clahe_clip_limit: float = 2.0
    invert_prob: float = 0.0  # For certain modalities


# Modality-specific presets
MODALITY_CONFIGS = {
    "xray": AugmentationConfig(
        horizontal_flip_prob=0.5,
        rotation_range=(-10, 10),
        brightness_range=(0.8, 1.2),
        gaussian_noise_prob=0.3,
        clahe_prob=0.4
    ),
    "ct": AugmentationConfig(
        horizontal_flip_prob=0.5,
        rotation_range=(-5, 5),  # Less rotation for CT
        brightness_range=(0.9, 1.1),
        contrast_range=(0.9, 1.1),
        gaussian_noise_prob=0.2,
        clahe_prob=0.2
    ),
    "mri": AugmentationConfig(
        horizontal_flip_prob=0.5,
        rotation_range=(-10, 10),
        brightness_range=(0.85, 1.15),
        gamma_range=(0.8, 1.2),
        gamma_prob=0.4,
        gaussian_noise_prob=0.2
    ),
    "ultrasound": AugmentationConfig(
        horizontal_flip_prob=0.5,
        rotation_range=(-15, 15),  # More variation in US
        scale_range=(0.85, 1.15),
        scale_prob=0.4,
        gaussian_noise_prob=0.4,
        gaussian_noise_std=0.03,  # More noise in US
        clahe_prob=0.5
    ),
    "default": AugmentationConfig()
}


class MedicalImageAugmentor:
    """
    Medical image augmentation with modality awareness.

    Example usage:
        augmentor = MedicalImageAugmentor(modality="xray")
        augmented_image = augmentor(pil_image)

        # Or with bounding boxes
        aug_image, aug_boxes = augmentor.transform_with_boxes(image, boxes)
    """

    def __init__(
        self,
        modality: str = "default",
        config: Optional[AugmentationConfig] = None,
        p: float = 0.5  # Overall augmentation probability
    ):
        """
        Initialize augmentor.

        Args:
            modality: One of "xray", "ct", "mri", "ultrasound", "default"
            config: Custom augmentation config (overrides modality preset)
            p: Probability of applying any augmentation
        """
        self.modality = modality.lower()
        self.config = config or MODALITY_CONFIGS.get(self.modality, MODALITY_CONFIGS["default"])
        self.overall_prob = p

    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply augmentations to a PIL image."""
        if random.random() > self.overall_prob:
            return image

        # Make a copy to avoid modifying original
        image = image.copy()

        # Apply augmentations in order
        image = self._apply_geometric(image)
        image = self._apply_intensity(image)
        image = self._apply_noise(image)
        image = self._apply_medical_specific(image)

        return image

    def transform_with_boxes(
        self,
        image: Image.Image,
        boxes: List[List[float]],
        normalized: bool = True
    ) -> Tuple[Image.Image, List[List[float]]]:
        """
        Apply augmentations that also transform bounding boxes.

        Args:
            image: PIL image
            boxes: List of [x1, y1, x2, y2] boxes
            normalized: If True, boxes are normalized 0-1000 (MedGamma format)

        Returns:
            Augmented image and transformed boxes
        """
        if random.random() > self.overall_prob or not boxes:
            return image, boxes

        image = image.copy()
        boxes = [list(b) for b in boxes]  # Deep copy
        w, h = image.size

        # Only apply box-safe augmentations

        # Horizontal flip
        if random.random() < self.config.horizontal_flip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if normalized:
                boxes = [[1000 - b[2], b[1], 1000 - b[0], b[3]] for b in boxes]
            else:
                boxes = [[w - b[2], b[1], w - b[0], b[3]] for b in boxes]

        # Intensity augmentations (no box change needed)
        image = self._apply_intensity(image)
        image = self._apply_noise(image)
        image = self._apply_medical_specific(image)

        return image, boxes

    def _apply_geometric(self, image: Image.Image) -> Image.Image:
        """Apply geometric transformations."""
        config = self.config

        # Horizontal flip
        if random.random() < config.horizontal_flip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # Vertical flip
        if random.random() < config.vertical_flip_prob:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)

        # Rotation
        if random.random() < config.rotation_prob:
            angle = random.uniform(*config.rotation_range)
            # Use expand=True to avoid cropping, then resize back
            orig_size = image.size
            image = image.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0))

        # Scale
        if random.random() < config.scale_prob:
            scale = random.uniform(*config.scale_range)
            w, h = image.size
            new_w, new_h = int(w * scale), int(h * scale)
            image = image.resize((new_w, new_h), Image.BILINEAR)
            # Crop or pad to original size
            if scale > 1:
                # Crop center
                left = (new_w - w) // 2
                top = (new_h - h) // 2
                image = image.crop((left, top, left + w, top + h))
            else:
                # Pad
                padded = Image.new(image.mode, (w, h), (0, 0, 0) if image.mode == 'RGB' else 0)
                left = (w - new_w) // 2
                top = (h - new_h) // 2
                padded.paste(image, (left, top))
                image = padded

        return image

    def _apply_intensity(self, image: Image.Image) -> Image.Image:
        """Apply intensity transformations."""
        config = self.config

        # Brightness
        if random.random() < config.brightness_prob:
            factor = random.uniform(*config.brightness_range)
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(factor)

        # Contrast
        if random.random() < config.contrast_prob:
            factor = random.uniform(*config.contrast_range)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(factor)

        # Gamma correction
        if random.random() < config.gamma_prob:
            gamma = random.uniform(*config.gamma_range)
            image = self._adjust_gamma(image, gamma)

        return image

    def _apply_noise(self, image: Image.Image) -> Image.Image:
        """Apply noise augmentations."""
        config = self.config

        # Gaussian noise
        if random.random() < config.gaussian_noise_prob:
            image = self._add_gaussian_noise(image, config.gaussian_noise_std)

        # Salt and pepper noise
        if random.random() < config.salt_pepper_prob:
            image = self._add_salt_pepper_noise(image, config.salt_pepper_amount)

        return image

    def _apply_medical_specific(self, image: Image.Image) -> Image.Image:
        """Apply medical imaging specific augmentations."""
        config = self.config

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if random.random() < config.clahe_prob:
            image = self._apply_clahe(image, config.clahe_clip_limit)

        # Inversion (for certain modalities)
        if random.random() < config.invert_prob:
            from PIL import ImageOps
            image = ImageOps.invert(image.convert('RGB'))

        return image

    @staticmethod
    def _adjust_gamma(image: Image.Image, gamma: float) -> Image.Image:
        """Adjust gamma of the image."""
        img_array = np.array(image).astype(np.float32) / 255.0
        img_array = np.power(img_array, gamma)
        img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(img_array)

    @staticmethod
    def _add_gaussian_noise(image: Image.Image, std: float) -> Image.Image:
        """Add Gaussian noise to the image."""
        img_array = np.array(image).astype(np.float32) / 255.0
        noise = np.random.normal(0, std, img_array.shape)
        img_array = img_array + noise
        img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(img_array)

    @staticmethod
    def _add_salt_pepper_noise(image: Image.Image, amount: float) -> Image.Image:
        """Add salt and pepper noise."""
        img_array = np.array(image)

        # Salt
        salt_mask = np.random.random(img_array.shape[:2]) < amount / 2
        img_array[salt_mask] = 255

        # Pepper
        pepper_mask = np.random.random(img_array.shape[:2]) < amount / 2
        img_array[pepper_mask] = 0

        return Image.fromarray(img_array)

    @staticmethod
    def _apply_clahe(image: Image.Image, clip_limit: float = 2.0) -> Image.Image:
        """Apply CLAHE (requires OpenCV)."""
        try:
            import cv2
            img_array = np.array(image)

            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                # Convert to LAB color space
                lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
                l_channel = lab[:, :, 0]

                # Apply CLAHE to L channel
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
                l_channel = clahe.apply(l_channel)

                lab[:, :, 0] = l_channel
                img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                # Grayscale
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
                img_array = clahe.apply(img_array)

            return Image.fromarray(img_array)
        except ImportError:
            return image


class MixUp:
    """
    MixUp augmentation for medical images.

    Note: Use carefully in medical imaging - mixing lesions from different
    patients may create unrealistic patterns.

    Best for:
    - Classification tasks
    - When you have many similar images

    Not recommended for:
    - Detection (box locations become ambiguous)
    - Small datasets
    """

    def __init__(self, alpha: float = 0.2, p: float = 0.5):
        """
        Args:
            alpha: Beta distribution parameter (smaller = less mixing)
            p: Probability of applying mixup
        """
        self.alpha = alpha
        self.p = p

    def __call__(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        label1: int,
        label2: int
    ) -> Tuple[torch.Tensor, float, int, int]:
        """
        Mix two images and return mixed image with mixing coefficient.

        Returns:
            mixed_image, lambda, label1, label2
        """
        if random.random() > self.p:
            return image1, 1.0, label1, label1

        lam = np.random.beta(self.alpha, self.alpha)
        mixed = lam * image1 + (1 - lam) * image2
        return mixed, lam, label1, label2


class CutMix:
    """
    CutMix augmentation - cut and paste patches between images.

    More appropriate for medical imaging than MixUp as it preserves
    local structure while providing regularization.
    """

    def __init__(self, alpha: float = 1.0, p: float = 0.5):
        """
        Args:
            alpha: Beta distribution parameter
            p: Probability of applying cutmix
        """
        self.alpha = alpha
        self.p = p

    def __call__(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        label1: int,
        label2: int
    ) -> Tuple[torch.Tensor, float, int, int]:
        """
        Apply CutMix to two images.

        Returns:
            mixed_image, lambda (area ratio), label1, label2
        """
        if random.random() > self.p:
            return image1, 1.0, label1, label1

        lam = np.random.beta(self.alpha, self.alpha)

        _, H, W = image1.shape

        # Get random box
        cut_ratio = np.sqrt(1 - lam)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        x1 = np.clip(cx - cut_w // 2, 0, W)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        y2 = np.clip(cy + cut_h // 2, 0, H)

        # Create mixed image
        mixed = image1.clone()
        mixed[:, y1:y2, x1:x2] = image2[:, y1:y2, x1:x2]

        # Adjust lambda for actual box size
        lam = 1 - ((x2 - x1) * (y2 - y1)) / (W * H)

        return mixed, lam, label1, label2


class TestTimeAugmentation:
    """
    Test-time augmentation (TTA) for improved predictions.

    Applies multiple augmentations at inference time and aggregates
    predictions for more robust results.
    """

    def __init__(
        self,
        augmentations: List[str] = None,
        n_augmentations: int = 5
    ):
        """
        Args:
            augmentations: List of augmentation names to apply
            n_augmentations: Number of augmented versions to create
        """
        self.augmentations = augmentations or [
            "original",
            "hflip",
            "rotate_5",
            "rotate_-5",
            "brightness_up",
            "brightness_down"
        ]
        self.n_augmentations = min(n_augmentations, len(self.augmentations))

    def __call__(self, image: Image.Image) -> List[Image.Image]:
        """Generate list of augmented images for TTA."""
        results = []

        for aug_name in self.augmentations[:self.n_augmentations]:
            aug_image = self._apply_single(image, aug_name)
            results.append(aug_image)

        return results

    def _apply_single(self, image: Image.Image, aug_name: str) -> Image.Image:
        """Apply a single augmentation."""
        if aug_name == "original":
            return image
        elif aug_name == "hflip":
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        elif aug_name == "vflip":
            return image.transpose(Image.FLIP_TOP_BOTTOM)
        elif aug_name.startswith("rotate_"):
            angle = float(aug_name.split("_")[1])
            return image.rotate(angle, resample=Image.BILINEAR, expand=False)
        elif aug_name == "brightness_up":
            return ImageEnhance.Brightness(image).enhance(1.1)
        elif aug_name == "brightness_down":
            return ImageEnhance.Brightness(image).enhance(0.9)
        elif aug_name == "contrast_up":
            return ImageEnhance.Contrast(image).enhance(1.1)
        elif aug_name == "contrast_down":
            return ImageEnhance.Contrast(image).enhance(0.9)
        else:
            return image

    @staticmethod
    def aggregate_predictions(
        predictions: List[np.ndarray],
        method: str = "mean"
    ) -> np.ndarray:
        """
        Aggregate predictions from multiple augmented versions.

        Args:
            predictions: List of prediction arrays
            method: "mean", "max", "vote" (for classification)
        """
        stacked = np.stack(predictions)

        if method == "mean":
            return np.mean(stacked, axis=0)
        elif method == "max":
            return np.max(stacked, axis=0)
        elif method == "vote":
            # For classification - majority vote
            return np.bincount(stacked.astype(int).flatten()).argmax()
        else:
            return np.mean(stacked, axis=0)

    @staticmethod
    def aggregate_boxes(
        all_boxes: List[List[List[float]]],
        iou_threshold: float = 0.5
    ) -> List[List[float]]:
        """
        Aggregate detection boxes from TTA using NMS-like approach.

        Args:
            all_boxes: List of box lists from each augmentation
            iou_threshold: IoU threshold for merging
        """
        # Flatten all boxes
        flat_boxes = []
        for aug_boxes in all_boxes:
            flat_boxes.extend(aug_boxes)

        if not flat_boxes:
            return []

        # Simple NMS
        boxes = np.array(flat_boxes)

        # Sort by confidence if available, else by area
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        order = areas.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            if order.size == 1:
                break

            # Compute IoU with remaining boxes
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return boxes[keep].tolist()


# Convenience function
def get_augmentor(modality: str = "default", training: bool = True) -> MedicalImageAugmentor:
    """
    Get an appropriate augmentor for the given modality.

    Args:
        modality: "xray", "ct", "mri", "ultrasound", or "default"
        training: If False, returns a no-op augmentor
    """
    if not training:
        return MedicalImageAugmentor(modality=modality, p=0.0)
    return MedicalImageAugmentor(modality=modality, p=0.5)
