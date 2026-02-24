"""
Domain Adaptation Techniques for MedGamma.

Implements strategies to improve generalization across different:
- Medical imaging modalities (X-ray, CT, MRI, Ultrasound)
- Datasets (VinDr, NIH, BUSI, etc.)
- Institutions (different scanners, protocols)

Key Techniques:
1. DOMAIN-ADVERSARIAL TRAINING: Learn domain-invariant features
2. GRADIENT REVERSAL: Penalize domain-specific features
3. CONSISTENCY REGULARIZATION: Enforce consistent predictions under augmentation
4. FEATURE ALIGNMENT: Match feature distributions across domains
5. META-LEARNING: Learn to adapt quickly to new domains
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Dict, List, Optional, Tuple
import numpy as np


# =============================================================================
# Gradient Reversal Layer (for DANN)
# =============================================================================

class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer for Domain-Adversarial Neural Networks (DANN).

    During forward pass: Identity
    During backward pass: Negate gradients and scale by lambda

    This forces the feature extractor to learn domain-invariant representations
    by confusing the domain classifier.
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        return -lambda_ * grads, None


class GradientReversal(nn.Module):
    """Gradient Reversal Layer wrapper."""

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_: float):
        self.lambda_ = lambda_


# =============================================================================
# Domain Classifier for DANN
# =============================================================================

class DomainClassifier(nn.Module):
    """
    Domain classifier head for domain-adversarial training.

    Attached to intermediate features from the model to classify
    which domain/modality the input comes from.
    """

    def __init__(
        self,
        feature_dim: int,
        num_domains: int,
        hidden_dim: int = 256,
        dropout: float = 0.3
    ):
        super().__init__()

        self.grl = GradientReversal()

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_domains)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, D] pooled features from encoder

        Returns:
            domain_logits: [B, num_domains]
        """
        # Apply gradient reversal
        reversed_features = self.grl(features)
        return self.classifier(reversed_features)

    def set_lambda(self, lambda_: float):
        """Update GRL lambda during training."""
        self.grl.set_lambda(lambda_)


# =============================================================================
# Consistency Regularization
# =============================================================================

class ConsistencyRegularizer(nn.Module):
    """
    Consistency regularization for semi-supervised learning.

    Enforces that predictions remain consistent under different
    augmentations of the same input.

    Key for medical imaging where labels may be noisy or incomplete.
    """

    def __init__(
        self,
        temperature: float = 0.5,
        threshold: float = 0.95,
        loss_type: str = "kl"  # "kl" or "mse"
    ):
        super().__init__()
        self.temperature = temperature
        self.threshold = threshold
        self.loss_type = loss_type

    def forward(
        self,
        logits_weak: torch.Tensor,
        logits_strong: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute consistency loss between weakly and strongly augmented views.

        Args:
            logits_weak: Predictions on weakly augmented input
            logits_strong: Predictions on strongly augmented input
            mask: Optional mask for valid samples

        Returns:
            consistency_loss: Scalar loss
        """
        # Compute pseudo-labels from weak augmentation
        with torch.no_grad():
            probs_weak = F.softmax(logits_weak / self.temperature, dim=-1)
            max_probs, pseudo_labels = probs_weak.max(dim=-1)

            # Only use confident predictions
            confident_mask = max_probs > self.threshold

            if mask is not None:
                confident_mask = confident_mask & mask

        if confident_mask.sum() == 0:
            return torch.tensor(0.0, device=logits_weak.device)

        if self.loss_type == "kl":
            log_probs_strong = F.log_softmax(logits_strong, dim=-1)
            loss = F.kl_div(
                log_probs_strong[confident_mask],
                probs_weak[confident_mask],
                reduction='batchmean'
            )
        else:
            probs_strong = F.softmax(logits_strong, dim=-1)
            loss = F.mse_loss(
                probs_strong[confident_mask],
                probs_weak[confident_mask]
            )

        return loss


# =============================================================================
# Feature Alignment (MMD Loss)
# =============================================================================

class MMDLoss(nn.Module):
    """
    Maximum Mean Discrepancy (MMD) loss for feature alignment.

    Minimizes the distribution mismatch between source and target domains
    in the feature space.
    """

    def __init__(self, kernel: str = "rbf", kernel_mul: float = 2.0, kernel_num: int = 5):
        super().__init__()
        self.kernel = kernel
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num

    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MMD loss between source and target feature distributions.

        Args:
            source_features: [N, D] features from source domain
            target_features: [M, D] features from target domain

        Returns:
            mmd_loss: Scalar loss
        """
        if self.kernel == "rbf":
            return self._mmd_rbf(source_features, target_features)
        else:
            return self._mmd_linear(source_features, target_features)

    def _mmd_rbf(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """RBF kernel MMD."""
        total = torch.cat([source, target], dim=0)
        n = source.size(0)
        m = target.size(0)

        # Compute pairwise distances
        total = total.unsqueeze(0) - total.unsqueeze(1)
        total = torch.sum(total ** 2, dim=-1)

        # RBF kernel bandwidth
        bandwidth = torch.sum(total) / (total.numel() - total.size(0))
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]

        # Compute kernel matrix
        kernel = torch.zeros_like(total)
        for bw in bandwidth_list:
            kernel += torch.exp(-total / bw)
        kernel /= self.kernel_num

        # MMD estimate
        ss = kernel[:n, :n].mean()
        tt = kernel[n:, n:].mean()
        st = kernel[:n, n:].mean()

        return ss + tt - 2 * st

    def _mmd_linear(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Linear kernel MMD (faster, less expressive)."""
        delta = source.mean(dim=0) - target.mean(dim=0)
        return torch.sum(delta ** 2)


# =============================================================================
# Multi-Domain Training Wrapper
# =============================================================================

class MultiDomainTrainer:
    """
    Wrapper for training with multiple domains/modalities.

    Handles:
    - Domain labels for each sample
    - Domain classifier training
    - Feature alignment losses
    - Progressive domain adaptation (curriculum)
    """

    def __init__(
        self,
        model: nn.Module,
        num_domains: int,
        feature_dim: int,
        use_dann: bool = True,
        use_mmd: bool = False,
        use_consistency: bool = True,
        dann_lambda: float = 0.1,
        mmd_weight: float = 0.05,
        consistency_weight: float = 0.1
    ):
        """
        Args:
            model: Base model (e.g., MedGemma)
            num_domains: Number of domains/modalities
            feature_dim: Dimension of pooled features
            use_dann: Enable domain-adversarial training
            use_mmd: Enable MMD feature alignment
            use_consistency: Enable consistency regularization
        """
        self.model = model
        self.num_domains = num_domains
        self.use_dann = use_dann
        self.use_mmd = use_mmd
        self.use_consistency = use_consistency

        # Weights
        self.dann_lambda = dann_lambda
        self.mmd_weight = mmd_weight
        self.consistency_weight = consistency_weight

        # Initialize components
        if use_dann:
            self.domain_classifier = DomainClassifier(feature_dim, num_domains)

        if use_mmd:
            self.mmd_loss = MMDLoss()

        if use_consistency:
            self.consistency_reg = ConsistencyRegularizer()

        # Training progress
        self.current_epoch = 0
        self.max_epochs = 1

    def compute_domain_loss(
        self,
        features: torch.Tensor,
        domain_labels: torch.Tensor,
        progress: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute domain adaptation losses.

        Args:
            features: [B, D] pooled features
            domain_labels: [B] domain indices
            progress: Training progress (0 to 1) for lambda scheduling

        Returns:
            Dict with loss components
        """
        losses = {}

        if self.use_dann:
            # Progressive lambda scheduling (ramps up during training)
            lambda_ = 2.0 / (1.0 + np.exp(-10 * progress)) - 1.0
            lambda_ *= self.dann_lambda
            self.domain_classifier.set_lambda(lambda_)

            domain_logits = self.domain_classifier(features)
            dann_loss = F.cross_entropy(domain_logits, domain_labels)
            losses['dann'] = dann_loss * self.dann_lambda

            # Domain accuracy (for monitoring)
            with torch.no_grad():
                domain_preds = domain_logits.argmax(dim=-1)
                domain_acc = (domain_preds == domain_labels).float().mean()
                losses['domain_acc'] = domain_acc

        return losses

    def compute_mmd_loss(
        self,
        features_by_domain: Dict[int, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute pairwise MMD loss across all domain pairs.

        Args:
            features_by_domain: Dict mapping domain_id -> features tensor
        """
        if not self.use_mmd or len(features_by_domain) < 2:
            return torch.tensor(0.0)

        total_mmd = 0.0
        num_pairs = 0

        domain_ids = list(features_by_domain.keys())
        for i, d1 in enumerate(domain_ids):
            for d2 in domain_ids[i + 1:]:
                f1 = features_by_domain[d1]
                f2 = features_by_domain[d2]
                if f1.size(0) > 0 and f2.size(0) > 0:
                    total_mmd += self.mmd_loss(f1, f2)
                    num_pairs += 1

        if num_pairs > 0:
            total_mmd /= num_pairs

        return total_mmd * self.mmd_weight

    def compute_consistency_loss(
        self,
        logits_weak: torch.Tensor,
        logits_strong: torch.Tensor
    ) -> torch.Tensor:
        """Compute consistency regularization loss."""
        if not self.use_consistency:
            return torch.tensor(0.0)

        return self.consistency_reg(logits_weak, logits_strong) * self.consistency_weight


# =============================================================================
# Domain-Specific Prompt Adaptation
# =============================================================================

class DomainPromptAdapter:
    """
    Adapts prompts based on the detected/known domain of the input.

    Different modalities require different analysis approaches:
    - X-ray: Look for consolidations, effusions, cardiomegaly
    - CT: 3D structures, contrast enhancement patterns
    - MRI: Tissue characteristics, enhancement patterns
    - Ultrasound: Echogenicity, shadows, Doppler flow
    """

    DOMAIN_PROMPTS = {
        "xray": {
            "detection": (
                "Analyze this chest X-ray for pathological findings including: "
                "consolidation, effusion, cardiomegaly, pneumothorax, nodules, "
                "and other abnormalities. For each finding, provide the class name "
                'and bounding box. Output in JSON: {"findings": [{"class": "...", "box": [x1,y1,x2,y2]}]}'
            ),
            "screening": (
                "Is this chest X-ray normal or abnormal? Look for: opacity, "
                "consolidation, effusion, masses, cardiomegaly. "
                "Respond with 'HEALTHY' or 'ABNORMAL' with reasoning."
            ),
        },
        "ct": {
            "detection": (
                "Analyze this CT scan for lesions, masses, calcifications, "
                "and other abnormalities. For each finding, provide the class "
                'and location. Output in JSON: {"findings": [{"class": "...", "box": [x1,y1,x2,y2]}]}'
            ),
            "screening": (
                "Is this CT scan showing any pathology? Look for: masses, lesions, "
                "abnormal densities, structural abnormalities. "
                "Respond with 'HEALTHY' or 'ABNORMAL' with reasoning."
            ),
        },
        "mri": {
            "detection": (
                "Analyze this MRI for lesions, tumors, edema, hemorrhage, "
                "and other signal abnormalities. For each finding, provide "
                'the class and location. Output in JSON: {"findings": [{"class": "...", "box": [x1,y1,x2,y2]}]}'
            ),
            "screening": (
                "Is this MRI showing any pathology? Look for: abnormal signal intensity, "
                "mass effect, enhancement patterns, structural changes. "
                "Respond with 'HEALTHY' or 'ABNORMAL' with reasoning."
            ),
        },
        "ultrasound": {
            "detection": (
                "Analyze this ultrasound for masses, cysts, calcifications, "
                "and other abnormalities. For each finding, provide the class "
                'and location. Output in JSON: {"findings": [{"class": "...", "box": [x1,y1,x2,y2]}]}'
            ),
            "screening": (
                "Is this ultrasound showing any pathology? Look for: masses, "
                "cysts, abnormal echogenicity, suspicious features. "
                "Respond with 'HEALTHY' or 'ABNORMAL' with reasoning."
            ),
        },
        "default": {
            "detection": (
                "Analyze this medical image for pathological findings. "
                "For each finding, provide the class name and bounding box. "
                'Output in JSON: {"findings": [{"class": "...", "box": [x1,y1,x2,y2]}]}'
            ),
            "screening": (
                "Is this medical image showing signs of disease? "
                "Respond with 'HEALTHY' or 'ABNORMAL' with reasoning."
            ),
        }
    }

    @classmethod
    def get_prompt(cls, modality: str, task: str) -> str:
        """
        Get domain-specific prompt.

        Args:
            modality: "xray", "ct", "mri", "ultrasound", or "default"
            task: "detection", "screening", or other task name
        """
        modality = modality.lower()
        if modality not in cls.DOMAIN_PROMPTS:
            modality = "default"

        prompts = cls.DOMAIN_PROMPTS[modality]
        return prompts.get(task, prompts.get("detection", ""))

    @classmethod
    def detect_modality(cls, image_path: str = None, metadata: dict = None) -> str:
        """
        Attempt to detect the imaging modality from path or metadata.

        Args:
            image_path: Path to the image file
            metadata: DICOM or other metadata dict

        Returns:
            Detected modality string
        """
        # Check metadata first
        if metadata:
            modality = metadata.get("Modality", "").upper()
            if modality in ["CR", "DX", "XR"]:
                return "xray"
            elif modality == "CT":
                return "ct"
            elif modality in ["MR", "MRI"]:
                return "mri"
            elif modality in ["US", "ULTRASOUND"]:
                return "ultrasound"

        # Check path for hints
        if image_path:
            path_lower = image_path.lower()
            if any(x in path_lower for x in ["xray", "chest", "cxr", "radiograph"]):
                return "xray"
            elif "ct" in path_lower or "computed" in path_lower:
                return "ct"
            elif "mri" in path_lower or "mr_" in path_lower:
                return "mri"
            elif any(x in path_lower for x in ["ultrasound", "us_", "sono", "busi"]):
                return "ultrasound"

        return "default"


# =============================================================================
# Utility: Progressive Domain Schedule
# =============================================================================

class ProgressiveDomainSchedule:
    """
    Schedule for progressively introducing harder domains during training.

    Implements curriculum learning at the domain level:
    1. Start with source domain (most data)
    2. Gradually introduce target domains
    3. Increase domain adversarial strength over time
    """

    def __init__(
        self,
        domains: List[str],
        source_domain: str,
        total_steps: int,
        warmup_ratio: float = 0.2
    ):
        """
        Args:
            domains: List of all domain names
            source_domain: Primary domain to start with
            total_steps: Total training steps
            warmup_ratio: Fraction of training before introducing all domains
        """
        self.domains = domains
        self.source_domain = source_domain
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_ratio)

        # Order domains: source first, then others
        self.domain_order = [source_domain] + [d for d in domains if d != source_domain]

    def get_domain_weights(self, current_step: int) -> Dict[str, float]:
        """
        Get sampling weights for each domain at current training step.

        Returns:
            Dict mapping domain name to sampling weight
        """
        progress = min(current_step / max(self.warmup_steps, 1), 1.0)

        weights = {}
        for i, domain in enumerate(self.domain_order):
            if i == 0:
                # Source domain: starts at 1.0, decreases to 0.5
                weights[domain] = 1.0 - 0.5 * progress
            else:
                # Target domains: start at 0, increase to equal weight
                weight = progress * (1.0 / len(self.domain_order))
                weights[domain] = weight

        # Normalize
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

        return weights

    def get_dann_lambda(self, current_step: int) -> float:
        """
        Get DANN lambda (gradient reversal strength) for current step.

        Uses the schedule from the DANN paper:
        lambda = 2 / (1 + exp(-10 * progress)) - 1
        """
        progress = current_step / max(self.total_steps, 1)
        return 2.0 / (1.0 + np.exp(-10 * progress)) - 1.0
