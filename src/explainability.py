"""
Enhanced Explainability Module for MedGamma.

IMPROVEMENTS (v2.0):
- Modality-aware visualization strategies
- Multi-layer attention aggregation
- Finding-specific attention maps (class-discriminative)
- Attention rollout for better flow visualization
- ROI validation against detected bounding boxes
- Confidence thresholding for cleaner visualizations
- Clinical explanation text generation
- Attention quality metrics

Usage:
    from src.explainability import MedGemmaExplainer
    explainer = MedGemmaExplainer(model_wrapper)
    results = explainer.explain_with_findings(image_path, prompt, findings)
"""

import torch
import numpy as np
import cv2
from PIL import Image
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# Try to import modality prompts for modality-specific processing
try:
    from src.prompts.modality_prompts import SUPPORTED_MODALITIES, MODALITY_CLASSES
    MODALITY_AWARE = True
except ImportError:
    MODALITY_AWARE = False
    SUPPORTED_MODALITIES = ["xray", "mri", "ct", "ultrasound"]


class AttentionAggregation(Enum):
    """Strategies for aggregating attention across layers."""
    LAST_LAYER = "last_layer"      # Use only the last layer (memory efficient)
    MEAN = "mean"                   # Average across all layers
    MAX = "max"                     # Maximum activation per patch
    ROLLOUT = "rollout"            # Attention rollout (tracks flow)
    WEIGHTED = "weighted"           # Gradient-weighted aggregation


@dataclass
class ExplainabilityResult:
    """Container for explainability outputs."""
    heatmap: np.ndarray                        # Main attention heatmap [H, W]
    overlay: np.ndarray                        # Heatmap overlaid on image [H, W, 3]
    finding_heatmaps: Dict[str, np.ndarray]    # Per-finding attention maps
    attention_peaks: List[Tuple[int, int]]     # Peak attention coordinates
    roi_coverage: float                        # % of attention in detected ROIs
    attention_quality: float                   # Quality score [0-1]
    clinical_explanation: str                  # Text explaining the visualization
    modality: str                              # Detected/specified modality


# =============================================================================
# Modality-Specific Visualization Configurations
# =============================================================================

MODALITY_VIS_CONFIG = {
    "xray": {
        "colormap": cv2.COLORMAP_INFERNO,
        "alpha": 0.5,
        "threshold": 0.3,           # Attention threshold for significance
        "blur_kernel": (7, 7),      # Larger blur for softer transitions
        "description": "chest radiograph attention",
        "anatomical_regions": ["cardiac", "pulmonary", "pleural", "mediastinal"],
    },
    "mri": {
        "colormap": cv2.COLORMAP_PLASMA,  # Better contrast for soft tissue
        "alpha": 0.6,
        "threshold": 0.25,
        "blur_kernel": (5, 5),
        "description": "brain parenchyma attention",
        "anatomical_regions": ["frontal", "temporal", "parietal", "occipital", "cerebellum"],
    },
    "ct": {
        "colormap": cv2.COLORMAP_MAGMA,
        "alpha": 0.55,
        "threshold": 0.28,
        "blur_kernel": (5, 5),
        "description": "CT density attention",
        "anatomical_regions": ["parenchyma", "bone", "soft_tissue", "vessels"],
    },
    "ultrasound": {
        "colormap": cv2.COLORMAP_VIRIDIS,  # Works well with grayscale US
        "alpha": 0.45,
        "threshold": 0.35,
        "blur_kernel": (5, 5),
        "description": "echogenicity attention",
        "anatomical_regions": ["lesion", "cyst", "solid_mass", "calcification"],
    },
}


class MedGemmaExplainer:
    """
    Enhanced explainability module for MedGemma with modality-awareness
    and finding-specific attention visualization.
    """

    def __init__(
        self,
        model_wrapper,
        aggregation: AttentionAggregation = AttentionAggregation.LAST_LAYER,
        multi_layer: bool = False
    ):
        """
        Initialize the explainer.

        Args:
            model_wrapper: MedGemmaWrapper instance
            aggregation: Attention aggregation strategy
            multi_layer: Whether to hook multiple layers (memory intensive)
        """
        self.model_wrapper = model_wrapper
        self.model = model_wrapper.model
        self.processor = model_wrapper.processor
        self.aggregation = aggregation
        self.multi_layer = multi_layer

        self.hooks = []
        self.attentions = []
        self.gradients = []
        self.layer_attentions = {}  # Store per-layer attention for aggregation

        if model_wrapper.model:
            self.model = model_wrapper.model
            self._register_hooks()
        else:
            self.model = None

    def _register_hooks(self):
        """
        Register hooks to capture attention maps and gradients.
        Supports both single-layer and multi-layer aggregation.
        """
        self._clear_hooks()

        def make_forward_hook(layer_idx: int):
            def forward_hook(module, input, output):
                if isinstance(output, tuple) and len(output) > 1:
                    attn = output[1].detach() if output[1] is not None else None
                elif isinstance(output, torch.Tensor):
                    attn = output.detach()
                else:
                    attn = None

                if attn is not None:
                    self.attentions.append(attn)
                    self.layer_attentions[layer_idx] = attn
            return forward_hook

        def make_backward_hook(layer_idx: int):
            def backward_hook(module, grad_input, grad_output):
                if isinstance(grad_output, tuple):
                    self.gradients.append(grad_output[0].detach())
                else:
                    self.gradients.append(grad_output.detach())
            return backward_hook

        # Find vision tower
        vision_tower = self._find_vision_tower()

        if vision_tower is None:
            print("DEBUG: Vision Tower NOT found. Explainability will be limited.")
            return

        try:
            layers = vision_tower.vision_model.encoder.layers

            if self.multi_layer:
                # Hook all layers for full attention rollout
                target_layers = list(enumerate(layers))
                print(f"DEBUG: Hooking all {len(layers)} vision layers for rollout")
            else:
                # Memory efficient: only last layer
                target_layers = [(len(layers) - 1, layers[-1])]
                print(f"DEBUG: Hooking last vision layer only")

            for layer_idx, layer in target_layers:
                self.hooks.append(
                    layer.self_attn.register_forward_hook(make_forward_hook(layer_idx))
                )
                self.hooks.append(
                    layer.self_attn.register_full_backward_hook(make_backward_hook(layer_idx))
                )
        except Exception as e:
            print(f"DEBUG: Error registering hooks: {e}")

    def _find_vision_tower(self):
        """Find vision tower through various model wrapping paths."""
        search_paths = [
            lambda m: m.model.vision_tower if hasattr(m, 'model') and hasattr(m.model, 'vision_tower') else None,
            lambda m: m.vision_tower if hasattr(m, 'vision_tower') else None,
            lambda m: m.base_model.model.model.vision_tower if hasattr(m, 'base_model') and hasattr(m.base_model, 'model') and hasattr(m.base_model.model, 'model') and hasattr(m.base_model.model.model, 'vision_tower') else None,
            lambda m: m.base_model.model.vision_tower if hasattr(m, 'base_model') and hasattr(m.base_model, 'model') and hasattr(m.base_model.model, 'vision_tower') else None,
        ]

        for path_fn in search_paths:
            try:
                vt = path_fn(self.model)
                if vt is not None:
                    return vt
            except:
                continue
        return None

    def _clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def _reset(self):
        """Reset captured attentions and gradients."""
        self.attentions = []
        self.gradients = []
        self.layer_attentions = {}

    # =========================================================================
    # Core Attention Computation
    # =========================================================================

    def _compute_attention_rollout(self, attentions: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute attention rollout to track information flow through layers.

        Attention rollout accounts for residual connections by computing
        the product of attention matrices across layers.

        Returns:
            Aggregated attention tensor [Patches]
        """
        if not attentions:
            return None

        # Start with identity + first attention
        rollout = None

        for attn in attentions:
            if attn.ndim != 4:
                continue

            # Average over heads: [B, H, L, L] -> [B, L, L]
            attn_avg = attn.mean(dim=1)

            # Add residual connection (identity matrix)
            eye = torch.eye(attn_avg.shape[-1], device=attn_avg.device)
            attn_residual = 0.5 * attn_avg + 0.5 * eye.unsqueeze(0)

            # Normalize rows
            attn_residual = attn_residual / attn_residual.sum(dim=-1, keepdim=True)

            if rollout is None:
                rollout = attn_residual
            else:
                rollout = torch.matmul(rollout, attn_residual)

        if rollout is None:
            return None

        # Extract attention to [CLS] or average across queries
        # Sum attention TO each patch (how much each patch is attended to)
        patch_attention = rollout.sum(dim=1).mean(dim=0)  # [Patches]

        return patch_attention

    def _compute_grad_weighted_attention(
        self,
        attentions: List[torch.Tensor],
        gradients: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute gradient-weighted attention for class-discriminative visualization.

        This weights attention maps by the gradient of the output w.r.t. features,
        highlighting regions that influenced the specific prediction.
        """
        if not attentions or not gradients:
            return None

        # Use last layer attention and gradient
        vision_att = None
        vision_grad = None

        for i in range(len(attentions) - 1, -1, -1):
            att = attentions[i]
            grad = gradients[i] if i < len(gradients) else None

            if att.ndim == 4:
                vision_att = att
                vision_grad = grad
                break

        if vision_att is None:
            return None

        # Average over heads: [B, H, L, L] -> [L]
        cam = vision_att.mean(dim=1).mean(dim=1).squeeze(0)

        if vision_grad is not None and vision_grad.ndim == 3:
            # Gradient magnitude per patch
            grad_map = vision_grad.norm(dim=-1).mean(dim=0)
            heatmap_vec = cam * grad_map
        else:
            heatmap_vec = cam

        return heatmap_vec

    def _aggregate_attention(
        self,
        attentions: List[torch.Tensor],
        gradients: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Aggregate attention based on configured strategy.
        """
        if self.aggregation == AttentionAggregation.ROLLOUT:
            return self._compute_attention_rollout(attentions)

        elif self.aggregation == AttentionAggregation.WEIGHTED:
            return self._compute_grad_weighted_attention(attentions, gradients)

        elif self.aggregation == AttentionAggregation.MEAN:
            valid_atts = [a for a in attentions if a.ndim == 4]
            if not valid_atts:
                return None
            # Stack and average
            stacked = torch.stack([
                a.mean(dim=1).mean(dim=1).squeeze(0) for a in valid_atts
            ])
            return stacked.mean(dim=0)

        elif self.aggregation == AttentionAggregation.MAX:
            valid_atts = [a for a in attentions if a.ndim == 4]
            if not valid_atts:
                return None
            stacked = torch.stack([
                a.mean(dim=1).mean(dim=1).squeeze(0) for a in valid_atts
            ])
            return stacked.max(dim=0)[0]

        else:  # LAST_LAYER (default)
            return self._compute_grad_weighted_attention(attentions, gradients)

    # =========================================================================
    # Heatmap Generation
    # =========================================================================

    def _vectormap_to_spatial(
        self,
        heatmap_vec: torch.Tensor,
        target_size: int = 1024,
        blur_kernel: Tuple[int, int] = (5, 5)
    ) -> np.ndarray:
        """
        Convert patch-level attention vector to spatial heatmap.
        """
        if heatmap_vec is None:
            return np.zeros((target_size, target_size))

        # Normalize to [0, 1]
        heatmap_vec = heatmap_vec.float()
        heatmap_vec = (heatmap_vec - heatmap_vec.min()) / (heatmap_vec.max() - heatmap_vec.min() + 1e-8)

        # Reshape to spatial grid
        num_patches = heatmap_vec.shape[0]
        side = int(np.sqrt(num_patches))

        if side * side != num_patches:
            # Handle non-square patch grids
            side = int(np.sqrt(num_patches))
            heatmap_vec = heatmap_vec[:side * side]

        heatmap = heatmap_vec.view(side, side).cpu().numpy()

        # Post-processing
        heatmap = cv2.GaussianBlur(heatmap, blur_kernel, 0)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        # Resize to target
        heatmap = cv2.resize(heatmap, (target_size, target_size), interpolation=cv2.INTER_CUBIC)

        return heatmap

    def _apply_threshold(self, heatmap: np.ndarray, threshold: float) -> np.ndarray:
        """
        Apply confidence threshold to filter out low-attention noise.
        """
        thresholded = heatmap.copy()
        thresholded[thresholded < threshold] = 0

        # Re-normalize after thresholding
        if thresholded.max() > 0:
            thresholded = thresholded / thresholded.max()

        return thresholded

    def _find_attention_peaks(
        self,
        heatmap: np.ndarray,
        min_distance: int = 50,
        threshold_rel: float = 0.5
    ) -> List[Tuple[int, int]]:
        """
        Find local maxima (peaks) in the attention heatmap.
        These represent regions of highest model focus.
        """
        from scipy import ndimage

        # Find local maxima
        data_max = ndimage.maximum_filter(heatmap, size=min_distance)
        maxima = (heatmap == data_max)

        # Apply threshold
        threshold = threshold_rel * heatmap.max()
        maxima = maxima & (heatmap > threshold)

        # Get coordinates
        peaks = list(zip(*np.where(maxima)))

        # Sort by attention value (descending)
        peaks = sorted(peaks, key=lambda p: heatmap[p[0], p[1]], reverse=True)

        return peaks[:10]  # Limit to top 10 peaks

    # =========================================================================
    # ROI Validation
    # =========================================================================

    def _calculate_roi_coverage(
        self,
        heatmap: np.ndarray,
        boxes: List[List[int]],
        threshold: float = 0.5
    ) -> float:
        """
        Calculate what percentage of high-attention regions fall within detected ROIs.

        This validates that the model is "looking at" the detected pathologies,
        ensuring explainability alignment.

        Args:
            heatmap: Attention heatmap [H, W]
            boxes: List of [x1, y1, x2, y2] normalized to 0-1000
            threshold: Attention threshold for "significant" attention

        Returns:
            Coverage ratio [0-1]
        """
        if not boxes or len(boxes) == 0:
            return 0.0

        h, w = heatmap.shape

        # Create ROI mask
        roi_mask = np.zeros((h, w), dtype=bool)

        for box in boxes:
            if len(box) != 4:
                continue
            x1, y1, x2, y2 = box
            # Normalize from 0-1000 to heatmap size
            x1_px = int(x1 * w / 1000)
            y1_px = int(y1 * h / 1000)
            x2_px = int(x2 * w / 1000)
            y2_px = int(y2 * h / 1000)

            # Ensure valid bounds
            x1_px = max(0, min(x1_px, w - 1))
            x2_px = max(0, min(x2_px, w - 1))
            y1_px = max(0, min(y1_px, h - 1))
            y2_px = max(0, min(y2_px, h - 1))

            roi_mask[y1_px:y2_px, x1_px:x2_px] = True

        # Calculate attention in ROI vs total
        significant_attention = heatmap > (threshold * heatmap.max())

        attention_in_roi = np.sum(heatmap[roi_mask & significant_attention])
        total_significant = np.sum(heatmap[significant_attention])

        if total_significant == 0:
            return 0.0

        return float(attention_in_roi / total_significant)

    def _calculate_attention_quality(
        self,
        heatmap: np.ndarray,
        peaks: List[Tuple[int, int]]
    ) -> float:
        """
        Calculate attention quality score based on:
        - Sparsity: Good attention is focused, not diffuse
        - Peak strength: Clear peaks indicate confident focus
        - Distribution: Attention should have clear hotspots

        Returns:
            Quality score [0-1]
        """
        # Sparsity: percentage of image with significant attention
        threshold = 0.3 * heatmap.max()
        significant_area = np.sum(heatmap > threshold) / heatmap.size
        sparsity_score = 1.0 - min(significant_area, 1.0)

        # Peak strength: how strong are the peaks relative to mean
        if len(peaks) > 0:
            peak_values = [heatmap[p[0], p[1]] for p in peaks]
            mean_peak = np.mean(peak_values)
            mean_overall = np.mean(heatmap)
            contrast = (mean_peak - mean_overall) / (mean_overall + 1e-8)
            peak_score = min(contrast / 5.0, 1.0)  # Normalize
        else:
            peak_score = 0.0

        # Distribution: entropy-based (lower is better for focused attention)
        hist, _ = np.histogram(heatmap.flatten(), bins=50, density=True)
        hist = hist + 1e-8
        entropy = -np.sum(hist * np.log2(hist))
        max_entropy = np.log2(50)
        distribution_score = 1.0 - (entropy / max_entropy)

        # Combined score
        quality = 0.3 * sparsity_score + 0.4 * peak_score + 0.3 * distribution_score

        return float(np.clip(quality, 0, 1))

    # =========================================================================
    # Clinical Explanation Generation
    # =========================================================================

    def _generate_clinical_explanation(
        self,
        modality: str,
        findings: List[Dict],
        peaks: List[Tuple[int, int]],
        roi_coverage: float,
        attention_quality: float
    ) -> str:
        """
        Generate clinical explanation text describing what the attention map shows.
        """
        config = MODALITY_VIS_CONFIG.get(modality, MODALITY_VIS_CONFIG["xray"])

        explanation_parts = []

        # Header
        explanation_parts.append(
            f"Attention Analysis ({config['description'].title()})"
        )

        # Attention quality assessment
        if attention_quality > 0.7:
            quality_text = "High-confidence attention patterns detected"
        elif attention_quality > 0.4:
            quality_text = "Moderate attention focus observed"
        else:
            quality_text = "Diffuse attention - interpretation requires caution"
        explanation_parts.append(f"Quality: {quality_text}")

        # Peak regions
        if peaks:
            num_peaks = len(peaks)
            explanation_parts.append(
                f"Identified {num_peaks} region(s) of focused attention"
            )

        # ROI coverage
        if roi_coverage > 0.8:
            coverage_text = "Excellent alignment - model attention strongly correlates with detected findings"
        elif roi_coverage > 0.5:
            coverage_text = "Good alignment - attention overlaps with detected regions"
        elif roi_coverage > 0.2:
            coverage_text = "Partial alignment - some attention falls outside detected ROIs"
        else:
            coverage_text = "Limited alignment - attention distribution differs from detected regions"
        explanation_parts.append(f"ROI Coverage: {coverage_text} ({roi_coverage:.0%})")

        # Finding-specific notes
        if findings:
            finding_classes = [f.get("class", "unknown") for f in findings if f.get("class") != "No significant abnormality"]
            if finding_classes:
                explanation_parts.append(
                    f"Detected pathologies: {', '.join(finding_classes[:3])}"
                )

        return "\n".join(explanation_parts)

    # =========================================================================
    # Main Explain Methods
    # =========================================================================

    def explain(
        self,
        image_path: str,
        prompt_text: str,
        target_text: str = "detected",
        modality: str = "xray"
    ) -> np.ndarray:
        """
        Generate attention heatmap (legacy interface for backwards compatibility).

        Returns:
            Heatmap as numpy array [1024, 1024]
        """
        self._reset()

        # Ensure hooks are registered
        if not self.hooks:
            self.model = self.model_wrapper.model
            if self.model:
                print("DEBUG: Registering Explainability Hooks (Lazy Load)...")
                self._register_hooks()
            else:
                print("(!) Model not loaded. Cannot explain.")
                return np.zeros((1024, 1024))

        # Forward pass with gradients
        outputs, inputs = self.model_wrapper.run_forward_with_grad(
            image_path, prompt_text, target_size=512
        )

        # Backward pass
        logits = outputs.logits
        predicted_token_logits, _ = logits.max(dim=-1)
        score = predicted_token_logits.sum()
        score.backward()

        # Aggregate attention
        heatmap_vec = self._aggregate_attention(self.attentions, self.gradients)

        # Get modality-specific config
        config = MODALITY_VIS_CONFIG.get(modality, MODALITY_VIS_CONFIG["xray"])

        # Convert to spatial heatmap
        heatmap = self._vectormap_to_spatial(
            heatmap_vec,
            target_size=1024,
            blur_kernel=config["blur_kernel"]
        )

        return heatmap

    def explain_with_findings(
        self,
        image_path: str,
        prompt_text: str,
        findings: List[Dict] = None,
        modality: str = "xray"
    ) -> ExplainabilityResult:
        """
        Generate comprehensive explainability output with findings alignment.

        This is the recommended method for clinical use as it validates
        attention against detected pathologies.

        Args:
            image_path: Path to medical image
            prompt_text: Detection prompt used
            findings: List of detected findings with boxes
            modality: Imaging modality

        Returns:
            ExplainabilityResult with all analysis outputs
        """
        findings = findings or []

        # Generate base heatmap
        heatmap = self.explain(image_path, prompt_text, modality=modality)

        # Get modality config
        config = MODALITY_VIS_CONFIG.get(modality, MODALITY_VIS_CONFIG["xray"])

        # Apply threshold
        thresholded_heatmap = self._apply_threshold(heatmap, config["threshold"])

        # Find attention peaks
        peaks = self._find_attention_peaks(thresholded_heatmap)

        # Extract boxes from findings
        boxes = [f.get("box", []) for f in findings if f.get("box")]

        # Calculate ROI coverage
        roi_coverage = self._calculate_roi_coverage(heatmap, boxes, config["threshold"])

        # Calculate attention quality
        attention_quality = self._calculate_attention_quality(heatmap, peaks)

        # Generate per-finding heatmaps (approximate: highlight box region)
        finding_heatmaps = {}
        for finding in findings:
            cls = finding.get("class", "unknown")
            box = finding.get("box", [])
            if box and len(box) == 4:
                # Create masked heatmap focusing on this finding's region
                finding_mask = self._create_box_mask(heatmap.shape, box)
                finding_heatmaps[cls] = heatmap * finding_mask

        # Generate overlay visualization
        overlay, _ = self.visualize(image_path, thresholded_heatmap, modality=modality)

        # Generate clinical explanation
        clinical_explanation = self._generate_clinical_explanation(
            modality, findings, peaks, roi_coverage, attention_quality
        )

        return ExplainabilityResult(
            heatmap=heatmap,
            overlay=overlay,
            finding_heatmaps=finding_heatmaps,
            attention_peaks=peaks,
            roi_coverage=roi_coverage,
            attention_quality=attention_quality,
            clinical_explanation=clinical_explanation,
            modality=modality
        )

    def _create_box_mask(
        self,
        shape: Tuple[int, int],
        box: List[int]
    ) -> np.ndarray:
        """Create a soft mask for a bounding box region."""
        h, w = shape
        mask = np.zeros((h, w), dtype=np.float32)

        x1, y1, x2, y2 = box
        x1_px = int(x1 * w / 1000)
        y1_px = int(y1 * h / 1000)
        x2_px = int(x2 * w / 1000)
        y2_px = int(y2 * h / 1000)

        # Clamp to valid range
        x1_px = max(0, min(x1_px, w - 1))
        x2_px = max(0, min(x2_px, w - 1))
        y1_px = max(0, min(y1_px, h - 1))
        y2_px = max(0, min(y2_px, h - 1))

        mask[y1_px:y2_px, x1_px:x2_px] = 1.0

        # Soft edges via Gaussian blur
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        mask = mask / (mask.max() + 1e-8)

        return mask

    def visualize(
        self,
        image_path: str,
        heatmap: np.ndarray,
        alpha: float = None,
        modality: str = "xray"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Overlay heatmap on original image with modality-appropriate colormap.

        Args:
            image_path: Path to original image
            heatmap: Attention heatmap
            alpha: Blend factor (default: modality-specific)
            modality: Imaging modality

        Returns:
            Tuple of (overlay image, colored heatmap)
        """
        config = MODALITY_VIS_CONFIG.get(modality, MODALITY_VIS_CONFIG["xray"])

        if alpha is None:
            alpha = config["alpha"]

        # Load and resize image
        img = cv2.imread(image_path)
        if img is None:
            # Try PIL for more formats
            pil_img = Image.open(image_path).convert("RGB")
            img = np.array(pil_img)[:, :, ::-1]  # RGB to BGR for OpenCV

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (1024, 1024))

        # Normalize heatmap
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap_uint8 = np.uint8(255 * heatmap_norm)

        # Apply modality-specific colormap
        try:
            heatmap_color = cv2.applyColorMap(heatmap_uint8, config["colormap"])
        except Exception:
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Blend
        overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)

        return overlay, heatmap_color

    def visualize_with_boxes(
        self,
        image_path: str,
        heatmap: np.ndarray,
        findings: List[Dict],
        modality: str = "xray",
        show_labels: bool = True
    ) -> np.ndarray:
        """
        Overlay heatmap AND draw bounding boxes with labels.

        This provides a complete visualization showing both model attention
        and detected pathology regions.
        """
        # Get base overlay
        overlay, _ = self.visualize(image_path, heatmap, modality=modality)

        # Draw bounding boxes
        for finding in findings:
            cls = finding.get("class", "")
            box = finding.get("box", [])

            if not box or len(box) != 4 or cls == "No significant abnormality":
                continue

            x1, y1, x2, y2 = box

            # Normalize from 0-1000 to 1024
            h, w = overlay.shape[:2]
            x1_px = int(x1 * w / 1000)
            y1_px = int(y1 * h / 1000)
            x2_px = int(x2 * w / 1000)
            y2_px = int(y2 * h / 1000)

            # Draw rectangle
            color = (0, 255, 0)  # Green
            cv2.rectangle(overlay, (x1_px, y1_px), (x2_px, y2_px), color, 2)

            # Draw label
            if show_labels:
                label = cls[:20]  # Truncate long labels
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2

                # Get text size for background
                (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

                # Draw background rectangle
                cv2.rectangle(
                    overlay,
                    (x1_px, y1_px - text_h - 10),
                    (x1_px + text_w + 4, y1_px),
                    color,
                    -1
                )

                # Draw text
                cv2.putText(
                    overlay,
                    label,
                    (x1_px + 2, y1_px - 5),
                    font,
                    font_scale,
                    (0, 0, 0),  # Black text
                    thickness
                )

        return overlay

    def cleanup(self):
        """Clean up hooks and free resources."""
        self._clear_hooks()
        self._reset()


# =============================================================================
# Utility Functions
# =============================================================================

def create_attention_summary_image(
    results: ExplainabilityResult,
    image_path: str,
    save_path: str = None
) -> np.ndarray:
    """
    Create a summary image with multiple panels:
    - Original image
    - Attention heatmap
    - Overlay with boxes
    - Quality metrics

    Args:
        results: ExplainabilityResult from explain_with_findings
        image_path: Path to original image
        save_path: Optional path to save the summary image

    Returns:
        Summary image as numpy array
    """
    # Load original
    original = cv2.imread(image_path)
    if original is None:
        pil_img = Image.open(image_path).convert("RGB")
        original = np.array(pil_img)[:, :, ::-1]
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    original = cv2.resize(original, (512, 512))

    # Resize overlay
    overlay = cv2.resize(results.overlay, (512, 512))

    # Create heatmap visualization
    heatmap_vis = (results.heatmap * 255).astype(np.uint8)
    heatmap_vis = cv2.resize(heatmap_vis, (512, 512))
    heatmap_color = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_INFERNO)

    # Create metrics panel
    metrics_panel = np.zeros((512, 512, 3), dtype=np.uint8)

    text_lines = [
        f"Modality: {results.modality.upper()}",
        f"ROI Coverage: {results.roi_coverage:.1%}",
        f"Attention Quality: {results.attention_quality:.2f}",
        f"Attention Peaks: {len(results.attention_peaks)}",
        "",
        "Clinical Summary:",
    ] + results.clinical_explanation.split("\n")

    y_offset = 30
    for line in text_lines[:12]:  # Limit lines
        cv2.putText(
            metrics_panel,
            line[:50],  # Truncate long lines
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        y_offset += 25

    # Stack panels (2x2 grid)
    top_row = np.hstack([original, heatmap_color])
    bottom_row = np.hstack([overlay, metrics_panel])
    summary = np.vstack([top_row, bottom_row])

    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(summary, cv2.COLOR_RGB2BGR))

    return summary
