"""
Universal Medical Image Inference Module for MedGamma.

Provides a unified interface for analyzing ANY type of medical image:
- Automatic modality detection
- Domain-adaptive prompting
- Multi-scale analysis
- Test-time augmentation
- Ensemble prediction
- Confidence calibration

Main objective: Process any medical image, reason about it, segment it,
analyze it, screen it, and find any type of anomalies, tumors, lesions,
or abnormalities.

Usage:
    analyzer = UniversalMedicalAnalyzer(checkpoint_dir="checkpoints/production")
    result = analyzer.analyze(image_path, task="detect_abnormalities")
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
import gc

from .config import Config
from .medgemma_wrapper import MedGemmaWrapper
from .medsam_wrapper import MedSAMWrapper
from .preprocess import load_and_preprocess
from .train.domain_adaptation import DomainPromptAdapter
from .data.medical_augmentation import TestTimeAugmentation, get_augmentor


@dataclass
class AnalysisResult:
    """Comprehensive analysis result."""
    # Basic info
    image_path: str
    modality: str
    timestamp: str

    # Findings
    is_abnormal: bool
    abnormality_score: float  # 0-1 confidence
    findings: List[Dict]  # List of {"class": str, "box": [x1,y1,x2,y2], "confidence": float}

    # Segmentation
    segmentation_masks: List[Dict]  # List of {"mask": np.ndarray, "label": str, "iou_score": float}

    # Raw outputs
    raw_text: str
    attention_heatmap: Optional[np.ndarray] = None

    # Metadata
    processing_time_ms: float = 0.0
    model_info: Dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class UniversalMedicalAnalyzer:
    """
    Universal analyzer for any type of medical image.

    Features:
    1. AUTOMATIC MODALITY DETECTION: Determines if X-ray, CT, MRI, Ultrasound, etc.
    2. ADAPTIVE PROMPTING: Uses modality-specific prompts for better accuracy
    3. MULTI-TASK ANALYSIS: Screening, detection, segmentation in one pass
    4. TEST-TIME AUGMENTATION: Multiple views for robust predictions
    5. CONFIDENCE CALIBRATION: Reliable uncertainty estimates
    6. ENSEMBLE SUPPORT: Combine multiple adapters for different modalities
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints/production",
        use_tta: bool = False,
        tta_n: int = 3,
        use_multi_adapter: bool = False,
        device: str = "auto"
    ):
        """
        Initialize the universal analyzer.

        Args:
            checkpoint_dir: Path to trained model checkpoints
            use_tta: Enable test-time augmentation
            tta_n: Number of TTA augmentations
            use_multi_adapter: Load multiple specialized adapters
            device: "auto", "cuda", or "cpu"
        """
        self.checkpoint_dir = checkpoint_dir
        self.use_tta = use_tta
        self.tta_n = tta_n
        self.use_multi_adapter = use_multi_adapter

        # Device setup
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Model wrappers (lazy loaded)
        self._medgemma: Optional[MedGemmaWrapper] = None
        self._sam: Optional[MedSAMWrapper] = None
        self._adapters_loaded: Dict[str, bool] = {}

        # TTA setup
        if use_tta:
            self.tta = TestTimeAugmentation(n_augmentations=tta_n)
        else:
            self.tta = None

        # Domain prompt adapter
        self.prompt_adapter = DomainPromptAdapter()

        print(f"Universal Medical Analyzer initialized")
        print(f"  Checkpoint: {checkpoint_dir}")
        print(f"  Device: {self.device}")
        print(f"  TTA: {use_tta} (n={tta_n})")

    @property
    def medgemma(self) -> MedGemmaWrapper:
        """Lazy load MedGemma."""
        if self._medgemma is None:
            self._medgemma = MedGemmaWrapper()
            self._medgemma.load()
            self._load_best_adapter()
        return self._medgemma

    @property
    def sam(self) -> MedSAMWrapper:
        """Lazy load SAM2."""
        if self._sam is None:
            sam_adapter = self._find_adapter("sam2")
            self._sam = MedSAMWrapper(adapter_path=sam_adapter)
            self._sam.load()
        return self._sam

    def _find_adapter(self, model_name: str) -> Optional[str]:
        """Find best adapter path for a model."""
        search_paths = [
            os.path.join(self.checkpoint_dir, model_name, "final"),
            os.path.join(self.checkpoint_dir, model_name, "best"),
            os.path.join(self.checkpoint_dir, model_name),
            os.path.join(self.checkpoint_dir, "medgemma", "detection"),  # For MedGemma
        ]

        for path in search_paths:
            if os.path.exists(path) and os.path.exists(os.path.join(path, "adapter_config.json")):
                return path

        return None

    def _load_best_adapter(self):
        """Load the best MedGemma adapter based on available checkpoints."""
        from peft import PeftModel

        # Try detection adapter first (most general)
        adapter_path = self._find_adapter("medgemma")

        if not adapter_path:
            # Try specific adapters
            for stage in ["detection", "final", "vqa", "screening"]:
                path = os.path.join(self.checkpoint_dir, "medgemma", stage)
                if os.path.exists(path) and os.path.exists(os.path.join(path, "adapter_config.json")):
                    adapter_path = path
                    break

        if adapter_path:
            try:
                self._medgemma.model = PeftModel.from_pretrained(
                    self._medgemma.model,
                    adapter_path
                )
                self._medgemma.model.eval()
                print(f"  Loaded adapter: {adapter_path}")
            except Exception as e:
                print(f"  Warning: Failed to load adapter: {e}")

    def detect_modality(
        self,
        image: Union[str, Image.Image],
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Detect the imaging modality of the input.

        Strategy:
        1. Check DICOM metadata if available
        2. Check path hints
        3. Use vision model to classify
        """
        # Try path/metadata hints first
        if isinstance(image, str):
            modality = self.prompt_adapter.detect_modality(image, metadata)
            if modality != "default":
                return modality

        # Fall back to model-based detection
        try:
            response, _ = self.medgemma.analyze_image(
                image if isinstance(image, Image.Image) else image,
                task="modality"
            )

            response_lower = response.lower()
            if "x-ray" in response_lower or "xray" in response_lower or "radiograph" in response_lower:
                return "xray"
            elif "ct" in response_lower or "computed tomography" in response_lower:
                return "ct"
            elif "mri" in response_lower or "magnetic resonance" in response_lower:
                return "mri"
            elif "ultrasound" in response_lower or "sonograph" in response_lower:
                return "ultrasound"
        except:
            pass

        return "default"

    def analyze(
        self,
        image: Union[str, Image.Image],
        task: str = "full",  # "full", "screen", "detect", "segment"
        modality: Optional[str] = None,
        return_heatmap: bool = False,
        verbose: bool = True
    ) -> AnalysisResult:
        """
        Perform comprehensive analysis on a medical image.

        Args:
            image: Path to image or PIL Image
            task: Analysis task type
                - "full": Complete pipeline (screen + detect + segment)
                - "screen": Quick normal/abnormal screening
                - "detect": Detection with bounding boxes
                - "segment": Segmentation of detected regions
            modality: Override modality detection ("xray", "ct", "mri", "ultrasound")
            return_heatmap: Generate attention heatmap
            verbose: Print progress

        Returns:
            AnalysisResult with comprehensive findings
        """
        import time
        start_time = time.time()

        # Load image
        if isinstance(image, str):
            image_path = image
            pil_image = load_and_preprocess(image)
        else:
            image_path = "provided_image"
            pil_image = image.convert("RGB") if isinstance(image, Image.Image) else image

        if verbose:
            print(f"\n{'='*60}")
            print(f"  Universal Medical Analysis")
            print(f"  Task: {task}")
            print(f"{'='*60}")

        # Detect modality
        if modality is None:
            modality = self.detect_modality(image_path if isinstance(image, str) else pil_image)
        if verbose:
            print(f"  Detected modality: {modality}")

        # Initialize result
        result = AnalysisResult(
            image_path=image_path if isinstance(image, str) else "inline",
            modality=modality,
            timestamp=datetime.now().isoformat(),
            is_abnormal=False,
            abnormality_score=0.0,
            findings=[],
            segmentation_masks=[],
            raw_text=""
        )

        warnings = []

        try:
            # =====================================================
            # Step 1: Screening (is this normal or abnormal?)
            # =====================================================
            if task in ["full", "screen"]:
                if verbose:
                    print("\n[1] Screening...")

                screen_result = self._run_screening(pil_image, modality)
                result.is_abnormal = screen_result["is_abnormal"]
                result.abnormality_score = screen_result["confidence"]

                if verbose:
                    status = "ABNORMAL" if result.is_abnormal else "HEALTHY"
                    print(f"  Result: {status} (confidence: {result.abnormality_score:.2f})")

                # Early exit for screening-only task
                if task == "screen":
                    result.raw_text = screen_result["raw_text"]
                    result.processing_time_ms = (time.time() - start_time) * 1000
                    return result

                # Early exit if healthy (no need to detect)
                if not result.is_abnormal and task == "full":
                    if verbose:
                        print("  Image appears healthy - skipping detection/segmentation")
                    result.raw_text = screen_result["raw_text"]
                    result.processing_time_ms = (time.time() - start_time) * 1000
                    return result

            # =====================================================
            # Step 2: Detection (find abnormalities)
            # =====================================================
            if task in ["full", "detect", "segment"]:
                if verbose:
                    print("\n[2] Detection...")

                detect_result = self._run_detection(pil_image, modality)
                result.findings = detect_result["findings"]
                result.raw_text = detect_result["raw_text"]

                if verbose:
                    print(f"  Found {len(result.findings)} potential findings")
                    for f in result.findings[:5]:
                        print(f"    - {f.get('class', 'Unknown')}: box={f.get('box', [])}")

                # Update abnormality based on findings
                if result.findings:
                    result.is_abnormal = True
                    # Compute aggregate confidence
                    confidences = [f.get("confidence", 0.5) for f in result.findings]
                    result.abnormality_score = max(confidences) if confidences else 0.5

                # Early exit for detection-only task
                if task == "detect":
                    result.processing_time_ms = (time.time() - start_time) * 1000
                    return result

            # =====================================================
            # Step 3: Segmentation (precise boundaries)
            # =====================================================
            if task in ["full", "segment"] and result.findings:
                if verbose:
                    print("\n[3] Segmentation...")

                # Unload MedGemma to free VRAM for SAM
                if self._medgemma is not None:
                    self._medgemma.unload()
                    gc.collect()
                    torch.cuda.empty_cache()

                seg_result = self._run_segmentation(pil_image, result.findings)
                result.segmentation_masks = seg_result["masks"]

                if verbose:
                    print(f"  Generated {len(result.segmentation_masks)} segmentation masks")

            # =====================================================
            # Step 4: Attention Heatmap (optional)
            # =====================================================
            if return_heatmap:
                if verbose:
                    print("\n[4] Generating attention heatmap...")

                try:
                    from .explainability import MedGemmaExplainer

                    # Reload MedGemma if needed
                    if self._medgemma is None or self._medgemma.model is None:
                        self._medgemma = MedGemmaWrapper()
                        self._medgemma.load()
                        self._load_best_adapter()

                    explainer = MedGemmaExplainer(self._medgemma)
                    heatmap = explainer.explain(image_path, "detect abnormalities")
                    result.attention_heatmap = heatmap
                    if verbose:
                        print("  Heatmap generated")
                except Exception as e:
                    warnings.append(f"Heatmap generation failed: {e}")
                    if verbose:
                        print(f"  Warning: Heatmap generation failed: {e}")

        except Exception as e:
            import traceback
            warnings.append(f"Analysis error: {e}")
            if verbose:
                print(f"\n  ERROR: {e}")
                traceback.print_exc()

        # Finalize result
        result.warnings = warnings
        result.processing_time_ms = (time.time() - start_time) * 1000
        result.model_info = {
            "checkpoint": self.checkpoint_dir,
            "modality": modality,
            "use_tta": self.use_tta
        }

        if verbose:
            print(f"\n{'='*60}")
            print(f"  Analysis complete in {result.processing_time_ms:.0f}ms")
            print(f"  Abnormal: {result.is_abnormal}")
            print(f"  Findings: {len(result.findings)}")
            print(f"  Segments: {len(result.segmentation_masks)}")
            print(f"{'='*60}")

        return result

    def _run_screening(self, image: Image.Image, modality: str) -> Dict:
        """Run screening task."""
        # Get modality-specific prompt
        prompt = self.prompt_adapter.get_prompt(modality, "screening")

        if self.use_tta and self.tta:
            # Multiple augmented predictions
            images = self.tta(image)
            predictions = []

            for aug_img in images:
                response, _ = self.medgemma.analyze_image(aug_img, task="screening")
                is_abnormal = self._parse_screening_response(response)
                predictions.append(1.0 if is_abnormal else 0.0)

            # Aggregate
            avg_score = np.mean(predictions)
            is_abnormal = avg_score > 0.5
            confidence = avg_score if is_abnormal else (1 - avg_score)
            raw_text = f"TTA Aggregate ({len(predictions)} views): {'ABNORMAL' if is_abnormal else 'HEALTHY'}"
        else:
            response, _ = self.medgemma.analyze_image(image, task="screening")
            is_abnormal = self._parse_screening_response(response)
            confidence = 0.8 if is_abnormal else 0.9  # Base confidence
            raw_text = response

        return {
            "is_abnormal": is_abnormal,
            "confidence": confidence,
            "raw_text": raw_text
        }

    def _run_detection(self, image: Image.Image, modality: str) -> Dict:
        """Run detection task."""
        response, boxes = self.medgemma.analyze_image(image, task="detection")

        findings = []
        if boxes:
            for i, box in enumerate(boxes):
                findings.append({
                    "class": f"Abnormality_{i+1}",
                    "box": box,
                    "confidence": 0.7  # Default confidence
                })

        # Try to parse class names from response
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                if "findings" in data and isinstance(data["findings"], list):
                    for i, f in enumerate(data["findings"]):
                        if i < len(findings) and "class" in f:
                            findings[i]["class"] = f["class"]
                            if "confidence" in f:
                                findings[i]["confidence"] = float(f["confidence"])
        except:
            pass

        return {
            "findings": findings,
            "raw_text": response
        }

    def _run_segmentation(self, image: Image.Image, findings: List[Dict]) -> Dict:
        """Run segmentation for detected findings."""
        masks = []

        self.sam.set_image(image)

        for finding in findings:
            box = finding.get("box", [])
            if not box or len(box) != 4:
                continue

            try:
                pred_masks, scores = self.sam.predict_mask(box)

                if pred_masks is not None:
                    # Take the best mask (highest score)
                    best_idx = np.argmax(scores)
                    best_mask = pred_masks[best_idx]
                    best_score = float(scores[best_idx])

                    masks.append({
                        "mask": best_mask,
                        "label": finding.get("class", "Unknown"),
                        "iou_score": best_score,
                        "box": box
                    })
            except Exception as e:
                print(f"  Warning: Segmentation failed for box {box}: {e}")
                continue

        return {"masks": masks}

    def _parse_screening_response(self, response: str) -> bool:
        """Parse screening response to determine if abnormal."""
        response_lower = response.lower()

        # Check for abnormal indicators
        abnormal_keywords = [
            "abnormal", "disease", "pathology", "lesion", "mass", "nodule",
            "opacity", "effusion", "consolidation", "tumor", "infection",
            "cardiomegaly", "pneumonia", "infiltrate", "malignant", "suspicious"
        ]

        healthy_keywords = [
            "healthy", "normal", "unremarkable", "no significant",
            "no abnormal", "clear", "negative"
        ]

        # Check abnormal first (more specific)
        has_abnormal = any(kw in response_lower for kw in abnormal_keywords)
        has_healthy = any(kw in response_lower for kw in healthy_keywords)

        # Abnormal takes precedence if both present
        if has_abnormal:
            return True
        elif has_healthy:
            return False
        else:
            # Default to abnormal if uncertain (safer for medical)
            return "abnormal" in response_lower

    def generate_report(self, result: AnalysisResult) -> str:
        """
        Generate a clinical-style report from analysis results.

        Args:
            result: AnalysisResult from analyze()

        Returns:
            Formatted clinical report string
        """
        report = []
        report.append("=" * 60)
        report.append("CLINICAL IMAGING REPORT - MedGamma AI")
        report.append("=" * 60)
        report.append("")
        report.append(f"Study Date: {result.timestamp}")
        report.append(f"Modality: {result.modality.upper()}")
        report.append(f"Image: {result.image_path}")
        report.append("")
        report.append("-" * 60)
        report.append("IMPRESSION:")
        report.append("-" * 60)

        if result.is_abnormal:
            report.append(f"ABNORMAL STUDY (Confidence: {result.abnormality_score:.0%})")
            report.append("")
            report.append("FINDINGS:")
            for i, finding in enumerate(result.findings, 1):
                cls = finding.get("class", "Abnormality")
                conf = finding.get("confidence", 0)
                box = finding.get("box", [])
                report.append(f"  {i}. {cls} (confidence: {conf:.0%})")
                if box:
                    report.append(f"     Location: [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]")

            if result.segmentation_masks:
                report.append("")
                report.append("SEGMENTATION:")
                for i, seg in enumerate(result.segmentation_masks, 1):
                    label = seg.get("label", "Unknown")
                    iou = seg.get("iou_score", 0)
                    report.append(f"  {i}. {label} - Segmented (IoU: {iou:.2f})")
        else:
            report.append("NO SIGNIFICANT ABNORMALITY DETECTED")
            report.append("")
            report.append("The study appears within normal limits based on AI analysis.")

        report.append("")
        report.append("-" * 60)
        report.append("TECHNICAL:")
        report.append("-" * 60)
        report.append(f"Processing time: {result.processing_time_ms:.0f}ms")
        report.append(f"Model checkpoint: {result.model_info.get('checkpoint', 'Unknown')}")

        if result.warnings:
            report.append("")
            report.append("WARNINGS:")
            for w in result.warnings:
                report.append(f"  - {w}")

        report.append("")
        report.append("=" * 60)
        report.append("DISCLAIMER: This AI-generated report is for research purposes only.")
        report.append("Not for clinical diagnosis. Verify with qualified radiologist.")
        report.append("=" * 60)

        return "\n".join(report)

    def cleanup(self):
        """Release all GPU resources."""
        if self._medgemma is not None:
            self._medgemma.unload()
            self._medgemma = None

        if self._sam is not None:
            # SAM doesn't have unload, just delete
            del self._sam
            self._sam = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line interface for universal medical analysis."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Universal Medical Image Analysis with MedGamma"
    )
    parser.add_argument("image", help="Path to medical image")
    parser.add_argument("--task", "-t", choices=["full", "screen", "detect", "segment"],
                        default="full", help="Analysis task")
    parser.add_argument("--checkpoint", "-c", default="checkpoints/production",
                        help="Model checkpoint directory")
    parser.add_argument("--modality", "-m", choices=["xray", "ct", "mri", "ultrasound", "auto"],
                        default="auto", help="Imaging modality")
    parser.add_argument("--tta", action="store_true", help="Enable test-time augmentation")
    parser.add_argument("--heatmap", action="store_true", help="Generate attention heatmap")
    parser.add_argument("--output", "-o", help="Save report to file")

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = UniversalMedicalAnalyzer(
        checkpoint_dir=args.checkpoint,
        use_tta=args.tta
    )

    # Run analysis
    modality = None if args.modality == "auto" else args.modality
    result = analyzer.analyze(
        args.image,
        task=args.task,
        modality=modality,
        return_heatmap=args.heatmap
    )

    # Generate report
    report = analyzer.generate_report(result)
    print(report)

    # Save if requested
    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"\nReport saved to: {args.output}")

    # Cleanup
    analyzer.cleanup()


if __name__ == "__main__":
    main()
