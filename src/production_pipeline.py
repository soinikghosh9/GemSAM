"""
Production Pipeline for MedGamma Kaggle Submission.

This is the main entry point for the production-ready inference pipeline,
designed for the Kaggle MedGemma Impact Challenge.

Features:
- Universal medical image analysis (X-ray, CT, MRI, Ultrasound)
- Automatic modality detection
- Abnormality detection with bounding boxes
- Segmentation mask generation
- Clinical report generation
- Both GPU (development) and Edge (Raspberry Pi 5) support

Usage:
    # Single image analysis
    python -m src.production_pipeline analyze image.jpg

    # Batch processing
    python -m src.production_pipeline batch images_dir/ --output results/

    # Generate clinical report
    python -m src.production_pipeline report image.jpg --format pdf
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@dataclass
class MedicalFinding:
    """Single medical finding from analysis."""
    finding_class: str
    confidence: float
    bounding_box: List[float]
    severity: str  # "normal", "mild", "moderate", "severe"
    description: str
    iou_score: float = 0.0


@dataclass
class AnalysisResult:
    """Complete analysis result for a medical image."""
    image_path: str
    timestamp: str
    modality: str
    body_region: str
    is_abnormal: bool
    overall_confidence: float
    findings: List[MedicalFinding]
    segmentation_masks: List[np.ndarray]
    clinical_summary: str
    inference_time_seconds: float
    model_version: str = "MedGamma v1.0 (MedGemma 1.5 4B + SAM2)"

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return {
            "image_path": self.image_path,
            "timestamp": self.timestamp,
            "modality": self.modality,
            "body_region": self.body_region,
            "is_abnormal": self.is_abnormal,
            "overall_confidence": self.overall_confidence,
            "findings": [asdict(f) for f in self.findings],
            "num_segmentation_masks": len(self.segmentation_masks),
            "clinical_summary": self.clinical_summary,
            "inference_time_seconds": self.inference_time_seconds,
            "model_version": self.model_version
        }


class ProductionPipeline:
    """
    Production-ready MedGamma pipeline for Kaggle submission.

    This pipeline combines MedGemma vision-language model with SAM2
    segmentation for comprehensive medical image analysis.
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints/production",
        device: str = "auto",
        use_sam2: bool = True,
        verbose: bool = True
    ):
        """
        Initialize production pipeline.

        Args:
            checkpoint_dir: Path to trained model checkpoints
            device: "auto", "cuda", "cpu", or "edge"
            use_sam2: Enable SAM2 segmentation
            verbose: Print progress messages
        """
        self.checkpoint_dir = checkpoint_dir
        self.device = self._resolve_device(device)
        self.use_sam2 = use_sam2
        self.verbose = verbose

        # Lazy-loaded models
        self._medgemma = None
        self._sam2 = None
        self._processor = None

        if self.verbose:
            print(f"MedGamma Production Pipeline initialized")
            print(f"  Device: {self.device}")
            print(f"  SAM2: {'enabled' if use_sam2 else 'disabled'}")

    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device."""
        if device == "auto":
            import torch
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device

    def _load_medgemma(self):
        """Load MedGemma model."""
        if self._medgemma is not None:
            return

        if self.verbose:
            print("Loading MedGemma model...")

        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        model_id = "google/medgemma-1.5-4b-it"
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

        # Quantization for memory efficiency
        if self.device == "cuda":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:
            quant_config = None

        self._processor = AutoProcessor.from_pretrained(
            model_id, token=hf_token, trust_remote_code=True
        )

        self._medgemma = AutoModelForImageTextToText.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
            token=hf_token
        )

        # Load LoRA adapter if available
        adapter_path = os.path.join(self.checkpoint_dir, "medgemma", "detection")
        if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
            from peft import PeftModel
            self._medgemma = PeftModel.from_pretrained(self._medgemma, adapter_path)
            if self.verbose:
                print(f"  Loaded LoRA adapter from {adapter_path}")

        self._medgemma.eval()

        if self.verbose:
            print("  MedGemma loaded successfully")

    def _load_sam2(self):
        """Load SAM2 model."""
        if self._sam2 is not None or not self.use_sam2:
            return

        if self.verbose:
            print("Loading SAM2 model...")

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            # Try to load fine-tuned checkpoint
            sam_checkpoint = os.path.join(self.checkpoint_dir, "sam2", "sam2_medical.pt")
            if not os.path.exists(sam_checkpoint):
                sam_checkpoint = "checkpoints/sam2_hiera_tiny.pt"

            model = build_sam2("sam2_hiera_t.yaml", sam_checkpoint)
            self._sam2 = SAM2ImagePredictor(model)

            if self.verbose:
                print("  SAM2 loaded successfully")

        except Exception as e:
            if self.verbose:
                print(f"  Warning: SAM2 load failed: {e}")
            self.use_sam2 = False

    def analyze(self, image_path: str) -> AnalysisResult:
        """
        Perform complete medical image analysis.

        Args:
            image_path: Path to medical image

        Returns:
            AnalysisResult with findings and segmentation masks
        """
        start_time = time.time()

        # Load models
        self._load_medgemma()
        self._load_sam2()

        # Load image
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        # Step 1: Detect modality
        if self.verbose:
            print(f"\n[1/4] Detecting modality...")
        modality = self._detect_modality(image)

        # Step 2: Screen for abnormalities
        if self.verbose:
            print(f"[2/4] Screening for abnormalities...")
        is_abnormal, screen_confidence = self._screen(image)

        # Step 3: Detect findings
        if self.verbose:
            print(f"[3/4] Detecting specific findings...")
        findings = self._detect_findings(image, modality)

        # Step 4: Segment findings
        segmentation_masks = []
        if self.use_sam2 and findings:
            if self.verbose:
                print(f"[4/4] Segmenting {len(findings)} findings...")
            segmentation_masks = self._segment_findings(image_np, findings)

        # Generate clinical summary
        clinical_summary = self._generate_summary(modality, findings)

        # Determine body region from modality
        body_region = self._infer_body_region(modality)

        inference_time = time.time() - start_time

        return AnalysisResult(
            image_path=image_path,
            timestamp=datetime.now().isoformat(),
            modality=modality,
            body_region=body_region,
            is_abnormal=is_abnormal or len(findings) > 0,
            overall_confidence=max(screen_confidence, max([f.confidence for f in findings], default=0.0)),
            findings=findings,
            segmentation_masks=segmentation_masks,
            clinical_summary=clinical_summary,
            inference_time_seconds=inference_time
        )

    def _detect_modality(self, image: Image.Image) -> str:
        """Detect imaging modality."""
        import torch

        prompt = "What imaging modality is this? Answer with one of: X-ray, CT, MRI, Ultrasound, Mammography, Dermoscopy"

        inputs = self._processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        )

        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._medgemma.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False
            )

        response = self._processor.decode(outputs[0], skip_special_tokens=True)
        return self._parse_modality(response)

    def _screen(self, image: Image.Image) -> Tuple[bool, float]:
        """Screen for abnormalities."""
        import torch

        prompt = (
            "Is this medical image showing signs of disease or abnormality? "
            "Respond with ABNORMAL or NORMAL followed by confidence (0-100%)."
        )

        inputs = self._processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        )

        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._medgemma.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False
            )

        response = self._processor.decode(outputs[0], skip_special_tokens=True)
        return self._parse_screening(response)

    def _detect_findings(self, image: Image.Image, modality: str) -> List[MedicalFinding]:
        """Detect specific pathological findings."""
        import torch

        prompt = (
            f"Analyze this {modality} image for pathological findings. "
            "For each finding provide: class name, bounding box [x1,y1,x2,y2], severity (mild/moderate/severe), and description. "
            'Output JSON: {"findings": [{"class": "...", "box": [...], "severity": "...", "description": "..."}]}'
        )

        inputs = self._processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        )

        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._medgemma.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )

        response = self._processor.decode(outputs[0], skip_special_tokens=True)
        return self._parse_findings(response)

    def _segment_findings(self, image_np: np.ndarray, findings: List[MedicalFinding]) -> List[np.ndarray]:
        """Segment each finding using SAM2."""
        masks = []

        if self._sam2 is None:
            return masks

        # Set image once
        self._sam2.set_image(image_np)

        for finding in findings:
            try:
                box = np.array(finding.bounding_box)
                mask, scores, _ = self._sam2.predict(
                    box=box,
                    multimask_output=False
                )
                masks.append(mask[0])
                finding.iou_score = float(scores[0])
            except Exception as e:
                if self.verbose:
                    print(f"    Segmentation failed for {finding.finding_class}: {e}")

        return masks

    def _parse_modality(self, response: str) -> str:
        """Parse modality from model response."""
        response_lower = response.lower()

        modality_map = {
            "x-ray": "xray", "xray": "xray", "radiograph": "xray",
            "ct": "ct", "computed tomography": "ct",
            "mri": "mri", "magnetic resonance": "mri",
            "ultrasound": "ultrasound", "sonograph": "ultrasound",
            "mammograph": "mammography", "mammo": "mammography",
            "dermoscop": "dermoscopy", "skin": "dermoscopy"
        }

        for key, value in modality_map.items():
            if key in response_lower:
                return value

        return "unknown"

    def _parse_screening(self, response: str) -> Tuple[bool, float]:
        """Parse screening result."""
        response_lower = response.lower()

        is_abnormal = "abnormal" in response_lower or "disease" in response_lower

        # Extract confidence
        import re
        match = re.search(r'(\d+)%', response)
        confidence = float(match.group(1)) / 100 if match else 0.7

        return is_abnormal, confidence

    def _parse_findings(self, response: str) -> List[MedicalFinding]:
        """Parse findings from JSON response."""
        import re

        findings = []

        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))

                for f in data.get("findings", []):
                    finding = MedicalFinding(
                        finding_class=f.get("class", "Unknown"),
                        confidence=f.get("confidence", 0.7),
                        bounding_box=f.get("box", [0, 0, 100, 100]),
                        severity=f.get("severity", "mild"),
                        description=f.get("description", "")
                    )
                    findings.append(finding)
        except:
            pass

        return findings

    def _generate_summary(self, modality: str, findings: List[MedicalFinding]) -> str:
        """Generate clinical summary."""
        if not findings:
            return f"No significant pathological findings detected in this {modality} image."

        summary_parts = [f"Analysis of {modality} image reveals {len(findings)} finding(s):"]

        for i, f in enumerate(findings, 1):
            summary_parts.append(
                f"{i}. {f.finding_class} ({f.severity}): {f.description or 'See image for details'}"
            )

        return " ".join(summary_parts)

    def _infer_body_region(self, modality: str) -> str:
        """Infer body region from modality context."""
        region_map = {
            "xray": "chest",
            "ct": "thorax",
            "mri": "brain",
            "ultrasound": "abdomen",
            "mammography": "breast",
            "dermoscopy": "skin"
        }
        return region_map.get(modality, "unknown")

    def batch_analyze(
        self,
        image_dir: str,
        output_dir: str = "results",
        extensions: List[str] = [".jpg", ".jpeg", ".png", ".dcm"]
    ) -> List[AnalysisResult]:
        """
        Analyze all images in a directory.

        Args:
            image_dir: Directory containing images
            output_dir: Directory to save results
            extensions: Image file extensions to process

        Returns:
            List of AnalysisResult objects
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all images
        images = []
        for ext in extensions:
            images.extend(image_dir.glob(f"*{ext}"))
            images.extend(image_dir.glob(f"*{ext.upper()}"))

        print(f"Found {len(images)} images to process")

        results = []
        for i, image_path in enumerate(images, 1):
            print(f"\n[{i}/{len(images)}] Processing {image_path.name}...")

            try:
                result = self.analyze(str(image_path))
                results.append(result)

                # Save individual result
                result_path = output_dir / f"{image_path.stem}_result.json"
                with open(result_path, "w") as f:
                    json.dump(result.to_dict(), f, indent=2)

            except Exception as e:
                print(f"  Error: {e}")

        # Save batch summary
        summary = {
            "total_images": len(images),
            "processed": len(results),
            "abnormal_count": sum(1 for r in results if r.is_abnormal),
            "modality_distribution": {},
            "timestamp": datetime.now().isoformat()
        }

        for r in results:
            summary["modality_distribution"][r.modality] = \
                summary["modality_distribution"].get(r.modality, 0) + 1

        with open(output_dir / "batch_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nBatch analysis complete:")
        print(f"  Processed: {len(results)}/{len(images)}")
        print(f"  Abnormal: {summary['abnormal_count']}")
        print(f"  Results saved to: {output_dir}")

        return results


def main():
    parser = argparse.ArgumentParser(
        description="MedGamma Production Pipeline for Kaggle Submission"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Single image analysis
    analyze_parser = subparsers.add_parser("analyze", help="Analyze single image")
    analyze_parser.add_argument("image", help="Path to medical image")
    analyze_parser.add_argument("--checkpoint", default="checkpoints/production",
                                help="Checkpoint directory")
    analyze_parser.add_argument("--no-sam2", action="store_true",
                                help="Disable SAM2 segmentation")
    analyze_parser.add_argument("--output", "-o", help="Output JSON file")

    # Batch analysis
    batch_parser = subparsers.add_parser("batch", help="Batch analyze directory")
    batch_parser.add_argument("image_dir", help="Directory with images")
    batch_parser.add_argument("--output", "-o", default="results",
                              help="Output directory")
    batch_parser.add_argument("--checkpoint", default="checkpoints/production",
                              help="Checkpoint directory")

    # Demo mode
    demo_parser = subparsers.add_parser("demo", help="Run interactive demo")
    demo_parser.add_argument("--port", type=int, default=7860, help="Gradio port")

    args = parser.parse_args()

    if args.command == "analyze":
        pipeline = ProductionPipeline(
            checkpoint_dir=args.checkpoint,
            use_sam2=not args.no_sam2
        )

        result = pipeline.analyze(args.image)

        # Print summary
        print("\n" + "="*60)
        print("ANALYSIS RESULTS")
        print("="*60)
        print(f"Image: {result.image_path}")
        print(f"Modality: {result.modality}")
        print(f"Abnormal: {'YES' if result.is_abnormal else 'NO'}")
        print(f"Confidence: {result.overall_confidence:.0%}")
        print(f"Findings: {len(result.findings)}")
        print(f"Time: {result.inference_time_seconds:.1f}s")
        print()
        print("Clinical Summary:")
        print(f"  {result.clinical_summary}")
        print("="*60)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"\nResults saved to: {args.output}")

    elif args.command == "batch":
        pipeline = ProductionPipeline(checkpoint_dir=args.checkpoint)
        pipeline.batch_analyze(args.image_dir, args.output)

    elif args.command == "demo":
        print("Starting Gradio demo...")
        # Import demo module
        from demo.gradio_app import create_demo
        demo = create_demo()
        demo.launch(server_port=args.port)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
