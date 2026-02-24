"""
MedGamma Edge Inference for Raspberry Pi 5 + Hailo AI HAT+2.

Optimized for:
- Raspberry Pi 5 (8GB RAM)
- Hailo-10H NPU (40 TOPS INT8)
- Privacy-preserving local inference

Architecture:
- MedGemma 4B: Runs on CPU with INT4 quantization (~30-60s per image)
- SAM2-Tiny: Runs on Hailo-10H NPU (~25-50ms per mask)

This module is designed for the Kaggle MedGemma Impact Challenge
Edge AI Prize ($5,000).

Usage on Raspberry Pi:
    python edge_inference.py --image chest_xray.jpg --task full
"""

import os
import sys
import time
import json
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import numpy as np
import gc
from PIL import Image

# Conditional imports for edge vs development
RUNNING_ON_EDGE = os.path.exists("/proc/device-tree/model")

if RUNNING_ON_EDGE:
    try:
        from hailo_platform import (
            VDevice, HailoStreamInterface, ConfigureParams,
            InferVStreams, InputVStreamParams, OutputVStreamParams,
            HEF, HailoSchedulingAlgorithm
        )
        HAILO_AVAILABLE = True
    except ImportError:
        HAILO_AVAILABLE = False
        print("Warning: Hailo SDK not found. Running in CPU-only mode.")
else:
    HAILO_AVAILABLE = False


@dataclass
class EdgeInferenceResult:
    """Result from edge inference."""
    image_path: str
    modality: str
    is_abnormal: bool
    confidence: float
    findings: List[Dict]
    segmentation_masks: List[np.ndarray]
    inference_time_ms: float
    device_info: Dict


class HailoSAM2Runner:
    """
    SAM2 inference on Hailo-10H NPU.

    The Hailo-10H provides 40 TOPS INT8 performance,
    enabling real-time segmentation (~25-50ms per mask).
    """

    def __init__(self, hef_path: str = "models/sam2_encoder_hailo10h.hef"):
        """
        Initialize Hailo SAM2 runner.

        Args:
            hef_path: Path to compiled Hailo Executable Format file
        """
        self.hef_path = hef_path
        self.device = None
        self.network_group = None
        self.input_vstreams = None
        self.output_vstreams = None

        if HAILO_AVAILABLE and os.path.exists(hef_path):
            self._initialize_hailo()
        else:
            print(f"Hailo HEF not found at {hef_path}. Using CPU fallback.")

    def _initialize_hailo(self):
        """Initialize Hailo device and load model."""
        try:
            # Create virtual device with scheduler
            params = VDevice.create_params()
            params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
            self.device = VDevice(params)

            # Create infer model
            self.infer_model = self.device.create_infer_model(self.hef_path)
            
            # Configure model
            self.configured_model = self.infer_model.configure()
            
            print(f"Hailo-10H initialized with {self.hef_path}")
            print(f"  Model: {self.hef_path}")

        except Exception as e:
            print(f"Failed to initialize Hailo: {e}")
            self.device = None

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """
        Run SAM2 image encoder on Hailo-10H.

        Args:
            image: Preprocessed image [1, 3, 1024, 1024]

        Returns:
            Image embeddings for mask decoder
        """
        if self.device is None:
            return self._cpu_fallback_encode(image)

        start = time.time()

        # Run inference using InferModel API
        with self.configured_model as model:
            bindings = model.create_bindings()
            bindings.input().set_buffer(image.astype(np.float32))
            
            # Assuming single output named 'output' or first output
            output_name = self.infer_model.output_names[0]
            output_buffer = np.empty(self.infer_model.output(output_name).shape, dtype=np.float32)
            bindings.output(output_name).set_buffer(output_buffer)
            
            job = model.run(bindings)
            job.wait(10000)
            
        elapsed_ms = (time.time() - start) * 1000
        print(f"  Hailo encoder: {elapsed_ms:.1f}ms")

        return output_buffer

    def _cpu_fallback_encode(self, image: np.ndarray) -> np.ndarray:
        """CPU fallback when Hailo is unavailable."""
        print("  Using CPU fallback for SAM2 encoder...")
        # This would use PyTorch SAM2 - slower but works
        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            model = build_sam2("sam2_hiera_t.yaml", "checkpoints/sam2_hiera_tiny.pt")
            predictor = SAM2ImagePredictor(model)

            # Process image
            predictor.set_image(image)
            return predictor.get_image_embedding()

        except Exception as e:
            print(f"  CPU fallback failed: {e}")
            return np.zeros((1, 256, 64, 64), dtype=np.float32)

    def segment(
        self,
        image: np.ndarray,
        box: List[float]
    ) -> Tuple[np.ndarray, float]:
        """
        Segment region of interest.

        Args:
            image: Original image
            box: [x1, y1, x2, y2] bounding box

        Returns:
            mask: Binary segmentation mask
            score: IoU prediction score
        """
        # Get image embeddings from Hailo
        embeddings = self.encode_image(image)

        # Run mask decoder on CPU (lightweight operation)
        mask, score = self._decode_mask(embeddings, box)

        return mask, score

    def _decode_mask(
        self,
        embeddings: np.ndarray,
        box: List[float]
    ) -> Tuple[np.ndarray, float]:
        """Decode mask from embeddings (runs on CPU)."""
        # Mask decoder is lightweight, CPU is fine
        # This is a placeholder - real implementation would use SAM2 decoder
        H, W = 1024, 1024
        x1, y1, x2, y2 = [int(c) for c in box]

        # Create simple rectangular mask as placeholder
        mask = np.zeros((H, W), dtype=np.float32)
        mask[y1:y2, x1:x2] = 1.0

        return mask, 0.85


class EdgeMedGemmaRunner:
    """
    MedGemma inference on Raspberry Pi 5 CPU.

    Uses FP16 precision for memory efficiency on ARM CPU.
    NOTE: bitsandbytes INT4 quantization requires CUDA and CANNOT work
    on ARM CPU — it will silently hang. We use torch.float16 instead.

    Expected: ~4 GB RAM, ~60-120s per inference on RPi5.
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints/production",
    ):
        """
        Initialize MedGemma for edge inference.

        Args:
            checkpoint_dir: Path to trained checkpoints
        """
        self.checkpoint_dir = checkpoint_dir
        self.model = None
        self.processor = None
        self._loaded = False

    @staticmethod
    def _check_system_memory():
        """Print system memory diagnostics. Warn if likely to OOM."""
        try:
            with open("/proc/meminfo", "r") as f:
                meminfo = {}
                for line in f:
                    parts = line.split()
                    meminfo[parts[0].rstrip(":")] = int(parts[1])  # kB

            total_gb = meminfo.get("MemTotal", 0) / 1024 / 1024
            avail_gb = meminfo.get("MemAvailable", 0) / 1024 / 1024
            swap_gb  = meminfo.get("SwapTotal", 0) / 1024 / 1024

            print(f"  System RAM : {total_gb:.1f} GB total, {avail_gb:.1f} GB available")
            print(f"  Swap       : {swap_gb:.1f} GB")

            if avail_gb < 4.0:
                print("  ⚠  WARNING: Less than 4 GB RAM available!")
                print("     Close browsers/apps to free memory.")
            if swap_gb < 2.0:
                print("  ⚠  WARNING: Swap is < 2 GB. Strongly recommend >= 4 GB swap.")
                print("     Fix: sudo dphys-swapfile swapoff")
                print("           sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=4096/' /etc/dphys-swapfile")
                print("           sudo dphys-swapfile setup && sudo dphys-swapfile swapon")
        except FileNotFoundError:
            print("  (Not running on Linux — memory check skipped)")

    def _load_model(self):
        """Load model in FP16 on CPU (ARM-compatible, no CUDA needed)."""
        if self._loaded:
            return

        print("Loading MedGemma (FP16) on CPU...")
        self._check_system_memory()
        start = time.time()

        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            model_id = "/media/ai-pi/One Touch1/huggingface_cache/hub/models--google--medgemma-1.5-4b-it/snapshots/e9792da5fb8ee651083d345ec4bce07c3c9f1641"

            # ── Processor (lightweight) ──
            self.processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=True
            )

            # ── Model (FP16, ~4 GB) ──
            # IMPORTANT: Do NOT use BitsAndBytesConfig here.
            # bitsandbytes requires CUDA and will hang/crash on ARM CPU.
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=torch.float16,   # Half-precision: ~4 GB vs ~8 GB FP32
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True,       # Load shard-by-shard, don't double memory
            )

            # ── LoRA adapter (tiny, ~30 MB) ──
            adapter_paths = [
                os.path.join(self.checkpoint_dir, "medgemma", "detection_best"),
                os.path.join(self.checkpoint_dir, "medgemma", "detection"),
                os.path.join(self.checkpoint_dir, "medgemma", "final"),
            ]

            adapter_path = None
            for path in adapter_paths:
                if os.path.exists(os.path.join(path, "adapter_config.json")):
                    adapter_path = path
                    break

            if adapter_path:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, adapter_path)
                print(f"  Loaded LoRA adapter from {adapter_path}")

            self.model.eval()
            self._loaded = True

            # Free any leftover allocation buffers
            gc.collect()

            elapsed = time.time() - start
            print(f"  ✓ Model loaded in {elapsed:.1f}s")
            self._check_system_memory()  # Show post-load memory

        except Exception as e:
            print(f"\n✗ Failed to load MedGemma: {e}")
            print("  Possible causes:")
            print("    1. Not enough RAM — ensure swap >= 4 GB")
            print("    2. Model files missing — check HF_HOME path")
            print("    3. Corrupted download — delete cache and re-download")
            raise

    def unload_model(self):
        """Explicitly free model memory (call before SAM2 if RAM is tight)."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self._loaded = False
        gc.collect()
        print("  MedGemma unloaded — RAM freed for next stage.")

    def analyze(
        self,
        image_path: str,
        task: str = "detection"
    ) -> Dict:
        """
        Analyze medical image.

        Args:
            image_path: Path to image
            task: "screening", "detection", or "modality"

        Returns:
            Analysis results
        """
        self._load_model()

        print(f"Analyzing image: {os.path.basename(image_path)}")
        start = time.time()

        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")

        # Build prompt based on task
        if task == "screening":
            task_prompt = (
                "Is this medical image showing signs of disease? "
                "Respond with 'HEALTHY' or 'ABNORMAL' with reasoning."
            )
        elif task == "detection":
            task_prompt = (
                "Analyze this medical image for pathological findings. "
                "For each finding, provide the class name and bounding box. "
                'Output JSON: {"findings": [{"class": "...", "box": [x1,y1,x2,y2]}]}'
            )
        else:
            task_prompt = "What imaging modality is this? (X-ray, CT, MRI, Ultrasound)"

        # Use chat template for correct formatting (instruct turns + image tokens)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": task_prompt}
                ]
            }
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        # Note: apply_chat_template might add a BOS token, so we set add_special_tokens=False in processor call if needed
        # but processor() usually handles it.

        # Process inputs — use FP16 to match model dtype
        import torch
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        )
        # Cast pixel_values to float16 to match model weights
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

        # Token budget: detection JSON is compact; screening/modality even shorter
        max_tokens = 150 if task == "detection" else 50

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )

        response = self.processor.decode(outputs[0], skip_special_tokens=True)

        # Aggressively free tensors
        del inputs, outputs
        gc.collect()

        elapsed = time.time() - start
        print(f"  Inference time: {elapsed:.1f}s")

        return {
            "response": response,
            "task": task,
            "inference_time_s": elapsed
        }


class MedGammaEdgePipeline:
    """
    Complete MedGamma pipeline for Raspberry Pi 5 + Hailo-10H.

    Combines:
    - MedGemma (CPU): Reasoning and detection
    - SAM2 (Hailo NPU): Fast segmentation

    This is the main entry point for edge inference.
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints/production",
        hailo_hef: str = "models/sam2_encoder_hailo10h.hef"
    ):
        """
        Initialize edge pipeline.

        Args:
            checkpoint_dir: Path to trained model checkpoints
            hailo_hef: Path to Hailo HEF file for SAM2
        """
        # Initialize lazily — don't allocate both at once on 8 GB system
        self._checkpoint_dir = checkpoint_dir
        self._hailo_hef = hailo_hef
        self.medgemma = None
        self.sam2 = None

        # Device info
        self.device_info = self._get_device_info()

    def _get_device_info(self) -> Dict:
        """Get device information."""
        info = {
            "platform": "Raspberry Pi 5" if RUNNING_ON_EDGE else "Development",
            "hailo_available": HAILO_AVAILABLE,
            "hailo_model": "Hailo-10H (40 TOPS)" if HAILO_AVAILABLE else "N/A"
        }

        if RUNNING_ON_EDGE:
            try:
                with open("/proc/device-tree/model", "r") as f:
                    info["device_model"] = f.read().strip()
            except:
                pass

            try:
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if "MemTotal" in line:
                            mem_kb = int(line.split()[1])
                            info["memory_gb"] = round(mem_kb / 1024 / 1024, 1)
                            break
            except:
                pass

        return info

    def analyze(
        self,
        image_path: str,
        task: str = "full"
    ) -> EdgeInferenceResult:
        """
        Run complete analysis pipeline.

        Args:
            image_path: Path to medical image
            task: "full", "screen", "detect", or "segment"

        Returns:
            EdgeInferenceResult with findings and masks
        """
        start = time.time()

        result = EdgeInferenceResult(
            image_path=image_path,
            modality="unknown",
            is_abnormal=False,
            confidence=0.0,
            findings=[],
            segmentation_masks=[],
            inference_time_ms=0.0,
            device_info=self.device_info
        )

        try:
            # ── Step 1: Load MedGemma and run detection ──
            print("\n[1/2] Loading MedGemma and analyzing image...")
            self.medgemma = EdgeMedGemmaRunner(self._checkpoint_dir)

            # Single combined pass: detect modality AND findings in one prompt
            detection_result = self.medgemma.analyze(image_path, task="detection")
            findings = self._parse_findings(detection_result["response"])
            result.modality = self._guess_modality_from_prompt(detection_result["response"])
            result.findings = findings
            result.is_abnormal = len(findings) > 0
            result.confidence = max(
                [f.get("confidence", 0.7) for f in findings], default=0.0
            )

            # FREE MedGemma before loading SAM2 — cannot hold both in 8 GB
            print("  Unloading MedGemma to free RAM for SAM2...")
            self.medgemma.unload_model()
            self.medgemma = None
            gc.collect()

            # ── Step 2: Segmentation (if findings detected) ──
            if findings and task in ["full", "segment"]:
                print("\n[2/2] Segmenting regions (Hailo-10H)...")
                self.sam2 = HailoSAM2Runner(self._hailo_hef)
                image = np.array(Image.open(image_path).convert("RGB"))

                for finding in findings:
                    box = finding.get("box", [])
                    if box and len(box) == 4:
                        mask, score = self.sam2.segment(image, box)
                        result.segmentation_masks.append(mask)
                        finding["iou_score"] = score

                # Free SAM2 as well
                self.sam2 = None
                gc.collect()
            else:
                print("\n[2/2] No findings — skipping segmentation.")

        except Exception as e:
            print(f"\n✗ Error during analysis: {e}")
            import traceback
            traceback.print_exc()

        result.inference_time_ms = (time.time() - start) * 1000
        return result

    def _guess_modality_from_prompt(self, response: str) -> str:
        """Best-effort modality detection from the detection response text."""
        r = response.lower()
        if "x-ray" in r or "xray" in r or "chest" in r or "cxr" in r or "radiograph" in r:
            return "xray"
        elif "ct" in r or "computed tomography" in r:
            return "ct"
        elif "mri" in r or "magnetic resonance" in r:
            return "mri"
        elif "ultrasound" in r or "sonograph" in r:
            return "ultrasound"
        return "unknown"

    def _parse_findings(self, response: str) -> List[Dict]:
        """Parse findings from JSON response."""
        import re

        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                if "findings" in data:
                    return data["findings"]
        except:
            pass

        return []

    def print_results(self, result: EdgeInferenceResult):
        """Print formatted results."""
        print("\n" + "="*60)
        print("MedGamma Edge Analysis Results")
        print("="*60)
        print(f"Image: {result.image_path}")
        print(f"Modality: {result.modality}")
        print(f"Abnormal: {'YES' if result.is_abnormal else 'NO'}")
        print(f"Confidence: {result.confidence:.0%}")
        print(f"Findings: {len(result.findings)}")
        print(f"Segments: {len(result.segmentation_masks)}")
        print(f"Inference Time: {result.inference_time_ms:.0f}ms")
        print()
        print("Device Info:")
        for k, v in result.device_info.items():
            print(f"  {k}: {v}")
        print("="*60)

        if result.findings:
            print("\nFindings:")
            for i, f in enumerate(result.findings, 1):
                cls = f.get("class", "Unknown")
                box = f.get("box", [])
                iou = f.get("iou_score", 0)
                print(f"  {i}. {cls}")
                print(f"     Box: {box}")
                print(f"     IoU: {iou:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="MedGamma Edge Inference (Raspberry Pi 5 + Hailo-10H)"
    )
    parser.add_argument("--image", "-i", required=True, help="Path to medical image")
    parser.add_argument("--task", "-t", choices=["full", "screen", "detect", "segment"],
                        default="full", help="Analysis task")
    parser.add_argument("--checkpoint", "-c", 
                        default="checkpoints/production/medgemma/detection_best",
                        help="Checkpoint directory")
    parser.add_argument("--hailo-hef", default="edge/models/hailo_encoder.hef",
                        help="Path to Hailo HEF file")

    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════╗
║          MedGamma Edge Inference                             ║
║          Raspberry Pi 5 + Hailo-10H (40 TOPS)                ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # Initialize pipeline
    pipeline = MedGammaEdgePipeline(
        checkpoint_dir=args.checkpoint,
        hailo_hef=args.hailo_hef
    )

    # Run analysis
    result = pipeline.analyze(args.image, task=args.task)

    # Print results
    pipeline.print_results(result)


if __name__ == "__main__":
    main()
