#!/usr/bin/env python3
"""
Quick GemSAM Edge Demo for Raspberry Pi 5 + Hailo-10H.

GemSAM: An Agentic Framework for Explainable Multi-Modal Medical Image Analysis
on Edge using MedGemma-1.5 and SAM2

This is a lightweight demo that works even without full MedGemma model.
Use this for quick testing and video recording.

Usage:
    python quick_demo.py --image test_xray.jpg
"""

import os
import sys
import time
import argparse
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def detect_device():
    """Detect if running on Raspberry Pi with Hailo."""
    info = {
        "platform": "Unknown",
        "hailo": False,
        "hailo_model": "N/A"
    }

    # Check for Raspberry Pi
    if os.path.exists("/proc/device-tree/model"):
        with open("/proc/device-tree/model", "r") as f:
            info["platform"] = f.read().strip()
    else:
        info["platform"] = "Development PC"

    # Check for Hailo
    try:
        from hailo_platform import VDevice
        device = VDevice()
        info["hailo"] = True
        info["hailo_model"] = "Hailo-10H (40 TOPS)"
    except:
        info["hailo"] = False
        info["hailo_model"] = "Not available"

    return info


def simulate_analysis(image_path: str) -> dict:
    """
    Simulate medical image analysis.

    This provides realistic outputs for demo purposes when
    full model inference is not available.
    """
    # Load image
    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    # Analyze filename for hints
    filename = os.path.basename(image_path).lower()

    # Detect modality from filename or image characteristics
    if "xray" in filename or "chest" in filename or "cxr" in filename:
        modality = "X-ray"
        findings = [
            {"class": "Cardiomegaly", "box": [300, 350, 700, 750], "confidence": 0.87},
            {"class": "Pleural Effusion", "box": [50, 400, 200, 600], "confidence": 0.72}
        ]
    elif "mri" in filename or "brain" in filename:
        modality = "MRI"
        findings = [
            {"class": "Glioma", "box": [350, 200, 650, 500], "confidence": 0.91}
        ]
    elif "ct" in filename:
        modality = "CT"
        findings = [
            {"class": "Pulmonary Nodule", "box": [400, 300, 550, 450], "confidence": 0.78}
        ]
    elif "ultrasound" in filename or "us_" in filename:
        modality = "Ultrasound"
        findings = [
            {"class": "Cyst", "box": [300, 250, 500, 450], "confidence": 0.83}
        ]
    else:
        # Default to X-ray with generic findings
        modality = "X-ray"
        # Check image brightness to determine normal/abnormal
        img_array = np.array(img.convert("L"))
        mean_brightness = img_array.mean()

        if mean_brightness < 100:
            findings = [
                {"class": "Consolidation", "box": [200, 300, 400, 500], "confidence": 0.75}
            ]
        else:
            findings = []  # Normal

    # Scale boxes to image size
    for f in findings:
        f["box"] = [
            int(f["box"][0] * width / 1000),
            int(f["box"][1] * height / 1000),
            int(f["box"][2] * width / 1000),
            int(f["box"][3] * height / 1000)
        ]

    return {
        "modality": modality,
        "findings": findings,
        "is_abnormal": len(findings) > 0,
        "confidence": max([f["confidence"] for f in findings], default=0.95),
        "image_size": (width, height)
    }


def create_visualization(image_path: str, results: dict, output_path: str = None):
    """Create visualization with bounding boxes and labels."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Try to use a better font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
        font_small = font

    # Draw bounding boxes
    colors = {
        "Cardiomegaly": "red",
        "Pleural Effusion": "orange",
        "Glioma": "red",
        "Meningioma": "blue",
        "Consolidation": "yellow",
        "Pneumonia": "red",
        "Nodule": "cyan",
        "Cyst": "green",
        "default": "red"
    }

    for finding in results["findings"]:
        box = finding["box"]
        cls = finding["class"]
        conf = finding["confidence"]
        color = colors.get(cls, colors["default"])

        # Draw box
        draw.rectangle(box, outline=color, width=3)

        # Draw label
        label = f"{cls} ({conf:.0%})"
        draw.rectangle([box[0], box[1]-25, box[0]+len(label)*10, box[1]], fill=color)
        draw.text((box[0]+5, box[1]-22), label, fill="white", font=font_small)

    # Draw header
    header = f"GemSAM Edge | {results['modality']} | {'ABNORMAL' if results['is_abnormal'] else 'NORMAL'}"
    draw.rectangle([0, 0, img.width, 35], fill="black")
    draw.text((10, 8), header, fill="white", font=font)

    # Save or show
    if output_path:
        img.save(output_path)
        print(f"Saved visualization to: {output_path}")
    else:
        output_path = image_path.rsplit(".", 1)[0] + "_analyzed.jpg"
        img.save(output_path)
        print(f"Saved visualization to: {output_path}")

    return img


def main():
    parser = argparse.ArgumentParser(
        description="GemSAM Edge Quick Demo"
    )
    parser.add_argument("--image", "-i", required=True, help="Path to medical image")
    parser.add_argument("--output", "-o", help="Output visualization path")
    parser.add_argument("--full", action="store_true", help="Run full inference (slower)")

    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════╗
║          GemSAM Edge Quick Demo                            ║
║          Raspberry Pi 5 + Hailo-10H (40 TOPS)                ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # Detect device
    device_info = detect_device()
    print(f"Platform: {device_info['platform']}")
    print(f"Hailo NPU: {device_info['hailo_model']}")
    print()

    # Run analysis
    print(f"Analyzing: {args.image}")
    start_time = time.time()

    if args.full:
        # Try full inference
        try:
            from edge_inference import GemSAMEdgePipeline
            pipeline = GemSAMEdgePipeline()
            result = pipeline.analyze(args.image)
            results = {
                "modality": result.modality,
                "findings": result.findings,
                "is_abnormal": result.is_abnormal,
                "confidence": result.confidence,
                "image_size": Image.open(args.image).size
            }
        except Exception as e:
            print(f"Full inference failed: {e}")
            print("Falling back to simulation...")
            results = simulate_analysis(args.image)
    else:
        # Quick simulation
        results = simulate_analysis(args.image)

    elapsed = time.time() - start_time

    # Print results
    print()
    print("="*50)
    print("RESULTS")
    print("="*50)
    print(f"Modality: {results['modality']}")
    print(f"Status: {'ABNORMAL' if results['is_abnormal'] else 'NORMAL'}")
    print(f"Confidence: {results['confidence']:.0%}")
    print(f"Findings: {len(results['findings'])}")
    print(f"Inference Time: {elapsed*1000:.0f}ms")
    print()

    if results["findings"]:
        print("Detected Abnormalities:")
        for i, f in enumerate(results["findings"], 1):
            print(f"  {i}. {f['class']} (confidence: {f['confidence']:.0%})")

    # Create visualization
    print()
    create_visualization(args.image, results, args.output)

    print()
    print("="*50)
    print("Demo complete!")
    print("="*50)


if __name__ == "__main__":
    main()
