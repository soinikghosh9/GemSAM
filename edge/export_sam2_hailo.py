"""
Export SAM2 encoder to ONNX and compile for Hailo-10H.

This script prepares SAM2-Tiny for deployment on Hailo AI HAT+2.

Steps:
1. Export SAM2 encoder to ONNX
2. Compile ONNX to Hailo HEF format (on Hailo Dataflow Compiler)

Requirements on Windows (export ONNX):
    pip install torch onnx onnxruntime

Requirements on Hailo Dataflow Compiler (compile HEF):
    Run on Hailo-provided Docker or Linux machine with Hailo SDK

Usage:
    # Step 1: Export ONNX (on Windows)
    python export_sam2_hailo.py --export-onnx

    # Step 2: Compile HEF (on Hailo SDK machine)
    hailo compiler sam2_encoder.onnx --target hailo10h -o sam2_encoder_hailo10h.hef
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np


def export_sam2_encoder_onnx(
    checkpoint_path: str = "checkpoints/sam2/sam2_hiera_tiny.pt",
    output_path: str = "edge/models/sam2_encoder.onnx",
    image_size: int = 1024
):
    """
    Export SAM2 Hiera-Tiny encoder to ONNX.

    The encoder is the heavy part - runs on Hailo-10H (40 TOPS).
    The mask decoder is lightweight - runs on CPU.
    """
    print("="*60)
    print("Exporting SAM2 Encoder to ONNX")
    print("="*60)

    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        from sam2.build_sam import build_sam2
        from sam2.modeling.sam2_base import SAM2Base

        print(f"Loading SAM2 from {checkpoint_path}...")

        # Build SAM2 model
        model = build_sam2(
            config_file="sam2_hiera_t.yaml",
            ckpt_path=checkpoint_path
        )
        model.eval()

        # Extract just the image encoder
        encoder = model.image_encoder

        print(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")

    except ImportError:
        print("SAM2 not installed. Creating a mock encoder for testing.")

        # Mock encoder for testing the export pipeline
        class MockHieraEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, 7, stride=4, padding=3)
                self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
                self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
                self.avg_pool = nn.AdaptiveAvgPool2d((64, 64))

            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = torch.relu(self.conv3(x))
                x = self.avg_pool(x)
                return x

        encoder = MockHieraEncoder()
        print("Using mock encoder (SAM2 not available)")

    # Create dummy input
    dummy_input = torch.randn(1, 3, image_size, image_size)

    print(f"Input shape: {dummy_input.shape}")

    # Test forward pass
    with torch.no_grad():
        output = encoder(dummy_input)
    print(f"Output shape: {output.shape}")

    # Export to ONNX
    print(f"\nExporting to {output_path}...")

    torch.onnx.export(
        encoder,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )

    print(f"ONNX model saved to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")

    # Verify ONNX model
    try:
        import onnx
        import onnxruntime as ort

        print("\nVerifying ONNX model...")
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("  ONNX model check: PASSED")

        # Test inference with ONNX Runtime
        session = ort.InferenceSession(output_path)
        ort_inputs = {"input": dummy_input.numpy()}
        ort_outputs = session.run(None, ort_inputs)
        print(f"  ONNX Runtime inference: PASSED")
        print(f"  Output shape: {ort_outputs[0].shape}")

    except ImportError:
        print("onnx/onnxruntime not installed. Skipping verification.")

    print("\n" + "="*60)
    print("NEXT STEPS for Hailo Compilation:")
    print("="*60)
    print("""
1. Copy the ONNX file to your Hailo compilation environment:
   scp edge/models/sam2_encoder.onnx user@hailo-machine:~/

2. On the Hailo machine, run:
   hailo compiler sam2_encoder.onnx \\
       --target hailo10h \\
       --output-dir ./hailo_output \\
       -o sam2_encoder_hailo10h.hef

3. Copy the HEF file back:
   scp user@hailo-machine:~/hailo_output/sam2_encoder_hailo10h.hef edge/models/

4. Or use Hailo Model Zoo precompiled models if available.
""")

    return output_path


def create_hailo_calibration_data(
    output_dir: str = "edge/models/calibration",
    num_samples: int = 100,
    image_size: int = 1024
):
    """
    Create calibration data for Hailo quantization.

    Hailo compiler needs representative data for INT8 quantization.
    """
    print("Creating calibration data for Hailo quantization...")
    os.makedirs(output_dir, exist_ok=True)

    # Try to use real medical images
    try:
        from PIL import Image
        import glob

        medical_images = glob.glob("medical_data/**/*.png", recursive=True)
        medical_images += glob.glob("medical_data/**/*.jpg", recursive=True)

        if medical_images:
            print(f"Found {len(medical_images)} medical images")
            num_samples = min(num_samples, len(medical_images))

            for i, img_path in enumerate(medical_images[:num_samples]):
                img = Image.open(img_path).convert("RGB")
                img = img.resize((image_size, image_size))
                img_array = np.array(img, dtype=np.float32) / 255.0
                img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW

                np.save(os.path.join(output_dir, f"sample_{i:04d}.npy"), img_array)

            print(f"Saved {num_samples} calibration samples")
            return
    except Exception as e:
        print(f"Could not load medical images: {e}")

    # Fall back to random data
    print("Using random calibration data (not optimal)")
    for i in range(num_samples):
        sample = np.random.randn(3, image_size, image_size).astype(np.float32)
        sample = (sample - sample.min()) / (sample.max() - sample.min())
        np.save(os.path.join(output_dir, f"sample_{i:04d}.npy"), sample)

    print(f"Saved {num_samples} random calibration samples")


def main():
    parser = argparse.ArgumentParser(
        description="Export SAM2 for Hailo-10H deployment"
    )
    parser.add_argument("--export-onnx", action="store_true",
                        help="Export SAM2 encoder to ONNX")
    parser.add_argument("--create-calibration", action="store_true",
                        help="Create calibration data for Hailo quantization")
    parser.add_argument("--checkpoint", default="checkpoints/sam2/sam2_hiera_tiny.pt",
                        help="SAM2 checkpoint path")
    parser.add_argument("--output", default="edge/models/sam2_encoder.onnx",
                        help="ONNX output path")
    parser.add_argument("--image-size", type=int, default=1024,
                        help="Input image size")

    args = parser.parse_args()

    if args.export_onnx:
        export_sam2_encoder_onnx(
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            image_size=args.image_size
        )

    if args.create_calibration:
        create_hailo_calibration_data(
            image_size=args.image_size
        )

    if not args.export_onnx and not args.create_calibration:
        parser.print_help()


if __name__ == "__main__":
    main()
