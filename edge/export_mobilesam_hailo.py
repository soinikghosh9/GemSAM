"""
Export MobileSAM encoder for Hailo-10H.

MobileSAM uses a lightweight TinyViT encoder that is more Hailo-friendly
than SAM2's Hiera backbone.

Usage:
    pip install mobile-sam
    python export_mobilesam_hailo.py --export-onnx
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np

def export_mobilesam_encoder(
    output_path: str = "edge/mobilesam_encoder.onnx",
    image_size: int = 1024
):
    """Export MobileSAM encoder to ONNX for Hailo."""
    print("=" * 60)
    print("Exporting MobileSAM Encoder for Hailo-10H")
    print("=" * 60)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    try:
        from mobile_sam import sam_model_registry

        # Load MobileSAM
        model_type = "vit_t"  # TinyViT
        checkpoint = "mobile_sam.pt"

        if not os.path.exists(checkpoint):
            print(f"Downloading MobileSAM checkpoint...")
            import urllib.request
            url = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
            urllib.request.urlretrieve(url, checkpoint)

        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.eval()

        encoder = sam.image_encoder
        print(f"MobileSAM encoder params: {sum(p.numel() for p in encoder.parameters()):,}")

    except ImportError:
        print("MobileSAM not installed. Creating simplified CNN encoder for demo.")
        print("Install with: pip install git+https://github.com/ChaoningZhang/MobileSAM.git")

        # Create a simple CNN encoder that Hailo CAN compile
        encoder = create_hailo_friendly_encoder()

    # Create dummy input
    dummy_input = torch.randn(1, 3, image_size, image_size)

    print(f"Input shape: {dummy_input.shape}")

    # Test forward
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
        opset_version=11,  # Hailo prefers opset 11
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )

    print(f"Saved to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")

    # Verify
    try:
        import onnx
        model = onnx.load(output_path)
        onnx.checker.check_model(model)
        print("ONNX validation: PASSED")
    except Exception as e:
        print(f"ONNX validation error: {e}")

    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print(f"""
# In WSL with Hailo DFC:
hailo parser onnx {output_path} \\
    --hw-arch hailo10h \\
    --har-path mobilesam_encoder.har

hailo compiler mobilesam_encoder.har \\
    --hw-arch hailo10h \\
    -o mobilesam_encoder_hailo10h.hef
""")


def create_hailo_friendly_encoder():
    """
    Create a simple CNN encoder that Hailo CAN definitely compile.

    This uses only operations that Hailo supports:
    - Conv2d
    - BatchNorm2d
    - ReLU
    - MaxPool2d
    - Global Average Pooling

    No attention, no dynamic reshapes, no complex tensor operations.
    """

    class HailoFriendlyEncoder(nn.Module):
        """
        Simple CNN encoder for medical image feature extraction.
        Designed to be 100% Hailo-compatible.

        Architecture similar to a small ResNet but without residual connections
        for maximum Hailo compatibility.
        """
        def __init__(self, in_channels=3, base_channels=64, output_dim=256):
            super().__init__()

            # Stage 1: 1024 -> 256
            self.stage1 = nn.Sequential(
                nn.Conv2d(in_channels, base_channels, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # 256
            )

            # Stage 2: 256 -> 128
            self.stage2 = nn.Sequential(
                nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(base_channels * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1, bias=False),
                nn.BatchNorm2d(base_channels * 2),
                nn.ReLU(inplace=True),
            )

            # Stage 3: 128 -> 64
            self.stage3 = nn.Sequential(
                nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(base_channels * 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1, bias=False),
                nn.BatchNorm2d(base_channels * 4),
                nn.ReLU(inplace=True),
            )

            # Stage 4: 64 -> 64 (output feature map)
            self.stage4 = nn.Sequential(
                nn.Conv2d(base_channels * 4, output_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(output_dim),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            x = self.stage1(x)  # [B, 64, 256, 256]
            x = self.stage2(x)  # [B, 128, 128, 128]
            x = self.stage3(x)  # [B, 256, 64, 64]
            x = self.stage4(x)  # [B, 256, 64, 64]
            return x

    encoder = HailoFriendlyEncoder()
    print(f"Created Hailo-friendly CNN encoder: {sum(p.numel() for p in encoder.parameters()):,} params")
    return encoder


def export_hailo_friendly_encoder(
    output_path: str = "edge/hailo_friendly_encoder.onnx",
    image_size: int = 1024
):
    """Export the simple Hailo-friendly encoder."""
    print("=" * 60)
    print("Exporting Hailo-Friendly CNN Encoder")
    print("=" * 60)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    encoder = create_hailo_friendly_encoder()
    encoder.eval()

    dummy_input = torch.randn(1, 3, image_size, image_size)

    with torch.no_grad():
        output = encoder(dummy_input)
    print(f"Input: {dummy_input.shape} -> Output: {output.shape}")

    # Export
    torch.onnx.export(
        encoder,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )

    print(f"Saved to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    # Verify
    try:
        import onnx
        import onnxruntime as ort

        model = onnx.load(output_path)
        onnx.checker.check_model(model)
        print("ONNX check: PASSED")

        # Test inference
        session = ort.InferenceSession(output_path)
        result = session.run(None, {"input": dummy_input.numpy()})
        print(f"ONNX Runtime test: PASSED (output shape: {result[0].shape})")

    except Exception as e:
        print(f"Verification error: {e}")

    print("\n" + "=" * 60)
    print("HAILO COMPILATION COMMANDS:")
    print("=" * 60)
    print(f"""
# This encoder uses ONLY Hailo-supported operations!
# Run in WSL with Hailo DFC:

hailo parser onnx {output_path} \\
    --hw-arch hailo10h \\
    --har-path hailo_encoder.har

hailo compiler hailo_encoder.har \\
    --hw-arch hailo10h \\
    -o hailo_encoder_hailo10h.hef

# Then copy to Raspberry Pi:
scp hailo_encoder_hailo10h.hef pi@raspberrypi:~/GemSAM/edge/models/
""")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export Hailo-compatible encoder")
    parser.add_argument("--export-onnx", action="store_true", help="Export MobileSAM")
    parser.add_argument("--export-simple", action="store_true", help="Export simple CNN (guaranteed Hailo compatible)")
    parser.add_argument("--output", default="edge/encoder.onnx", help="Output path")
    parser.add_argument("--image-size", type=int, default=1024, help="Input size")

    args = parser.parse_args()

    if args.export_onnx:
        export_mobilesam_encoder(args.output, args.image_size)
    elif args.export_simple:
        export_hailo_friendly_encoder(args.output, args.image_size)
    else:
        print("Use --export-simple for guaranteed Hailo compatibility")
        print("Use --export-onnx to try MobileSAM (may have compatibility issues)")
        parser.print_help()


if __name__ == "__main__":
    main()
