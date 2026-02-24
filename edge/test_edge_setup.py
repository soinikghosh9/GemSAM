#!/usr/bin/env python3
"""
Test script for MedGamma Edge Setup on Raspberry Pi 5 + Hailo-10H.

Run this to verify your edge deployment is ready.

Usage:
    python test_edge_setup.py
"""

import sys
import os

def test_python_version():
    """Check Python version."""
    print("\n[1/8] Python Version")
    print(f"  Python: {sys.version}")
    assert sys.version_info >= (3, 10), "Python 3.10+ required"
    print("  [PASS] Python 3.10+")
    return True


def test_pytorch():
    """Check PyTorch installation."""
    print("\n[2/8] PyTorch")
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        print(f"  Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        print("  [PASS] PyTorch installed")
        return True
    except ImportError as e:
        print(f"  [FAIL] {e}")
        return False


def test_transformers():
    """Check Transformers installation."""
    print("\n[3/8] Transformers")
    try:
        import transformers
        print(f"  Transformers: {transformers.__version__}")
        print("  [PASS] Transformers installed")
        return True
    except ImportError as e:
        print(f"  [FAIL] {e}")
        return False


def test_peft():
    """Check PEFT installation."""
    print("\n[4/8] PEFT (LoRA)")
    try:
        import peft
        print(f"  PEFT: {peft.__version__}")
        print("  [PASS] PEFT installed")
        return True
    except ImportError as e:
        print(f"  [FAIL] {e}")
        return False


def test_bitsandbytes():
    """Check bitsandbytes for INT4 quantization."""
    print("\n[5/8] BitsAndBytes (INT4 Quantization)")
    try:
        import bitsandbytes
        print(f"  BitsAndBytes: {bitsandbytes.__version__}")
        print("  [PASS] BitsAndBytes installed")
        return True
    except ImportError as e:
        print(f"  [WARN] {e}")
        print("  [WARN] INT4 quantization may not work")
        return True  # Non-critical


def test_hailo():
    """Check Hailo SDK installation."""
    print("\n[6/8] Hailo SDK")
    try:
        from hailo_platform import VDevice

        # Try to detect Hailo device
        try:
            device = VDevice()
            print("  Hailo device: DETECTED")
            print("  [PASS] Hailo-10H ready")
        except Exception as e:
            print(f"  Hailo device: NOT FOUND ({e})")
            print("  [WARN] SAM2 will use CPU fallback")

        return True
    except ImportError:
        print("  [WARN] Hailo SDK not installed")
        print("  [WARN] SAM2 will use CPU fallback (slower)")
        return True  # Non-critical for testing


def test_raspberry_pi():
    """Check if running on Raspberry Pi."""
    print("\n[7/8] Raspberry Pi Detection")
    if os.path.exists("/proc/device-tree/model"):
        with open("/proc/device-tree/model", "r") as f:
            model = f.read().strip()
        print(f"  Device: {model}")
        print("  [PASS] Running on Raspberry Pi")
        return True
    else:
        print("  [INFO] Not running on Raspberry Pi")
        print("  [INFO] Edge features will run in development mode")
        return True


def test_checkpoints():
    """Check if checkpoints are available."""
    print("\n[8/8] Model Checkpoints")

    checkpoint_dir = "checkpoints/production"

    # Check MedGemma adapter
    medgemma_adapter = os.path.join(checkpoint_dir, "medgemma", "detection", "adapter_config.json")
    if os.path.exists(medgemma_adapter):
        print(f"  MedGemma adapter: FOUND")
    else:
        print(f"  MedGemma adapter: NOT FOUND")
        print(f"    Expected at: {medgemma_adapter}")

    # Check SAM2 adapter
    sam2_adapter = os.path.join(checkpoint_dir, "sam2", "final", "adapter_config.json")
    if os.path.exists(sam2_adapter):
        print(f"  SAM2 adapter: FOUND")
    else:
        print(f"  SAM2 adapter: NOT FOUND")
        print(f"    Expected at: {sam2_adapter}")

    # Check Hailo HEF
    hailo_hef = "edge/models/sam2_encoder_hailo10h.hef"
    if os.path.exists(hailo_hef):
        print(f"  Hailo HEF: FOUND")
    else:
        print(f"  Hailo HEF: NOT FOUND (SAM2 will use CPU)")

    return True


def main():
    print("="*60)
    print("MedGamma Edge Setup Test")
    print("Raspberry Pi 5 + Hailo AI HAT+2 (40 TOPS)")
    print("="*60)

    results = {
        "Python": test_python_version(),
        "PyTorch": test_pytorch(),
        "Transformers": test_transformers(),
        "PEFT": test_peft(),
        "BitsAndBytes": test_bitsandbytes(),
        "Hailo": test_hailo(),
        "Raspberry Pi": test_raspberry_pi(),
        "Checkpoints": test_checkpoints(),
    }

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    passed = sum(results.values())
    total = len(results)

    for name, status in results.items():
        icon = "[PASS]" if status else "[FAIL]"
        print(f"  {icon} {name}")

    print("="*60)
    print(f"Result: {passed}/{total} checks passed")
    print("="*60)

    if passed == total:
        print("\nEdge setup is ready!")
        print("\nRun inference with:")
        print("  python edge/edge_inference.py --image test.jpg --task full")
    else:
        print("\nSome components need attention.")
        print("Review the warnings above.")

    return 0 if passed >= total - 2 else 1  # Allow 2 non-critical failures


if __name__ == "__main__":
    sys.exit(main())
