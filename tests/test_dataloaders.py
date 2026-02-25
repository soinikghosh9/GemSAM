"""
Test suite for medical data loaders.
Tests all datasets: SLAKE, VinDr, VQA-RAD, BUSI, Brain Tumor, etc.
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.factory import MedicalDatasetFactory


def test_loaders():
    print("=" * 60)
    print("TESTING DATA LOADERS")
    print("=" * 60)

    factory = MedicalDatasetFactory(base_data_dir="medical_data")

    # 1. Test SLAKE (VQA)
    print("\n--- Testing SLAKE (VQA) ---")
    try:
        loader = factory.get_loader("slake", task="vqa", batch_size=2, split="train")
        print(f"Loaded {len(loader.dataset)} SLAKE samples.")
        batch = next(iter(loader))
        print(f"Sample keys: {batch.keys()}")
        if "question" in batch:
            print(f"Question: {batch['question'][0][:100]}...")
    except Exception as e:
        print(f"SLAKE Failed: {e}")

    # 2. Test VinDr (Detection)
    print("\n--- Testing VinDr-CXR (Detection) ---")
    try:
        loader = factory.get_loader("vindr", task="detection", batch_size=2, split="train")
        print(f"Loaded {len(loader.dataset)} VinDr samples.")
        batch = next(iter(loader))
        print(f"Sample keys: {batch.keys()}")
        if "boxes" in batch:
            print(f"Boxes shape: {len(batch['boxes'][0])} boxes")
    except Exception as e:
        print(f"VinDr Failed: {e}")

    # 3. Test VQA-RAD
    print("\n--- Testing VQA-RAD ---")
    try:
        loader = factory.get_loader("vqa_rad", task="vqa", batch_size=2, split="train")
        print(f"Loaded {len(loader.dataset)} VQA-RAD samples.")
        batch = next(iter(loader))
        print(f"Sample keys: {batch.keys()}")
    except Exception as e:
        print(f"VQA-RAD Failed: {e}")

    # 4. Test BUSI (Classification/Segmentation - Blind Test)
    print("\n--- Testing BUSI (Blind Test Dataset) ---")
    try:
        loader = factory.get_loader("busi", task="both", batch_size=2, split="test")
        print(f"Loaded {len(loader.dataset)} BUSI test samples.")
        batch = next(iter(loader))
        print(f"Sample keys: {batch.keys()}")
        if "class_name" in batch:
            print(f"Class: {batch['class_name'][0]}")
        if "mask" in batch and batch["mask"][0] is not None:
            print(f"Mask shape: {batch['mask'][0].shape}")
    except Exception as e:
        print(f"BUSI Failed: {e}")

    # 5. Test SAM2 Dataset
    print("\n--- Testing SAM2 Dataset ---")
    try:
        loader = factory.get_loader("sam2", task="segmentation", batch_size=2, split="train")
        print(f"Loaded {len(loader.dataset)} SAM2 samples.")
        batch = next(iter(loader))
        print(f"Sample keys: {batch.keys()}")
        if "mask" in batch:
            print(f"Mask shape: {batch['mask'][0].shape}")
    except Exception as e:
        print(f"SAM2 Failed: {e}")

    # 6. Test Kaggle Pneumonia (Screening)
    print("\n--- Testing Kaggle Pneumonia (Screening) ---")
    try:
        loader = factory.get_loader("kaggle_pneumonia", task="screening", batch_size=2, split="train")
        print(f"Loaded {len(loader.dataset)} Kaggle Pneumonia samples.")
        batch = next(iter(loader))
        print(f"Sample keys: {batch.keys()}")
    except Exception as e:
        print(f"Kaggle Pneumonia Failed: {e}")

    # 7. Test Brain Tumor MRI
    print("\n--- Testing Brain Tumor MRI ---")
    try:
        loader = factory.get_loader("brain_tumor_mri", task="screening", batch_size=2, split="train")
        print(f"Loaded {len(loader.dataset)} Brain Tumor MRI samples.")
        batch = next(iter(loader))
        print(f"Sample keys: {batch.keys()}")
    except Exception as e:
        print(f"Brain Tumor MRI Failed: {e}")

    print("\n" + "=" * 60)
    print("DATA LOADER TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_loaders()
