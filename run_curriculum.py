"""
Run Curriculum Training for MedGamma.

This script orchestrates the multi-stage curriculum learning across all datasets.

Usage:
    python run_curriculum.py                    # Full training
    python run_curriculum.py --quick            # Quick test (100 samples/stage)
    python run_curriculum.py --stage detection  # Start from detection stage
"""

import os
import sys
import argparse
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

def check_environment():
    """Check that all required packages and data are available."""
    print("="*60)
    print("MedGamma Curriculum Training - Environment Check")
    print("="*60)
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"[OK] CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("[X] CUDA not available - training will be SLOW")
    
    # Check data directories
    data_dirs = [
        "medical_data/vinbigdata-chest-xray",
        "medical_data/nih chest xray",
        "medical_data/chest_xray_kaggle",
        "medical_data/Slake1.0",
        "medical_data/osfstorage-archive",
        "medical_data/Brain Tumor MRI Dataset",
        "medical_data/Brain tumor multimodal image (CT & MRI)"
    ]
    
    print("\nDataset Check:")
    for d in data_dirs:
        if os.path.exists(d):
            print(f"  [OK] {os.path.basename(d)}")
        else:
            print(f"  [X] {os.path.basename(d)} (NOT FOUND)")
    
    # Check for HuggingFace token (hardcoded for development)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        print(f"\n[OK] HuggingFace token configured")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Curriculum Training for MedGamma Multi-Task Clinical AI"
    )
    
    parser.add_argument(
        "--data-dir", 
        default="medical_data",
        help="Base directory containing all datasets"
    )
    
    parser.add_argument(
        "--output-dir",
        default="checkpoints/curriculum",
        help="Directory to save checkpoints"
    )
    
    parser.add_argument(
        "--stage",
        default="screening",
        choices=["screening", "modality", "detection", "vqa", "segmentation"],
        help="Stage to start training from"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode - 100 samples per stage, 1 epoch"
    )
    
    parser.add_argument(
        "--detection-only",
        action="store_true",
        help="Only train detection stage (for VinDr F1 fix)"
    )
    
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip environment check"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # CRITICAL FIX: Sanitize sys.argv
    # Some imported modules might have their own argument parsers that run on import.
    # We strip our arguments to prevent downstream crashes (e.g., "unrecognized arguments: --quick").
    sys.argv = [sys.argv[0]]
    
    # Environment check
    if not args.skip_check:
        check_environment()
    
    # Import trainer (AFTER parsing and sanitizing args)
    from train.curriculum_trainer import CurriculumTrainer, CurriculumConfig, TrainingStage
    
    # Create config
    config = CurriculumConfig(
        base_data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    # Quick test mode
    if args.quick:
        print("\n[QUICK MODE] Limiting to 100 samples per stage, 1 epoch")
        for stage in config.stages:
            stage.max_samples = 100
            stage.epochs = 1
    
    # Detection-only mode
    if args.detection_only:
        print("\n[DETECTION ONLY] Training only detection stage")
        args.stage = "detection"
        # Keep only detection stage config
        config.stages = [s for s in config.stages if s.name == "detection"]
        config.stages[0].epochs = 3  # More epochs for detection
    
    # Map stage name to enum
    stage_map = {
        "screening": TrainingStage.SCREENING,
        "modality": TrainingStage.MODALITY,
        "detection": TrainingStage.DETECTION,
        "vqa": TrainingStage.VQA,
        "segmentation": TrainingStage.SEGMENTATION
    }
    
    start_stage = stage_map[args.stage]
    
    # Create trainer and run
    print(f"\nStarting curriculum training from stage: {args.stage}")
    trainer = CurriculumTrainer(config)
    trainer.train_all(start_stage=start_stage)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Checkpoints saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
