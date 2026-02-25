import os
import argparse
import sys
import shutil
import subprocess

def run_step(command_args, step_name):
    """
    Runs a subprocess and checks return code.
    """
    print(f"\n{'='*20}")
    print(f"STARTING DEMO STEP: {step_name}")
    print(f"COMMAND: {' '.join(command_args)}")
    print(f"{'='*20}\n")
    
    try:
        full_cmd = [sys.executable, "-m"] + command_args
        subprocess.check_call(full_cmd)
        print(f"\n[SUCCESS] {step_name} completed.")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {step_name} failed with exit code {e.returncode}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="MedGemma DEMO Pipeline")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to train on")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    args = parser.parse_args()

    print("="*40)
    print("      Refined Demo Pipeline")
    print("="*40)

    # 1. Clear Cache
    print("[*] Clearing pycache...")
    for dirpath, dirnames, filenames in os.walk("."):
        if "__pycache__" in dirnames:
            try:
                shutil.rmtree(os.path.join(dirpath, "__pycache__"))
            except: pass
            dirnames.remove("__pycache__")
    
    # 2. Preprocess Data (Fast enough to run on all, but ensures cache exists)
    # Train Data
    run_step(["src.data.preprocess_images",
              "--src", "medical_data/vinbigdata-chest-xray/train",
              "--dest", "medical_data/vinbigdata-chest-xray/train_448"
    ], "Preprocessing TRAIN Data")
    
    # Test Data (Added)
    run_step(["src.data.preprocess_images",
              "--src", "medical_data/vinbigdata-chest-xray/test",
              "--dest", "medical_data/vinbigdata-chest-xray/test_448"
    ], "Preprocessing TEST Data")

    # 3. Train VinDr (Small)
    print("\n>>> DATASET 1: VinDr-CXR (Demonstration)")
    vindr_out = "outputs/demo_vindr"
    run_step([
        "src.train.train_medgemma", 
        "--dataset", "vindr",
        "--epochs", str(args.epochs),
        "--max_samples", str(args.samples),
        "--output_dir", vindr_out,
        "--grad_accum", "4" # Lower accum for demo so we see updates
    ], "Train VinDr (100 Samples)")

    # 4. Train SLAKE (Small)
    print("\n>>> DATASET 2: SLAKE (Demonstration)")
    slake_out = "outputs/demo_slake"
    run_step([
        "src.train.train_medgemma", 
        "--dataset", "slake",
        "--epochs", str(args.epochs),
        "--max_samples", str(args.samples),
        "--output_dir", slake_out,
        "--grad_accum", "4"
    ], "Train SLAKE (100 Samples)")

    # 4a. Train VQA-RAD (Small)
    print("\n>>> DATASET 2b: VQA-RAD (Demonstration)")
    vqa_out = "outputs/demo_vqa_rad"
    run_step([
        "src.train.train_medgemma", 
        "--dataset", "vqa_rad",
        "--epochs", str(args.epochs),
        "--max_samples", str(args.samples),
        "--output_dir", vqa_out,
        "--grad_accum", "4"
    ], "Train VQA-RAD (100 Samples)")


    # 4b. Train SAM 2 (Small - SLAKE Masks)
    print("\n>>> DATASET 3: SAM 2 Segmentor (SLAKE Masks)")
    sam2_out = "outputs/demo_sam2"
    run_step([
        "src.train.train_sam2_adapter",
        "--epochs", str(args.epochs),
        "--max_samples", str(args.samples),
        "--output_dir", sam2_out
    ], "Train SAM 2 (100 Samples)")

    # 5. Evaluation (Mock/Quick)
    # We'll run eval on the VinDr model just to prove it works
    print("\n>>> EVALUATION (VinDr Model)")
    run_step([
        "src.eval.evaluate_pipeline", 
        "--max_samples", str(args.samples)
    ], "Evaluate VinDr (Detection)")

    print("\n>>> EVALUATION (SLAKE VQA)")
    run_step([
        "src.eval.evaluate_pipeline", 
        "--dataset", "slake",
        "--task", "vqa",
        "--medgemma_adapter", os.path.join(slake_out, "final"),
        "--max_samples", str(args.samples)
    ], "Evaluate SLAKE (VQA)")

    print("\n>>> EVALUATION (VQA-RAD)")
    run_step([
        "src.eval.evaluate_pipeline", 
        "--dataset", "vqa_rad",
        "--task", "vqa",
        "--medgemma_adapter", os.path.join(vqa_out, "final"),
        "--max_samples", str(args.samples)
    ], "Evaluate VQA-RAD (VQA)")

    print("\n[DONE] Demo Pipeline Completed Successfully!")

if __name__ == "__main__":
    main()
