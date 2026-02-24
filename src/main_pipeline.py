import os
import argparse
import shutil
import subprocess
import sys
import pandas as pd
import matplotlib.pyplot as plt
import json

def clear_caches(root_dir="."):
    """
    Cleans up __pycache__, .pytest_cache, and optionally unused outputs.
    """
    print(f"[*] Cleaning caches in {root_dir}...")
    cleaned_count = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Remove __pycache__
        if "__pycache__" in dirnames:
            p = os.path.join(dirpath, "__pycache__")
            try:
                shutil.rmtree(p)
                cleaned_count += 1
            except Exception as e:
                print(f"Error removing {p}: {e}")
            dirnames.remove("__pycache__") # Don't recurse into it
            
    print(f"[*] Removed {cleaned_count} __pycache__ directories.")

def run_step(command_args, step_name):
    """
    Runs a subprocess and checks return code.
    """
    print(f"\n{'='*20}")
    print(f"STARTING STEP: {step_name}")
    print(f"COMMAND: {' '.join(command_args)}")
    print(f"{'='*20}\n")
    
    try:
        # Use sys.executable to ensure we use the same python interpreter
        full_cmd = [sys.executable, "-m"] + command_args
        subprocess.check_call(full_cmd)
        print(f"\n[SUCCESS] {step_name} completed.")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {step_name} failed with exit code {e.returncode}")
        sys.exit(1)

def plot_results(medgemma_out, sam2_out):
    """
    Generates plots from CSV logs.
    """
    print("\n[*] Generating Plots...")
    os.makedirs("plots", exist_ok=True)
    
    # 1. MedGemma Loss (VinDr)
    mg_loss_path = os.path.join(medgemma_out, "loss.csv")
    if os.path.exists(mg_loss_path):
        try:
            df = pd.read_csv(mg_loss_path)
            plt.figure(figsize=(10, 5))
            plt.plot(df['step'], df['loss'], label="Training Loss")
            plt.title("MedGemma Training Loss")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig("plots/medgemma_loss.png")
            plt.close()
            print("Saved plots/medgemma_loss.png")
        except Exception as e:
            print(f"Failed to plot MedGemma loss: {e}")
            
    # 2. SAM 2 Loss
    sam_loss_path = os.path.join(sam2_out, "loss.csv")
    if os.path.exists(sam_loss_path):
        try:
            df = pd.read_csv(sam_loss_path)
            plt.figure(figsize=(10, 5))
            plt.plot(df['epoch'], df['loss'], label="Training Loss")
            plt.title("SAM 2 Adapter Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig("plots/sam2_loss.png")
            plt.close()
            print("Saved plots/sam2_loss.png")
        except Exception as e:
            print(f"Failed to plot SAM 2 loss: {e}")

def run_inference(checkpoint_dir, image_path=None, query=None):
    """Run inference using trained checkpoints."""
    from src.orchestrator import ClinicalOrchestrator

    orchestrator = ClinicalOrchestrator(checkpoint_dir=checkpoint_dir)

    # Find test image if not provided
    if not image_path:
        possible_paths = [
            "medical_data/vinbigdata-chest-xray/train_448",
            "medical_data/chest_xray_kaggle/train/PNEUMONIA",
            "medical_data/Slake1.0/imgs/xmlab0"
        ]
        for base_path in possible_paths:
            if os.path.exists(base_path):
                for f in os.listdir(base_path):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(base_path, f)
                        break
                if image_path:
                    break

    if not image_path or not os.path.exists(image_path):
        print("No test image found. Provide --image path.")
        return

    query = query or "Detect any pathologies and localize them."
    print(f"\nRunning inference on: {image_path}")
    print(f"Query: {query}")

    state = orchestrator.run_pipeline(image_path, query)

    print(f"\n{'='*60}")
    print("INFERENCE COMPLETE")
    print(f"{'='*60}")
    print(f"Detections: {len(state.get('detections', []))}")
    print(f"Segmentations: {len(state.get('segmentations', []))}")

    if state.get('error'):
        print(f"Error: {state['error']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedGemma Full Pipeline Orchestrator")
    parser.add_argument("--clear_cache", action="store_true", help="Clear __pycache__ before running")
    parser.add_argument("--skip_train", action="store_true", help="Skip training steps")
    parser.add_argument("--skip_eval", action="store_true", help="Skip evaluation step")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs for training (default 5)")
    parser.add_argument("--output_root", type=str, default="outputs", help="Root directory for outputs")
    parser.add_argument("--inference", action="store_true", help="Run inference mode")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint directory for inference")
    parser.add_argument("--image", type=str, default=None, help="Image path for inference")
    parser.add_argument("--query", type=str, default=None, help="Query for inference")

    args = parser.parse_args()

    # Handle inference mode
    if args.inference:
        checkpoint = args.checkpoint or "checkpoints/production"
        run_inference(checkpoint, args.image, args.query)
        sys.exit(0)
    
    if args.clear_cache:
        clear_caches()
        
    vindr_out = os.path.join(args.output_root, "medgemma_lora_vindr")
    slake_out = os.path.join(args.output_root, "medgemma_lora_slake")
    vqa_out = os.path.join(args.output_root, "medgemma_lora_vqa_rad")
    sam2_out = os.path.join(args.output_root, "sam2_adapter")
    
    if not args.skip_train:
        # --- PREPROCESSING ---
        # 1. VinDr Train
        run_step(["src.data.preprocess_images", 
                  "--src", "medical_data/vinbigdata-chest-xray/train", 
                  "--dest", "medical_data/vinbigdata-chest-xray/train_448"], 
                 "Preprocessing VinDr TRAIN")
        
        # 2. VinDr Test
        run_step(["src.data.preprocess_images", 
                  "--src", "medical_data/vinbigdata-chest-xray/test", 
                  "--dest", "medical_data/vinbigdata-chest-xray/test_448"], 
                 "Preprocessing VinDr TEST")

        # --- TRAINING ---
        # 1. MedGemma (VinDr Detection)
        run_step(["src.train.train_medgemma", 
                  "--dataset", "vindr",
                  "--epochs", str(args.epochs), 
                  "--output_dir", vindr_out], 
                 "MedGemma Training (VinDr Detection)")
        
        # 2. MedGemma (SLAKE VQA)
        run_step(["src.train.train_medgemma", 
                  "--dataset", "slake",
                  "--epochs", str(args.epochs), 
                  "--output_dir", slake_out], 
                 "MedGemma Training (SLAKE VQA)")

        # 3. MedGemma (VQA-RAD VQA)
        run_step(["src.train.train_medgemma", 
                  "--dataset", "vqa_rad",
                  "--epochs", str(args.epochs), 
                  "--output_dir", vqa_out], 
                 "MedGemma Training (VQA-RAD VQA)")

        # 4. SAM 2 (Segmentation)
        run_step(["src.train.train_sam2_adapter", 
                  "--epochs", str(args.epochs), 
                  "--output_dir", sam2_out], 
                 "SAM 2 Adapter Training")
        
    if not args.skip_eval:
        # --- EVALUATION ---
        # 1. VinDr (Detection)
        run_step(["src.eval.evaluate_pipeline", 
                  "--dataset", "vindr",
                  "--task", "detection",
                  "--medgemma_adapter", os.path.join(vindr_out, "final")], 
                 "Evaluation (VinDr Detection)")
        
        # 2. SLAKE (VQA)
        run_step(["src.eval.evaluate_pipeline", 
                  "--dataset", "slake",
                  "--task", "vqa",
                  "--medgemma_adapter", os.path.join(slake_out, "final")], 
                 "Evaluation (SLAKE VQA)")

        # 3. VQA-RAD (VQA)
        run_step(["src.eval.evaluate_pipeline", 
                  "--dataset", "vqa_rad",
                  "--task", "vqa",
                  "--medgemma_adapter", os.path.join(vqa_out, "final")], 
                 "Evaluation (VQA-RAD VQA)")
        
    # 4. Plots (Focusing on VinDr loss for now as primary)
    plot_results(vindr_out, sam2_out)
    
    print("\n[DONE] Full Pipeline Execution Completed.")
