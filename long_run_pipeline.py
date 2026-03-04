import os
import sys
import time
import subprocess
import argparse
from datetime import datetime
from src.utils.persistence import setup_persistent_logging

def run_command(name, cmd, log_file):
    """Run a command and monitor success with real-time persistent logging."""
    print(f"\n{'='*80}")
    print(f"  EXECUTING STAGE: {name}")
    print(f"  COMMAND:         {' '.join(cmd)}")
    print(f"  START TIME:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    # We use subprocess.Popen and read from its pipe in real-time
    # to ensure all output is captured by the PersistentTee.
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1 # Line buffered
        )
        
        # Read and print output as it comes
        for line in process.stdout:
            sys.stdout.write(line)
            # system's sys.stdout.write already calls PersistenceTee.write which fsyncs
            
        process.wait()
        return_code = process.returncode
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Failed to launch stage '{name}': {e}")
        return False
    
    elapsed = time.time() - start_time
    
    if return_code == 0:
        print(f"\n[SUCCESS] Stage '{name}' completed in {elapsed/60:.2f} minutes.")
        return True
    else:
        print(f"\n[ERROR] Stage '{name}' failed with exit code {return_code}.")
        return False

def check_checkpoint(path):
    """Check if a checkpoint exists at the path."""
    config_path = os.path.join(path, "adapter_config.json")
    return os.path.exists(config_path)

def main():
    parser = argparse.ArgumentParser(description="MedGamma Robust E2E Pipeline Orchestrator")
    parser.add_argument("--data-dir", default="medical_data", help="Path to data directory")
    parser.add_argument("--checkpoint-dir", default="checkpoints/production", help="Path to checkpoint directory")
    parser.add_argument("--cache-dir", default="D:\\medgamma_cache", help="Path to disk cache")
    parser.add_argument("--mode", default="balanced", choices=["turbo", "balanced", "full", "extended"], help="Detection training mode")
    parser.add_argument("--skip-finished", action="store_true", help="Skip stages where checkpoints already exist")
    parser.add_argument("--dry-run", action="store_true", help="Print the command sequence without executing")
    parser.add_argument("--log-path", default="pipeline_exec.log", help="Path to the persistent log file")
    
    args = parser.parse_args()
    
    # Set up persistent logging (fsync-enabled)
    setup_persistent_logging(args.log_path)
    
    print(f"MedGamma E2E Pipeline Orchestrator")
    print(f"Configuration:")
    print(f"  - Data Dir:       {args.data_dir}")
    print(f"  - Checkpoint Dir: {args.checkpoint_dir}")
    print(f"  - Cache Dir:      {args.cache_dir}")
    print(f"  - Mode:           {args.mode}")
    print(f"  - Log File:       {args.log_path} (fsync enabled)")
    print("-" * 40)
    
    python = sys.executable
    stages = []
    
    # Stage 1: Validation
    stages.append({
        "name": "Pipeline Validation",
        "cmd": [python, "test_pipeline.py"],
        "critical": False,
        "check": None
    })
    
    # Stage 2: Curriculum Training
    stages.append({
        "name": "Curriculum Training",
        "cmd": [python, "run_curriculum.py", "--data-dir", args.data_dir, "--output-dir", args.checkpoint_dir],
        "critical": True,
        "check": os.path.join(args.checkpoint_dir, "vqa")
    })
    
    # Stage 3: Detection Cache
    stages.append({
        "name": "Pre-build Detection Cache",
        "cmd": [python, "retrain_detection_optimized.py", "--preprocess-cache", "--disk-cache-dir", args.cache_dir, "--data_dir", args.data_dir],
        "critical": True,
        "check": os.path.join(args.cache_dir, "img_0.pt") # Check for first cached file
    })
    
    # Stage 4: Detection Training
    stages.append({
        "name": f"Detection Training ({args.mode})",
        "cmd": [python, "retrain_detection_optimized.py", f"--{args.mode}", "--disk-cache", "--disk-cache-dir", args.cache_dir, "--data_dir", args.data_dir],
        "critical": True,
        "check": os.path.join(args.checkpoint_dir, "medgemma", "detection")
    })
    
    # Stage 5: SAM2 Training
    stages.append({
        "name": "SAM2 Training",
        "cmd": [python, "-m", "src.train.train_sam2_medical", "--data-dir", args.data_dir, "--output-dir", os.path.join(args.checkpoint_dir, "sam2"), "--epochs", "5"],
        "critical": True,
        "check": os.path.join(args.checkpoint_dir, "sam2", "final")
    })
    
    # Stage 6: Multi-Stage Evaluation
    stages.append({
        "name": "Multi-Stage Evaluation",
        "cmd": [python, "-m", "src.eval.evaluate_all_stages", "--checkpoint", args.checkpoint_dir, "--max_samples", "200"],
        "critical": True,
        "check": None
    })
    
    # Stage 7: Blind Benchmark
    stages.append({
        "name": "Blind Benchmark",
        "cmd": [python, "blind_benchmark.py", "--checkpoint", args.checkpoint_dir, "--data-dir", args.data_dir],
        "critical": False,
        "check": None
    })
    
    overall_start = time.time()
    
    for stage in stages:
        # Check if we should skip
        if args.skip_finished and stage["check"]:
            if os.path.exists(stage["check"]):
                print(f"\n[SKIP] Stage '{stage['name']}' already exists at {stage['check']}. Skipping.")
                continue
        
        if args.dry_run:
            print(f"\n[DRY-RUN] Would execute: {stage['name']}")
            print(f"          Command:      {' '.join(stage['cmd'])}")
            continue

        success = run_command(stage["name"], stage["cmd"], args.log_path)
        
        if not success and stage["critical"]:
            print(f"\n[HALT] Critical stage '{stage['name']}' failed. Terminating pipeline.")
            sys.exit(1)
            
    overall_elapsed = time.time() - overall_start
    print(f"\n{'='*80}")
    print(f"  PIPELINE FINISHED")
    print(f"  TOTAL ELAPSED TIME: {overall_elapsed/3600:.2f} hours")
    print(f"  LOG FILE:           {args.log_path}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
