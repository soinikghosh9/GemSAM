"""
Run ALL MedGamma Pipeline Steps: Train → Evaluate → Blind Test.

Usage:
    python run_all.py --all                         # Full pipeline (curriculum + detection + SAM2 + eval + blind test)
    python run_all.py --all --quick                 # Quick test mode (~1h total)
    python run_all.py --all --skip-curriculum       # Skip curriculum, run detection + eval only
    python run_all.py --all --skip-detection        # Skip detection training (if already done)
    python run_all.py --eval-only                   # Evaluation + blind test only (no training)
    python run_all.py --all --disk-cache-dir E:\\cache  # Custom cache directory
"""

import os
import sys
import time
import subprocess
import argparse
from datetime import datetime


def run_step(name, cmd, cwd=None, required=True):
    """Run a pipeline step as a subprocess. Returns True on success."""
    print(f"\n{'='*70}")
    print(f"  STEP: {name}")
    print(f"  CMD:  {' '.join(cmd)}")
    print(f"  TIME: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*70}\n")

    start = time.time()
    result = subprocess.run(cmd, cwd=cwd or os.path.dirname(os.path.abspath(__file__)))
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n  [FAIL] {name} failed (exit code {result.returncode}) after {elapsed:.0f}s")
        if required:
            print(f"  Pipeline halted. Fix the error above and re-run.")
            return False
        else:
            print(f"  [WARN] Continuing despite failure (step marked optional)...")
            return True
    else:
        print(f"\n  [OK] {name} completed in {elapsed:.0f}s ({elapsed/60:.1f} min)")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="MedGamma Full Pipeline: Train + Evaluate + Blind Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all.py --all                   Full pipeline
  python run_all.py --all --quick           Quick test (~1h)
  python run_all.py --eval-only             Eval + blind test only
  python run_all.py --all --skip-curriculum Skip curriculum stages
        """
    )

    # Main modes
    parser.add_argument("--all", action="store_true",
                        help="Run ALL steps: curriculum -> detection -> SAM2 -> eval -> blind test -> demo")
    parser.add_argument("--eval-only", action="store_true",
                        help="Run evaluation + blind test only (no training)")

    # Training options
    parser.add_argument("--quick", action="store_true",
                        help="Quick test mode: minimal samples, 1 epoch per stage")
    parser.add_argument("--detection-mode", default="full",
                        choices=["turbo", "balanced", "full", "extended"],
                        help="Detection training mode (default: full)")
    parser.add_argument("--skip-curriculum", action="store_true",
                        help="Skip curriculum training (screening/modality/VQA)")
    parser.add_argument("--skip-detection", action="store_true",
                        help="Skip detection training")
    parser.add_argument("--skip-sam2", action="store_true",
                        help="Skip SAM2 segmentation training")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip standard evaluation")

    # Paths
    parser.add_argument("--data-dir", default="medical_data",
                        help="Base data directory")
    parser.add_argument("--checkpoint-dir", default="checkpoints/production",
                        help="Checkpoint directory")
    parser.add_argument("--disk-cache-dir", default="D:\\medgamma_cache",
                        help="Disk cache directory for detection training")

    # Evaluation
    parser.add_argument("--max-eval-samples", type=int, default=200,
                        help="Max samples for standard evaluation (-1 for all)")
    parser.add_argument("--blind-test-samples", type=int, default=-1,
                        help="Max samples for blind test (-1 for all)")

    # Demo
    parser.add_argument("--no-demo", action="store_true",
                        help="Don't launch Gradio demo at the end")

    args = parser.parse_args()

    if not args.all and not args.eval_only:
        parser.print_help()
        print("\nError: Specify --all or --eval-only")
        sys.exit(1)

    python = sys.executable  # Use the same Python interpreter
    results = []
    pipeline_start = time.time()

    print("\n" + "#"*70)
    print("  MEDGAMMA FULL PIPELINE")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Mode: {'QUICK TEST' if args.quick else args.detection_mode.upper()}")
    print("#"*70)

    # ================================================================
    # STEP 0: Validate Pipeline
    # ================================================================
    ok = run_step("Pipeline Validation", [python, "test_pipeline.py"], required=False)
    results.append(("Validation", ok))

    # ================================================================
    # STEP 1: Curriculum Training (screening, modality, VQA)
    # ================================================================
    if args.all and not args.skip_curriculum and not args.eval_only:
        curriculum_cmd = [
            python, "run_curriculum.py",
            "--data-dir", args.data_dir,
            "--output-dir", args.checkpoint_dir
        ]
        if args.quick:
            curriculum_cmd.append("--quick")

        ok = run_step("Curriculum Training (Screening + Modality + VQA)", curriculum_cmd)
        results.append(("Curriculum", ok))
        if not ok:
            sys.exit(1)
    else:
        results.append(("Curriculum", "SKIPPED"))

    # ================================================================
    # STEP 2: Pre-build Disk Cache
    # ================================================================
    if args.all and not args.skip_detection and not args.eval_only:
        cache_dir = args.disk_cache_dir
        if not os.path.exists(os.path.join(cache_dir, "images")):
            ok = run_step("Pre-build Disk Cache", [
                python, "retrain_detection_optimized.py",
                "--preprocess-cache",
                "--disk-cache-dir", cache_dir,
                "--data_dir", args.data_dir
            ])
            results.append(("Disk Cache", ok))
            if not ok:
                sys.exit(1)
        else:
            print(f"\n  [SKIP] Disk cache already exists at {cache_dir}")
            results.append(("Disk Cache", "EXISTS"))

    # ================================================================
    # STEP 3: Detection Training
    # ================================================================
    if args.all and not args.skip_detection and not args.eval_only:
        det_cmd = [
            python, "retrain_detection_optimized.py",
            f"--{args.detection_mode}",
            "--disk-cache",
            "--disk-cache-dir", args.disk_cache_dir,
            "--data_dir", args.data_dir
        ]
        if args.quick:
            det_cmd = [
                python, "retrain_detection_optimized.py",
                "--turbo",
                "--disk-cache",
                "--disk-cache-dir", args.disk_cache_dir,
                "--data_dir", args.data_dir
            ]

        ok = run_step(f"Detection Training ({args.detection_mode})", det_cmd)
        results.append(("Detection Training", ok))
        if not ok:
            sys.exit(1)
    else:
        results.append(("Detection Training", "SKIPPED"))

    # ================================================================
    # STEP 4: SAM2 Segmentation Training
    # ================================================================
    if args.all and not args.skip_sam2 and not args.eval_only:
        sam_epochs = "2" if args.quick else "5"
        sam_cmd = [
            python, "-m", "src.train.train_sam2_medical",
            "--data-dir", args.data_dir,
            "--output-dir", os.path.join(args.checkpoint_dir, "sam2"),
            "--epochs", sam_epochs,
            "--batch-size", "1",
            "--lr", "1e-4"
        ]
        if args.quick:
            sam_cmd.append("--quick")

        ok = run_step("SAM2 Segmentation Training (BUSI)", sam_cmd, required=False)
        results.append(("SAM2 Training", ok))
    else:
        results.append(("SAM2 Training", "SKIPPED"))

    # ================================================================
    # STEP 5: Multi-Stage Evaluation
    # ================================================================
    if not args.skip_eval:
        eval_cmd = [
            python, "-m", "src.eval.evaluate_all_stages",
            "--checkpoint", args.checkpoint_dir,
            "--max_samples", str(args.max_eval_samples)
        ]
        ok = run_step("Multi-Stage Evaluation (all stages)", eval_cmd)
        results.append(("Evaluation", ok))
    else:
        results.append(("Evaluation", "SKIPPED"))

    # ================================================================
    # STEP 6: Blind Benchmark (multi-modal, all stages)
    # ================================================================
    blind_cmd = [
        python, "blind_benchmark.py",
        "--checkpoint", args.checkpoint_dir,
        "--data-dir", args.data_dir if hasattr(args, 'data_dir') else "medical_data",
        "--max-samples", str(args.blind_test_samples),
        "--medmnist-samples", "100"
    ]
    ok = run_step("Blind Benchmark (multi-modal, unseen data)", blind_cmd)
    results.append(("Blind Benchmark", ok))

    # ================================================================
    # STEP 7: Launch Demo (optional)
    # ================================================================
    if not args.no_demo and not args.eval_only:
        print("\n  Launching Gradio demo...")
        print("  Press Ctrl+C to stop the demo.\n")
        subprocess.run([python, "demo/gradio_app.py"])

    # ================================================================
    # SUMMARY
    # ================================================================
    total_time = time.time() - pipeline_start
    print("\n" + "#"*70)
    print("  PIPELINE SUMMARY")
    print("#"*70)
    print(f"  {'Step':<35} {'Status':<15}")
    print(f"  {'-'*35} {'-'*15}")
    for name, status in results:
        if status is True:
            s = "OK"
        elif status is False:
            s = "FAILED"
        else:
            s = str(status)
        print(f"  {name:<35} {s:<15}")
    print(f"\n  Total time: {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"  Results:    outputs/multi_stage_eval/evaluation_report.json")
    print(f"  Raw data:   outputs/multi_stage_eval/raw_samples.json")
    print("#"*70)


if __name__ == "__main__":
    main()
