"""
GemSAM Interactive CLI - Command-line launcher for all pipeline operations.

Run:  python cli.py
"""

import os
import sys
import subprocess
import time
import io
from datetime import datetime

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Enable ANSI on Windows
os.system("")

# ---- Colors ----
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
MAGENTA = "\033[95m"
BLUE    = "\033[94m"
WHITE   = "\033[97m"


LOGO = f"""{CYAN}{BOLD}
    ______                 _____ ___    __  ___
   / ____/__  ____ ___    / ___//   |  /  |/  /
  / / __/ _ \\/ __ `__ \\   \\__ \\/ /| | / /|_/ /
 / /_/ /  __/ / / / / /  ___/ / ___ |/ /  / /
 \\____/\\___/_/ /_/ /_/  /____/_/  |_/_/  /_/{RESET}

  {DIM}Agentic Medical Image Analysis{RESET}
  {DIM}MedGemma 1.5 + SAM 2 + Edge Deployment{RESET}
  {DIM}--------------------------------------------{RESET}
"""


# ---- Settings ----
settings = {
    "data_dir":       "medical_data",
    "checkpoint_dir": "checkpoints/production",
    "cache_dir":      "D:\\medgamma_cache",
    "detection_mode": "full",
    "sam2_epochs":    5,
    "sam2_dataset":   "all",
    "max_eval":       200,
    "blind_samples":  -1,
    "medmnist_n":     100,
}


PYTHON = sys.executable
ROOT = os.path.dirname(os.path.abspath(__file__))


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def pause():
    input(f"\n  {DIM}Press Enter to continue...{RESET}")


def header(title):
    print(f"\n  {CYAN}{'=' * 56}{RESET}")
    print(f"  {CYAN}{BOLD}  {title}{RESET}")
    print(f"  {CYAN}{'=' * 56}{RESET}\n")


def run(cmd, label=""):
    if label:
        print(f"\n  {YELLOW}>{RESET} {BOLD}{label}{RESET}")
    print(f"  {DIM}$ {' '.join(cmd)}{RESET}\n")
    start = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - start
    if result.returncode == 0:
        print(f"\n  {GREEN}[OK]{RESET} Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    else:
        print(f"\n  {RED}[FAIL]{RESET} Exit code {result.returncode}")
    return result.returncode == 0


def ask(prompt, default=""):
    val = input(f"  {prompt} [{default}]: ").strip()
    return val if val else default


# ---- Menu ----

def show_main_menu():
    clear()
    print(LOGO)
    print(f"  {WHITE}{BOLD}MAIN MENU{RESET}")
    print()
    print(f"  {CYAN}--- PIPELINE ---------------------------------{RESET}")
    print(f"   {GREEN}1{RESET}   Run Full Robust Pipeline  {DIM}E2E + persistent log{RESET}")
    print(f"   {GREEN}2{RESET}   Quick Pipeline             {DIM}~1h turbo mode{RESET}")
    print(f"   {GREEN}3{RESET}   Evaluation Only            {DIM}no training{RESET}")
    print()
    print(f"  {CYAN}--- TRAINING ---------------------------------{RESET}")
    print(f"   {BLUE}4{RESET}   Curriculum Training        {DIM}screen/modal/VQA{RESET}")
    print(f"   {BLUE}5{RESET}   Detection Training         {DIM}main fine-tuning{RESET}")
    print(f"   {BLUE}6{RESET}   SAM2 Segmentation          {DIM}BUSI+SLAKE{RESET}")
    print()
    print(f"  {CYAN}--- EVALUATION -------------------------------{RESET}")
    print(f"   {MAGENTA}7{RESET}   Multi-Stage Eval           {DIM}all adapters{RESET}")
    print(f"   {MAGENTA}8{RESET}   Blind Benchmark            {DIM}unseen data{RESET}")
    print(f"   {MAGENTA}9{RESET}   MedMNIST External          {DIM}6 subsets{RESET}")
    print()
    print(f"  {CYAN}--- TOOLS ------------------------------------{RESET}")
    print(f"   {YELLOW}10{RESET}  Pipeline Validation        {DIM}pre-flight checks{RESET}")
    print(f"   {YELLOW}11{RESET}  Launch Demo (Gradio)       {DIM}web interface{RESET}")
    print(f"   {YELLOW}12{RESET}  Settings                   {DIM}paths & config{RESET}")
    print()
    print(f"   {DIM}0   Exit{RESET}")
    print()

    return input(f"  {CYAN}>{RESET} Choose [1-12]: ").strip()


def show_settings():
    clear()
    print(LOGO)
    header("SETTINGS")
    s = settings
    print(f"   {WHITE}1{RESET}  Data directory      : {GREEN}{s['data_dir']}{RESET}")
    print(f"   {WHITE}2{RESET}  Checkpoint directory : {GREEN}{s['checkpoint_dir']}{RESET}")
    print(f"   {WHITE}3{RESET}  Cache directory      : {GREEN}{s['cache_dir']}{RESET}")
    print(f"   {WHITE}4{RESET}  Detection mode       : {GREEN}{s['detection_mode']}{RESET}  {DIM}(turbo/balanced/full/extended){RESET}")
    print(f"   {WHITE}5{RESET}  SAM2 epochs          : {GREEN}{s['sam2_epochs']}{RESET}")
    print(f"   {WHITE}6{RESET}  SAM2 dataset         : {GREEN}{s['sam2_dataset']}{RESET}  {DIM}(all/busi/slake){RESET}")
    print(f"   {WHITE}7{RESET}  Max eval samples     : {GREEN}{s['max_eval']}{RESET}  {DIM}(-1 = all){RESET}")
    print(f"   {WHITE}8{RESET}  Blind test samples   : {GREEN}{s['blind_samples']}{RESET}  {DIM}(-1 = all){RESET}")
    print(f"   {WHITE}9{RESET}  MedMNIST per subset  : {GREEN}{s['medmnist_n']}{RESET}")
    print(f"\n   {DIM}0  Back{RESET}")
    print()

    choice = input(f"  {CYAN}>{RESET} Edit setting [1-9]: ").strip()

    if choice == "1":
        settings["data_dir"] = ask("Data directory", s["data_dir"])
    elif choice == "2":
        settings["checkpoint_dir"] = ask("Checkpoint directory", s["checkpoint_dir"])
    elif choice == "3":
        settings["cache_dir"] = ask("Cache directory", s["cache_dir"])
    elif choice == "4":
        settings["detection_mode"] = ask("Detection mode", s["detection_mode"])
    elif choice == "5":
        settings["sam2_epochs"] = int(ask("SAM2 epochs", str(s["sam2_epochs"])))
    elif choice == "6":
        settings["sam2_dataset"] = ask("SAM2 dataset", s["sam2_dataset"])
    elif choice == "7":
        settings["max_eval"] = int(ask("Max eval samples", str(s["max_eval"])))
    elif choice == "8":
        settings["blind_samples"] = int(ask("Blind test samples", str(s["blind_samples"])))
    elif choice == "9":
        settings["medmnist_n"] = int(ask("MedMNIST samples per subset", str(s["medmnist_n"])))


# ---- Actions ----

def action_full_pipeline():
    header("FULL ROBUST PIPELINE")
    s = settings
    print(f"  {YELLOW}NOTE:{RESET} This uses the robust orchestrator with persistent logging.")
    print(f"        Output is mirrored to {BOLD}pipeline_exec.log{RESET} with fsync.")
    print(f"        Ideal for long, unattended clinical validation runs.\n")
    
    run([
        PYTHON, "long_run_pipeline.py",
        "--data-dir", s["data_dir"],
        "--checkpoint-dir", s["checkpoint_dir"],
        "--cache-dir", s["cache_dir"],
        "--mode", s["detection_mode"],
        "--skip-finished"
    ], "Full Robust E2E Pipeline (fsync logging enabled)")
    pause()


def action_quick_pipeline():
    header("QUICK PIPELINE")
    run([
        PYTHON, "run_all.py", "--all", "--quick",
        "--data-dir", settings["data_dir"],
        "--checkpoint-dir", settings["checkpoint_dir"],
        "--no-demo",
    ], "Quick Pipeline (~1 hour, turbo mode)")
    pause()


def action_eval_only():
    header("EVALUATION ONLY")
    run([
        PYTHON, "run_all.py", "--eval-only",
        "--data-dir", settings["data_dir"],
        "--checkpoint-dir", settings["checkpoint_dir"],
        "--max-eval-samples", str(settings["max_eval"]),
        "--blind-test-samples", str(settings["blind_samples"]),
    ], "Evaluation + Blind Benchmark (no training)")
    pause()


def action_curriculum():
    header("CURRICULUM TRAINING")
    print(f"  {DIM}Stages: Screening > Modality > VQA{RESET}")
    print(f"  {DIM}Trains lightweight adapters for each stage{RESET}\n")
    run([
        PYTHON, "run_all.py", "--all",
        "--skip-detection", "--skip-sam2", "--skip-eval",
        "--data-dir", settings["data_dir"],
        "--checkpoint-dir", settings["checkpoint_dir"],
        "--no-demo",
    ], "Curriculum Training (3 stages)")
    pause()


def action_detection():
    header("DETECTION TRAINING")
    s = settings
    mode = s["detection_mode"]
    mode_info = {
        "turbo":    "~45-60 min, 1K images, demo quality",
        "balanced": "~2-3 hours, 3K images, good quality",
        "full":     "~14-24 hours, all images, best quality",
        "extended": "~24-48 hours, 3 epochs, maximum quality",
    }
    print(f"  {DIM}Mode: {mode} -- {mode_info.get(mode, '')}{RESET}\n")
    run([
        PYTHON, "retrain_detection_optimized.py",
        f"--{mode}",
        "--disk-cache",
        "--disk-cache-dir", s["cache_dir"],
    ], f"Detection Training ({mode} mode)")
    pause()


def action_sam2():
    header("SAM2 SEGMENTATION TRAINING")
    s = settings
    print(f"  {DIM}Dataset: {s['sam2_dataset']} | Epochs: {s['sam2_epochs']}{RESET}")
    print(f"  {DIM}Combined BUSI (647) + SLAKE (581) = ~1228 samples{RESET}\n")
    run([
        PYTHON, "-m", "src.train.train_sam2_medical",
        "--data-dir", s["data_dir"],
        "--dataset", s["sam2_dataset"],
        "--epochs", str(s["sam2_epochs"]),
        "--batch-size", "1",
        "--lr", "1e-4",
    ], "SAM2 Medical Fine-tuning")
    pause()


def action_multi_eval():
    header("MULTI-STAGE EVALUATION")
    run([
        PYTHON, "-m", "src.eval.evaluate_all_stages",
        "--checkpoint", settings["checkpoint_dir"],
        "--max_samples", str(settings["max_eval"]),
    ], "Evaluate all adapter stages")
    pause()


def action_blind_benchmark():
    header("BLIND BENCHMARK")
    s = settings
    print(f"  {DIM}6 sections: Screening, Modality, Detection, VQA, SAM2, MedMNIST{RESET}\n")
    run([
        PYTHON, "blind_benchmark.py",
        "--checkpoint", s["checkpoint_dir"],
        "--data-dir", s["data_dir"],
        "--max-samples", str(s["blind_samples"]),
        "--medmnist-samples", str(s["medmnist_n"]),
    ], "Blind Benchmark (unseen held-out data)")
    pause()


def action_medmnist_only():
    header("MEDMNIST EXTERNAL BENCHMARK")
    s = settings
    print(f"  {DIM}6 subsets: Breast, Pneumonia, Derma, Organ, Retina, Blood{RESET}")
    print(f"  {DIM}Samples per subset: {s['medmnist_n']}{RESET}\n")
    run([
        PYTHON, "blind_benchmark.py",
        "--checkpoint", s["checkpoint_dir"],
        "--data-dir", s["data_dir"],
        "--max-samples", "0",
        "--medmnist-samples", str(s["medmnist_n"]),
        "--sections", "medmnist",
    ], "MedMNIST v2 External Benchmark")
    pause()


def action_validate():
    header("PIPELINE VALIDATION")
    run([PYTHON, "test_pipeline.py"], "Pre-flight checks (6 tests)")
    pause()


def action_demo():
    header("GRADIO DEMO")
    print(f"  {DIM}Launching web interface at http://localhost:7860{RESET}\n")
    run([PYTHON, "demo/gradio_app.py"], "Gradio Demo UI")


# ---- Dispatch ----

ACTIONS = {
    "1":  action_full_pipeline,
    "2":  action_quick_pipeline,
    "3":  action_eval_only,
    "4":  action_curriculum,
    "5":  action_detection,
    "6":  action_sam2,
    "7":  action_multi_eval,
    "8":  action_blind_benchmark,
    "9":  action_medmnist_only,
    "10": action_validate,
    "11": action_demo,
    "12": show_settings,
}


def main():
    while True:
        choice = show_main_menu()
        if choice in ("0", "q", "quit", "exit"):
            print(f"\n  {CYAN}Goodbye!{RESET}\n")
            break
        elif choice in ACTIONS:
            ACTIONS[choice]()
        else:
            print(f"\n  {RED}Invalid choice.{RESET}")
            time.sleep(0.8)


if __name__ == "__main__":
    main()
