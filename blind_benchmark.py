"""
Blind Benchmark: Multi-Modal Evaluation on Unseen Data + External Benchmarks.

Evaluates ALL MedGamma pipeline stages using strictly held-out test splits
and external MedMNIST v2 datasets for unbiased performance assessment.

Sections:
  1. Screening     — Kaggle Pneumonia test/  (NORMAL vs PNEUMONIA)
  2. Modality      — Brain Tumor MRI Testing/ (4-class)
  3. Detection     — VinBigData test_split.csv (bounding boxes)
  4. VQA           — SLAKE test.json (English questions)
  5. Segmentation  — BUSI validation split (SAM2 Dice/IoU)
  6. External      — MedMNIST v2 (6 subsets, classification)

Usage:
    python blind_benchmark.py --checkpoint checkpoints/production --data-dir medical_data
    python blind_benchmark.py --checkpoint checkpoints/production --data-dir medical_data --max-samples 50
"""

import os
import sys
import json
import time
import argparse
import random
import string
import gc
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =====================================================================
# Helpers
# =====================================================================

def safe_load_json(path):
    """Load JSON with multiple encoding attempts."""
    for enc in ["utf-8", "utf-8-sig", "latin-1"]:
        try:
            with open(path, encoding=enc) as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    return None


def normalize_answer(text):
    """Normalize answer for robust matching."""
    t = text.lower().strip()
    for prefix in ["the answer is", "based on the image,", "answer:",
                    "the image shows", "this image shows", "it shows", "yes,", "no,"]:
        if t.startswith(prefix):
            t = t[len(prefix):].strip()
    return t.strip(string.punctuation + " ")


def compute_dice(pred_mask, gt_mask):
    """Compute Dice coefficient between two binary masks."""
    pred = pred_mask.astype(bool).flatten()
    gt = gt_mask.astype(bool).flatten()
    intersection = np.sum(pred & gt)
    if pred.sum() + gt.sum() == 0:
        return 1.0  # Both empty
    return float(2.0 * intersection / (pred.sum() + gt.sum()))


def compute_iou_masks(pred_mask, gt_mask):
    """Compute IoU between two binary masks."""
    pred = pred_mask.astype(bool).flatten()
    gt = gt_mask.astype(bool).flatten()
    intersection = np.sum(pred & gt)
    union = np.sum(pred | gt)
    if union == 0:
        return 1.0
    return float(intersection / union)


# =====================================================================
# Section 1: Screening Benchmark
# =====================================================================

def benchmark_screening(model, data_dir, max_samples=-1):
    """Evaluate screening on Kaggle Pneumonia test split."""
    test_dir = os.path.join(data_dir, "chest_xray_kaggle", "test")
    if not os.path.exists(test_dir):
        test_dir = os.path.join(data_dir, "chest_xray_kaggle", "chest_xray", "test")
    if not os.path.exists(test_dir):
        return {"status": "skipped", "reason": f"Test dir not found: {test_dir}"}

    print("\n  Loading Kaggle Pneumonia test set...")
    samples = []
    for label_dir in ["NORMAL", "PNEUMONIA"]:
        label_path = os.path.join(test_dir, label_dir)
        if not os.path.exists(label_path):
            continue
        for fname in os.listdir(label_path):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                samples.append({
                    "image": os.path.join(label_path, fname),
                    "label": label_dir
                })

    if not samples:
        return {"status": "skipped", "reason": "No test images found"}

    random.shuffle(samples)
    if max_samples > 0:
        samples = samples[:max_samples]
    print(f"  Samples: {len(samples)} (NORMAL + PNEUMONIA)")

    correct, tp, tn, fp, fn = 0, 0, 0, 0, 0
    raw_samples = []

    for i, s in tqdm(enumerate(samples), total=len(samples), desc="  Screening"):
        gt_is_normal = s["label"] == "NORMAL"
        try:
            img = Image.open(s["image"]).convert("RGB")
            response, _ = model.analyze_image(img, task="screening")
            resp_lower = response.lower().strip()

            healthy_kw = ["healthy", "normal", "no significant", "unremarkable", "clear"]
            abnormal_kw = ["abnormal", "pneumonia", "infection", "opacity", "effusion",
                           "consolidation", "infiltrate", "mass", "nodule", "patholog"]

            pred_normal = any(kw in resp_lower for kw in healthy_kw)
            if any(kw in resp_lower for kw in abnormal_kw):
                pred_normal = False

            if pred_normal == gt_is_normal:
                correct += 1
            if gt_is_normal and pred_normal:
                tn += 1
            elif gt_is_normal and not pred_normal:
                fp += 1
            elif not gt_is_normal and not pred_normal:
                tp += 1
            else:
                fn += 1

            if i < 3:
                raw_samples.append({"gt": s["label"], "pred": response[:150]})
        except Exception as e:
            if i < 3:
                raw_samples.append({"gt": s["label"], "error": str(e)})

    total = len(samples)
    acc = correct / max(total, 1)
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)

    print(f"  Accuracy: {acc:.4f} | Sensitivity: {sensitivity:.4f} | "
          f"Specificity: {specificity:.4f} | F1: {f1:.4f}")

    return {
        "accuracy": round(acc, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "total": total,
        "raw_samples": raw_samples
    }


# =====================================================================
# Section 2: Modality Classification Benchmark
# =====================================================================

def benchmark_modality(model, data_dir, max_samples=-1):
    """Evaluate modality classification on Brain Tumor MRI Testing split."""
    test_dir = os.path.join(data_dir, "Brain Tumor MRI Dataset", "Testing")
    if not os.path.exists(test_dir):
        return {"status": "skipped", "reason": f"Not found: {test_dir}"}

    print("\n  Loading Brain Tumor MRI Testing split...")
    classes = ["glioma", "meningioma", "notumor", "pituitary"]
    samples = []
    for cls in classes:
        cls_dir = os.path.join(test_dir, cls)
        if not os.path.exists(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                samples.append({
                    "image": os.path.join(cls_dir, fname),
                    "class": cls
                })

    if not samples:
        return {"status": "skipped", "reason": "No test images found"}

    random.shuffle(samples)
    if max_samples > 0:
        samples = samples[:max_samples]
    print(f"  Samples: {len(samples)} ({', '.join(classes)})")

    correct = 0
    class_metrics = {c: {"tp": 0, "fp": 0, "fn": 0} for c in classes}
    raw_samples = []

    for i, s in tqdm(enumerate(samples), total=len(samples), desc="  Modality"):
        gt_class = s["class"]
        try:
            img = Image.open(s["image"]).convert("RGB")
            response, _ = model.analyze_image(img, task="modality")
            resp_lower = response.lower().strip()

            # Match predicted class
            pred_class = None
            for cls in classes:
                if cls in resp_lower:
                    pred_class = cls
                    break
            # Also check common alternatives
            if not pred_class:
                if "mri" in resp_lower:
                    pred_class = "mri_detected"
                elif "no tumor" in resp_lower or "no finding" in resp_lower:
                    pred_class = "notumor"

            if pred_class == gt_class:
                correct += 1
                class_metrics[gt_class]["tp"] += 1
            else:
                class_metrics[gt_class]["fn"] += 1
                if pred_class and pred_class in class_metrics:
                    class_metrics[pred_class]["fp"] += 1

            if i < 3:
                raw_samples.append({"gt": gt_class, "pred": response[:150]})
        except Exception as e:
            class_metrics[gt_class]["fn"] += 1
            if i < 3:
                raw_samples.append({"gt": gt_class, "error": str(e)})

    total = len(samples)
    acc = correct / max(total, 1)

    per_class = {}
    for cls in classes:
        m = class_metrics[cls]
        precision = m["tp"] / max(m["tp"] + m["fp"], 1)
        recall = m["tp"] / max(m["tp"] + m["fn"], 1)
        per_class[cls] = {"precision": round(precision, 4), "recall": round(recall, 4)}

    print(f"  Accuracy: {acc:.4f}")
    for cls, m in per_class.items():
        print(f"    {cls}: P={m['precision']:.3f} R={m['recall']:.3f}")

    return {
        "accuracy": round(acc, 4),
        "correct": correct, "total": total,
        "per_class": per_class,
        "raw_samples": raw_samples
    }


# =====================================================================
# Section 3: Detection Benchmark
# =====================================================================

def benchmark_detection(model, data_dir, max_samples=200):
    """Evaluate detection on VinBigData test split."""
    import csv

    csv_path = os.path.join(data_dir, "vinbigdata-chest-xray", "test_split.csv")
    if not os.path.exists(csv_path):
        return {"status": "skipped", "reason": f"Not found: {csv_path}"}

    img_dir = os.path.join(data_dir, "vinbigdata-chest-xray", "test")
    if not os.path.exists(img_dir):
        img_dir = os.path.join(data_dir, "vinbigdata-chest-xray", "test_448")

    print("\n  Loading VinBigData test split...")
    # Group annotations by image_id
    annotations = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = row["image_id"]
            if img_id not in annotations:
                annotations[img_id] = []
            annotations[img_id].append(row)

    # Find images with actual annotations
    image_ids = list(annotations.keys())
    random.shuffle(image_ids)
    if max_samples > 0:
        image_ids = image_ids[:max_samples]
    print(f"  Images: {len(image_ids)}")

    tp, fp, fn = 0, 0, 0
    total_images = 0
    raw_samples = []

    for idx, img_id in tqdm(enumerate(image_ids), total=len(image_ids), desc="  Detection"):
        # Find image file
        img_path = None
        for ext in [".png", ".jpg", ".jpeg", ".dicom"]:
            candidate = os.path.join(img_dir, f"{img_id}{ext}")
            if os.path.exists(candidate):
                img_path = candidate
                break

        if not img_path:
            continue

        gt_findings = annotations[img_id]
        gt_has_finding = any(r["class_name"] != "No finding" for r in gt_findings)

        try:
            img = Image.open(img_path).convert("RGB")
            response, _ = model.analyze_image(img, task="detection")
            total_images += 1

            # Parse response for findings
            resp_lower = response.lower()
            pred_has_finding = ("finding" in resp_lower and "no finding" not in resp_lower) or \
                               "abnormal" in resp_lower or \
                               any(cls in resp_lower for cls in [
                                   "pleural", "cardiomegaly", "pneumonia", "opacity",
                                   "effusion", "mass", "nodule", "consolidation",
                                   "fibrosis", "atelectasis", "edema", "emphysema"
                               ])

            if gt_has_finding and pred_has_finding:
                tp += 1
            elif gt_has_finding and not pred_has_finding:
                fn += 1
            elif not gt_has_finding and pred_has_finding:
                fp += 1
            # TN counted implicitly

            if idx < 5:
                raw_samples.append({
                    "image_id": img_id,
                    "gt_has_finding": gt_has_finding,
                    "gt_classes": [r["class_name"] for r in gt_findings],
                    "pred": response[:200]
                })
        except Exception as e:
            total_images += 1
            if idx < 3:
                raw_samples.append({"image_id": img_id, "error": str(e)})

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    print(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    print(f"  TP={tp} FP={fp} FN={fn} (out of {total_images} images)")

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn,
        "total_images": total_images,
        "raw_samples": raw_samples
    }


# =====================================================================
# Section 4: VQA Benchmark
# =====================================================================

def benchmark_vqa(model, data_dir, max_samples=-1):
    """Evaluate VQA on SLAKE test.json (English only)."""
    test_json = os.path.join(data_dir, "Slake1.0", "test.json")
    if not os.path.exists(test_json):
        return {"status": "skipped", "reason": f"Not found: {test_json}"}

    print("\n  Loading SLAKE test.json (English only)...")
    data = safe_load_json(test_json)
    if not data:
        return {"status": "skipped", "reason": "Failed to parse test.json"}

    # Filter English questions only
    samples = [q for q in data if q.get("q_lang", "en") == "en"]
    random.shuffle(samples)
    if max_samples > 0:
        samples = samples[:max_samples]
    print(f"  Questions: {len(samples)} (English)")

    imgs_dir = os.path.join(data_dir, "Slake1.0", "imgs")

    SYNONYM_GROUPS = [
        {"yes", "correct", "true", "affirmative", "positive"},
        {"no", "incorrect", "false", "negative", "none"},
        {"normal", "healthy", "no abnormality", "unremarkable"},
        {"abnormal", "pathological", "disease"},
        {"lung", "pulmonary"}, {"heart", "cardiac"},
        {"brain", "cerebral"}, {"liver", "hepatic"}, {"kidney", "renal"},
    ]

    def synonym_match(pred, gt):
        for group in SYNONYM_GROUPS:
            if pred in group and gt in group:
                return True
        return False

    def token_overlap(pred, gt):
        pt = set(pred.split())
        gt_t = set(gt.split())
        if not pt or not gt_t:
            return 0.0
        return len(pt & gt_t) / len(pt | gt_t)

    correct = 0
    total_count = 0
    raw_samples = []

    for i, q in tqdm(enumerate(samples), total=len(samples), desc="  VQA"):
        img_name = q.get("img_name", "")
        question = q.get("question", "")
        gt_answer = str(q.get("answer", ""))

        img_path = os.path.join(imgs_dir, img_name)
        if not os.path.exists(img_path):
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            response, _ = model.analyze_image(img, question, task="vqa")
            pred = response.strip()

            pred_clean = normalize_answer(pred)
            gt_clean = normalize_answer(gt_answer)

            is_correct = (pred_clean == gt_clean or
                          gt_clean in pred_clean or
                          pred_clean in gt_clean or
                          synonym_match(pred_clean, gt_clean) or
                          token_overlap(pred_clean, gt_clean) >= 0.5)

            if is_correct:
                correct += 1
            total_count += 1

            if i < 5:
                raw_samples.append({
                    "question": question[:100],
                    "gt": gt_answer,
                    "pred": pred[:150],
                    "correct": is_correct
                })
        except Exception as e:
            total_count += 1
            if i < 3:
                raw_samples.append({"question": question[:80], "error": str(e)})

    acc = correct / max(total_count, 1)
    print(f"  VQA Accuracy: {acc:.4f} ({correct}/{total_count})")

    return {
        "accuracy": round(acc, 4),
        "correct": correct,
        "total": total_count,
        "raw_samples": raw_samples
    }


# =====================================================================
# Section 5: SAM2 Segmentation Benchmark
# =====================================================================

def benchmark_segmentation(data_dir, checkpoint_dir, max_samples=-1):
    """Evaluate SAM2 segmentation on BUSI held-out samples."""
    try:
        from src.medsam_wrapper import MedSAMWrapper
    except ImportError:
        try:
            from medsam_wrapper import MedSAMWrapper
        except ImportError:
            return {"status": "skipped", "reason": "MedSAMWrapper not available"}

    # Load BUSI images and masks
    busi_dir = os.path.join(data_dir, "Dataset_BUSI_with_GT")
    if not os.path.exists(busi_dir):
        return {"status": "skipped", "reason": f"Not found: {busi_dir}"}

    print("\n  Loading BUSI dataset for segmentation evaluation...")
    samples = []
    for category in ["benign", "malignant"]:
        cat_dir = os.path.join(busi_dir, category)
        if not os.path.exists(cat_dir):
            continue
        for fname in os.listdir(cat_dir):
            if not fname.endswith(".png") or "_mask" in fname:
                continue
            base_name = fname.replace(".png", "")
            mask_path = os.path.join(cat_dir, f"{base_name}_mask.png")
            if os.path.exists(mask_path):
                samples.append({
                    "image": os.path.join(cat_dir, fname),
                    "mask": mask_path,
                    "category": category
                })

    if not samples:
        return {"status": "skipped", "reason": "No BUSI samples with masks"}

    # Use last 10% as validation/blind test (same as training split logic)
    random.seed(42)
    random.shuffle(samples)
    val_size = max(int(len(samples) * 0.1), 1)
    val_samples = samples[-val_size:]

    if max_samples > 0:
        val_samples = val_samples[:max_samples]
    print(f"  Validation samples: {len(val_samples)} (10% held-out)")

    # Load SAM2
    print("  Loading SAM2 model...")
    sam = MedSAMWrapper()
    sam.load()
    if sam.predictor == "SIMULATION":
        return {"status": "skipped", "reason": "SAM2 running in simulation mode"}

    dice_scores = []
    iou_scores = []
    raw_samples = []

    for i, s in tqdm(enumerate(val_samples), total=len(val_samples), desc="  Segmentation"):
        try:
            # Load GT mask
            gt_mask = np.array(Image.open(s["mask"]).convert("L"))
            gt_binary = (gt_mask > 127).astype(np.uint8)

            if gt_binary.max() == 0:
                continue

            # Get bounding box from GT mask (simulate detection output)
            rows = np.any(gt_binary, axis=1)
            cols = np.any(gt_binary, axis=0)
            y1, y2 = np.where(rows)[0][[0, -1]]
            x1, x2 = np.where(cols)[0][[0, -1]]

            # Add small margin (5%)
            h, w = gt_binary.shape
            margin_x = max(int((x2 - x1) * 0.05), 2)
            margin_y = max(int((y2 - y1) * 0.05), 2)
            box = [max(0, x1 - margin_x), max(0, y1 - margin_y),
                   min(w, x2 + margin_x), min(h, y2 + margin_y)]

            # Run SAM2
            sam.set_image(s["image"])
            masks, scores = sam.predict_mask(box)
            if masks is None or len(masks) == 0:
                continue

            # Use best mask by score
            best_idx = np.argmax(scores)
            pred_mask = masks[best_idx].astype(np.uint8)

            # Resize pred mask to match GT if needed
            if pred_mask.shape != gt_binary.shape:
                from PIL import Image as PILImage
                pred_pil = PILImage.fromarray(pred_mask * 255)
                pred_pil = pred_pil.resize((gt_binary.shape[1], gt_binary.shape[0]),
                                           PILImage.NEAREST)
                pred_mask = (np.array(pred_pil) > 127).astype(np.uint8)

            dice = compute_dice(pred_mask, gt_binary)
            iou = compute_iou_masks(pred_mask, gt_binary)
            dice_scores.append(dice)
            iou_scores.append(iou)

            if i < 3:
                raw_samples.append({
                    "image": os.path.basename(s["image"]),
                    "dice": round(dice, 4),
                    "iou": round(iou, 4)
                })
        except Exception as e:
            if i < 3:
                raw_samples.append({"image": os.path.basename(s["image"]), "error": str(e)})

    if not dice_scores:
        return {"status": "failed", "reason": "No valid segmentation results"}

    mean_dice = float(np.mean(dice_scores))
    mean_iou = float(np.mean(iou_scores))
    print(f"  Mean Dice: {mean_dice:.4f} | Mean IoU: {mean_iou:.4f} "
          f"({len(dice_scores)} samples)")

    # Cleanup
    del sam
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return {
        "mean_dice": round(mean_dice, 4),
        "mean_iou": round(mean_iou, 4),
        "std_dice": round(float(np.std(dice_scores)), 4),
        "num_samples": len(dice_scores),
        "raw_samples": raw_samples
    }


# =====================================================================
# Section 6: MedMNIST External Benchmark
# =====================================================================

def benchmark_medmnist(model, max_samples=100):
    """Evaluate on MedMNIST v2 test splits (6 subsets)."""
    try:
        import medmnist
        from medmnist import INFO
    except ImportError:
        return {"status": "skipped", "reason": "medmnist not installed (pip install medmnist)"}

    SUBSETS = {
        "breastmnist": {"modality": "Ultrasound", "task": "Breast cancer classification"},
        "pneumoniamnist": {"modality": "X-ray", "task": "Pneumonia detection"},
        "dermamnist": {"modality": "Dermatoscopy", "task": "Skin lesion classification"},
        "organamnist":  {"modality": "CT", "task": "Organ classification (axial)"},
        "retinamnist": {"modality": "Fundus", "task": "Retinal disease classification"},
        "bloodmnist": {"modality": "Microscopy", "task": "Blood cell classification"},
    }

    print("\n  Running MedMNIST v2 external benchmark...")
    results = {}

    for subset_name, info in SUBSETS.items():
        try:
            subset_info = INFO[subset_name]
            n_classes = len(subset_info["label"])
            class_names = {int(k): v for k, v in subset_info["label"].items()}

            # Load test dataset
            DataClass = getattr(medmnist, subset_info["python_class"])
            test_dataset = DataClass(split="test", download=True, size=224)

            # Sample from test set
            indices = list(range(len(test_dataset)))
            random.shuffle(indices)
            n_eval = min(max_samples, len(indices))
            indices = indices[:n_eval]

            correct = 0
            total = 0

            for idx in tqdm(indices, desc=f"  {subset_name}", leave=False):
                img, label = test_dataset[idx]

                # Convert to PIL Image if tensor
                if hasattr(img, "numpy"):
                    img_arr = img.numpy()
                    if img_arr.ndim == 3 and img_arr.shape[0] in (1, 3):
                        img_arr = np.transpose(img_arr, (1, 2, 0))
                    if img_arr.max() <= 1.0:
                        img_arr = (img_arr * 255).astype(np.uint8)
                    if img_arr.ndim == 2 or (img_arr.ndim == 3 and img_arr.shape[2] == 1):
                        img_arr = np.squeeze(img_arr)
                        pil_img = Image.fromarray(img_arr).convert("RGB")
                    else:
                        pil_img = Image.fromarray(img_arr).convert("RGB")
                else:
                    pil_img = img.convert("RGB") if isinstance(img, Image.Image) else Image.fromarray(np.array(img)).convert("RGB")

                gt_label = int(label.flatten()[0]) if hasattr(label, "flatten") else int(label)
                gt_name = class_names.get(gt_label, str(gt_label))

                # Build classification prompt
                prompt = (f"This is a {info['modality']} image. "
                          f"Classify this image. The possible classes are: "
                          f"{', '.join(class_names.values())}. "
                          f"Answer with ONLY the class name.")

                try:
                    response, _ = model.analyze_image(pil_img, prompt, task="vqa")
                    resp_lower = response.lower().strip()

                    # Check if GT class name appears in response
                    if gt_name.lower() in resp_lower:
                        correct += 1

                    total += 1
                except Exception:
                    total += 1

            acc = correct / max(total, 1)
            results[subset_name] = {
                "modality": info["modality"],
                "accuracy": round(acc, 4),
                "correct": correct,
                "total": total,
                "n_classes": n_classes
            }
            print(f"    {subset_name}: {acc:.4f} ({correct}/{total}), "
                  f"{n_classes} classes ({info['modality']})")

        except Exception as e:
            results[subset_name] = {"status": "error", "error": str(e)}
            print(f"    {subset_name}: ERROR - {e}")

    # Overall average
    accs = [r["accuracy"] for r in results.values() if "accuracy" in r]
    avg_acc = float(np.mean(accs)) if accs else 0.0
    results["_average"] = round(avg_acc, 4)
    print(f"  MedMNIST Average Accuracy: {avg_acc:.4f} ({len(accs)} subsets)")

    return results


# =====================================================================
# Main Runner
# =====================================================================

def run_blind_benchmark(args):
    """Run all benchmark sections."""
    start_time = time.time()
    report = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": args.checkpoint,
        "data_dir": args.data_dir,
        "max_samples": args.max_samples,
        "sections": {}
    }

    print("=" * 70)
    print("  MEDGAMMA BLIND BENCHMARK")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Data Dir: {args.data_dir}")
    print(f"  Max Samples/Section: {args.max_samples}")
    print("=" * 70)

    # Load model once
    from src.medgemma_wrapper import MedGemmaWrapper
    model = MedGemmaWrapper()
    _base_loaded = False

    def load_with_adapter(stage_name):
        nonlocal _base_loaded
        adapter_map = {
            "screening": "screening",
            "modality": "modality",
            "detection": "final",
            "vqa": "vqa"
        }
        adapter_name = adapter_map.get(stage_name, stage_name)

        if not _base_loaded:
            print("\n  [INIT] Loading base MedGemma model...")
            model.load(use_base_model=True)
            _base_loaded = True
        else:
            # Unload previous adapter
            try:
                if hasattr(model.model, 'disable_adapter_layers'):
                    model.model.disable_adapter_layers()
                if hasattr(model.model, 'delete_adapter'):
                    model.model.delete_adapter('default')
            except Exception:
                pass

        # Load stage adapter
        adapter_path = os.path.join(args.checkpoint, adapter_name)
        if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
            adapter_path = os.path.join(args.checkpoint, "medgemma", adapter_name)

        if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
            print(f"  [ADAPTER] Loading: {adapter_path}")
            try:
                model.model.load_adapter(adapter_path, adapter_name="default")
                model.model.set_adapter("default")
                # Move LoRA weights to GPU
                import torch
                device = getattr(model.model, 'device', torch.device('cuda:0'))
                for _, param in model.model.named_parameters():
                    if param.device.type == 'cpu':
                        param.data = param.data.to(device)
            except Exception as e:
                print(f"  [WARN] Adapter load failed: {e}")
                from peft import PeftModel
                model.model = PeftModel.from_pretrained(model.model, adapter_path)
            model.model.eval()
        else:
            print(f"  [WARN] No adapter found at {adapter_path}, using base model")

    # ── Section 1: Screening ──
    print("\n" + "─" * 60)
    print("  SECTION 1: SCREENING BENCHMARK")
    print("─" * 60)
    load_with_adapter("screening")
    report["sections"]["screening"] = benchmark_screening(
        model, args.data_dir, args.max_samples)

    # ── Section 2: Modality ──
    print("\n" + "─" * 60)
    print("  SECTION 2: MODALITY CLASSIFICATION BENCHMARK")
    print("─" * 60)
    load_with_adapter("modality")
    report["sections"]["modality"] = benchmark_modality(
        model, args.data_dir, args.max_samples)

    # ── Section 3: Detection ──
    print("\n" + "─" * 60)
    print("  SECTION 3: DETECTION BENCHMARK")
    print("─" * 60)
    load_with_adapter("detection")
    report["sections"]["detection"] = benchmark_detection(
        model, args.data_dir, args.max_samples)

    # ── Section 4: VQA ──
    print("\n" + "─" * 60)
    print("  SECTION 4: VQA BENCHMARK")
    print("─" * 60)
    load_with_adapter("vqa")
    report["sections"]["vqa"] = benchmark_vqa(
        model, args.data_dir, args.max_samples)

    # ── Unload MedGemma to free VRAM for SAM2 ──
    print("\n  Freeing VRAM for SAM2...")
    del model
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    # ── Section 5: Segmentation ──
    print("\n" + "─" * 60)
    print("  SECTION 5: SAM2 SEGMENTATION BENCHMARK")
    print("─" * 60)
    report["sections"]["segmentation"] = benchmark_segmentation(
        args.data_dir, args.checkpoint, args.max_samples)

    # Free SAM2 VRAM before reloading MedGemma for MedMNIST
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    # ── Section 6: MedMNIST External ──
    if not args.skip_medmnist:
        print("\n" + "─" * 60)
        print("  SECTION 6: MEDMNIST v2 EXTERNAL BENCHMARK")
        print("─" * 60)
        # Reload model for MedMNIST (use detection adapter as general-purpose)
        from src.medgemma_wrapper import MedGemmaWrapper
        model2 = MedGemmaWrapper()
        model2.load(use_base_model=True)
        model2.model.eval()
        report["sections"]["medmnist"] = benchmark_medmnist(
            model2, max_samples=args.medmnist_samples)
        del model2
        gc.collect()
    else:
        report["sections"]["medmnist"] = {"status": "skipped", "reason": "Disabled via --skip-medmnist"}

    # ── Summary ──
    total_time = time.time() - start_time
    report["total_time_seconds"] = round(total_time, 1)

    print("\n" + "=" * 70)
    print("  BLIND BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"  {'Section':<25} {'Metric':<15} {'Value':<10}")
    print(f"  {'-'*25} {'-'*15} {'-'*10}")

    for section_name, result in report["sections"].items():
        if isinstance(result, dict) and result.get("status") in ("skipped", "error"):
            print(f"  {section_name:<25} {'SKIPPED':<15} {result.get('reason', '')[:30]}")
            continue

        if section_name == "screening":
            print(f"  {section_name:<25} {'Accuracy':<15} {result.get('accuracy', '?')}")
            print(f"  {'':25} {'F1':<15} {result.get('f1', '?')}")
        elif section_name == "modality":
            print(f"  {section_name:<25} {'Accuracy':<15} {result.get('accuracy', '?')}")
        elif section_name == "detection":
            print(f"  {section_name:<25} {'F1':<15} {result.get('f1', '?')}")
            print(f"  {'':25} {'Precision':<15} {result.get('precision', '?')}")
            print(f"  {'':25} {'Recall':<15} {result.get('recall', '?')}")
        elif section_name == "vqa":
            print(f"  {section_name:<25} {'Accuracy':<15} {result.get('accuracy', '?')}")
        elif section_name == "segmentation":
            print(f"  {section_name:<25} {'Dice':<15} {result.get('mean_dice', '?')}")
            print(f"  {'':25} {'IoU':<15} {result.get('mean_iou', '?')}")
        elif section_name == "medmnist":
            avg = result.get("_average", "?")
            print(f"  {section_name:<25} {'Avg Accuracy':<15} {avg}")

    print(f"\n  Total Time: {total_time:.0f}s ({total_time/60:.1f} min)")

    # Save report
    output_dir = "outputs/blind_benchmark"
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "benchmark_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Report: {report_path}")
    print("=" * 70)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedGamma Blind Benchmark")
    parser.add_argument("--checkpoint", default="checkpoints/production",
                        help="Checkpoint directory with stage adapters")
    parser.add_argument("--data-dir", default="medical_data",
                        help="Root data directory")
    parser.add_argument("--max-samples", type=int, default=-1,
                        help="Max samples per section (-1 = all)")
    parser.add_argument("--medmnist-samples", type=int, default=100,
                        help="Max samples per MedMNIST subset")
    parser.add_argument("--skip-medmnist", action="store_true",
                        help="Skip MedMNIST external benchmark")
    parser.add_argument("--sections", nargs="+",
                        choices=["screening", "modality", "detection", "vqa",
                                 "segmentation", "medmnist"],
                        help="Run only specific sections")

    args = parser.parse_args()
    run_blind_benchmark(args)
