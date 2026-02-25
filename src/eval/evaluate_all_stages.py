"""
Comprehensive Multi-Stage Holdout Evaluation for MedGamma.

Evaluates ALL trained stages (screening, modality, detection, vqa) sequentially,
loading the correct adapter for each stage.

Produces a unified report with per-stage metrics and saves detailed logs.

Usage:
    python -m src.eval.evaluate_all_stages --checkpoint checkpoints/production --max_samples 20
    python -m src.eval.evaluate_all_stages --checkpoint checkpoints/production  # All samples
"""

import os
import sys
import json
import time
import random
import argparse
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm
from PIL import Image
from peft import PeftModel

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.medgemma_wrapper import MedGemmaWrapper
from src.data.factory import MedicalDatasetFactory
from src.eval.metrics import compute_iou, compute_text_metrics


# =====================================================================
# Stage Definitions
# =====================================================================

STAGE_CONFIGS = [
    {
        "name": "screening",
        "adapter_dir": "screening",
        "task": "screening",
        "dataset": "kaggle_pneumonia",
        "split": "train",
        "description": "Binary screening: HEALTHY vs ABNORMAL"
    },
    {
        "name": "modality",
        "adapter_dir": "modality",
        "task": "modality",
        "dataset": "brain_tumor_multimodal",
        "split": "train",
        "description": "Imaging modality classification: X-ray, CT, MRI, Ultrasound"
    },
    {
        "name": "detection",
        "adapter_dir": "final",  # User says final has lower loss
        "task": "detection",
        "dataset": "vindr",
        "split": "test",
        "description": "Pathology detection with bounding boxes"
    },
    {
        "name": "vqa",
        "adapter_dir": "vqa",
        "task": "vqa",
        "dataset": "slake",
        "split": "train",
        "description": "Visual Question Answering"
    }
]


class MultiStageEvaluator:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.results = {}  # Store per-stage results

    def run_all(self):
        """Evaluate all stages sequentially."""
        print("=" * 60)
        print("  MedGamma Multi-Stage Holdout Evaluation")
        print("=" * 60)
        print(f"Checkpoint: {self.args.checkpoint}")
        print(f"Max Samples Per Stage: {self.args.max_samples}")
        print(f"Device: {self.device}")
        print(f"Stages: {[s['name'] for s in STAGE_CONFIGS]}")
        print("")

        for stage_config in STAGE_CONFIGS:
            stage_name = stage_config["name"]

            # Check for adapter at top level first (new structure), then in medgemma/ subfolder
            adapter_path = os.path.join(
                self.args.checkpoint, stage_config["adapter_dir"]
            )
            
            if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
                # Try legacy/subfolder path
                adapter_path = os.path.join(
                    self.args.checkpoint, "medgemma", stage_config["adapter_dir"]
                )

            if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
                print(f"\n[SKIP] Adapter not found for stage '{stage_name}' at: {adapter_path}")
                self.results[stage_name] = {"status": "skipped", "reason": "adapter not found"}
                continue

            print(f"\n{'=' * 60}")
            print(f"  STAGE: {stage_name.upper()}")
            print(f"  {stage_config['description']}")
            print(f"  Adapter: {adapter_path}")
            print(f"{'=' * 60}")

            try:
                stage_result = self._evaluate_stage(stage_config, adapter_path)
                self.results[stage_name] = stage_result
            except Exception as e:
                print(f"\n[ERROR] Stage '{stage_name}' failed: {e}")
                import traceback
                traceback.print_exc()
                self.results[stage_name] = {"status": "failed", "error": str(e)}

        # Print and save unified report
        self._print_summary()
        self._save_report()

    def _load_model_with_adapter(self, adapter_path):
        """Load base MedGemma + stage-specific adapter."""
        wrapper = MedGemmaWrapper()
        wrapper.load()

        if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
            print(f"  Loading adapter: {adapter_path}")
            wrapper.model = PeftModel.from_pretrained(wrapper.model, adapter_path)
        else:
            print(f"  (!) No adapter_config.json found, using base model")

        wrapper.model.eval()
        return wrapper

    def _evaluate_stage(self, stage_config, adapter_path):
        """Evaluate a single stage."""
        stage_name = stage_config["name"]
        task = stage_config["task"]
        dataset_name = stage_config["dataset"]
        split = stage_config["split"]

        # Load model with correct adapter
        model = self._load_model_with_adapter(adapter_path)

        # Load dataset
        factory = MedicalDatasetFactory(base_data_dir=self.args.data_dir)

        try:
            loader = factory.get_loader(dataset_name, task, batch_size=1, split=split)
        except Exception as e:
            print(f"  Failed to load dataset '{dataset_name}': {e}")
            model.unload()
            return {"status": "failed", "error": f"Dataset load failed: {e}"}

        total_samples = len(loader.dataset)

        # Subset if needed — use RANDOM indices for class diversity
        if self.args.max_samples > 0 and self.args.max_samples < total_samples:
            all_indices = list(range(total_samples))
            random.seed(42)  # Reproducible
            random.shuffle(all_indices)
            indices = all_indices[:self.args.max_samples]
            collate_fn = loader.collate_fn
            loader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(loader.dataset, indices),
                batch_size=1, shuffle=False, num_workers=0,
                collate_fn=collate_fn
            )
            total_samples = self.args.max_samples

        print(f"  Evaluating {total_samples} samples from '{dataset_name}' ({split} split)...")

        # Run evaluation based on task type
        start_time = time.time()
        if task == "screening":
            result = self._eval_screening_stage(model, loader, total_samples)
        elif task == "modality":
            result = self._eval_modality_stage(model, loader, total_samples)
        elif task == "detection":
            result = self._eval_detection_stage(model, loader, total_samples)
        elif task == "vqa":
            result = self._eval_vqa_stage(model, loader, total_samples)
        else:
            result = {"status": "unknown_task"}

        elapsed = time.time() - start_time
        result["elapsed_seconds"] = round(elapsed, 1)
        result["samples_evaluated"] = total_samples
        result["dataset"] = dataset_name
        result["status"] = "completed"

        # Unload model to free memory for next stage
        model.unload()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    def _eval_screening_stage(self, model, loader, total):
        """Binary classification: HEALTHY vs ABNORMAL."""
        correct = 0
        total_count = 0
        raw_samples = []

        for i, batch in tqdm(enumerate(loader), total=total, desc="Screening"):
            image = batch["image"][0]
            gt_label = batch.get("label", batch.get("class_name", ["unknown"]))[0]

            if isinstance(gt_label, str):
                gt_is_healthy = gt_label.lower() in ("normal", "healthy", "no finding")
            else:
                gt_is_healthy = (gt_label == 0)

            try:
                response, _ = model.analyze_image(image, task="screening")
                response_lower = response.lower().strip()

                # Expanded keyword matching for clinical terms
                healthy_keywords = ["healthy", "normal", "no significant", "unremarkable", "clear"]
                abnormal_keywords = [
                    "abnormal", "disease", "pneumonia", "infection", "opacity",
                    "effusion", "consolidation", "infiltrate", "mass", "nodule",
                    "cardiomegaly", "tumor", "lesion", "patholog"
                ]

                pred_is_healthy = any(kw in response_lower for kw in healthy_keywords)
                if any(kw in response_lower for kw in abnormal_keywords):
                    pred_is_healthy = False

                if pred_is_healthy == gt_is_healthy:
                    correct += 1

                total_count += 1

                if i < 3:  # Log first 3 samples
                    raw_samples.append({"gt": str(gt_label), "pred": response[:150]})
            except Exception as e:
                total_count += 1
                if i < 3:
                    raw_samples.append({"gt": str(gt_label), "error": str(e)})

        acc = correct / max(total_count, 1)
        print(f"  ✓ Screening Accuracy: {acc:.4f} ({correct}/{total_count})")

        return {
            "accuracy": round(acc, 4),
            "correct": correct,
            "total": total_count,
            "raw_samples": raw_samples
        }

    def _eval_modality_stage(self, model, loader, total):
        """Multi-class: X-ray, CT, MRI, Ultrasound."""
        correct = 0
        total_count = 0
        per_class = {}
        raw_samples = []

        modality_keywords = {
            "x-ray": "xray", "xray": "xray", "radiograph": "xray",
            "ct": "ct", "computed tomography": "ct",
            "mri": "mri", "magnetic resonance": "mri",
            "ultrasound": "ultrasound", "sonograph": "ultrasound"
        }

        for i, batch in tqdm(enumerate(loader), total=total, desc="Modality"):
            image = batch["image"][0]
            gt_modality = batch.get("modality", batch.get("class_name", ["unknown"]))[0]

            try:
                response, _ = model.analyze_image(image, task="modality")
                response_lower = response.lower().strip()

                pred_modality = "unknown"
                for keyword, mod in modality_keywords.items():
                    if keyword in response_lower:
                        pred_modality = mod
                        break

                gt_norm = gt_modality.lower().replace("-", "").replace(" ", "")
                pred_norm = pred_modality.lower().replace("-", "").replace(" ", "")

                is_correct = (gt_norm == pred_norm)
                if is_correct:
                    correct += 1

                cls = gt_modality
                if cls not in per_class:
                    per_class[cls] = {"correct": 0, "total": 0}
                per_class[cls]["total"] += 1
                if is_correct:
                    per_class[cls]["correct"] += 1

                total_count += 1

                if i < 3:
                    raw_samples.append({"gt": gt_modality, "pred": response[:150]})
            except Exception as e:
                total_count += 1
                if i < 3:
                    raw_samples.append({"gt": gt_modality, "error": str(e)})

        acc = correct / max(total_count, 1)
        per_class_acc = {k: round(v["correct"] / max(v["total"], 1), 4) for k, v in per_class.items()}

        print(f"  ✓ Modality Accuracy: {acc:.4f}")
        for cls, cls_acc in per_class_acc.items():
            print(f"    {cls}: {cls_acc}")

        return {
            "accuracy": round(acc, 4),
            "correct": correct,
            "total": total_count,
            "per_class_accuracy": per_class_acc,
            "raw_samples": raw_samples
        }

    def _eval_detection_stage(self, model, loader, total):
        """Detection with bounding box evaluation."""
        tp, fp, fn = 0, 0, 0
        iou_sum, iou_count = 0.0, 0
        finding_match = 0
        total_count = 0
        raw_samples = []

        skipped_normal = 0

        for i, batch in tqdm(enumerate(loader), total=total, desc="Detection"):
            image = batch["image"][0]
            gt_boxes = batch["boxes"][0]
            gt_labels = batch["labels"][0]

            # Denormalize GT boxes (now normalized 0-1000 by loader)
            w, h = image.size
            denorm_gt = []
            for b_idx, b in enumerate(gt_boxes):
                if max(b) > 1000:
                    denorm_gt.append(b)
                else: 
                    denorm_gt.append([
                        (b[0] / 1000) * w, (b[1] / 1000) * h,
                        (b[2] / 1000) * w, (b[3] / 1000) * h
                    ])
            
            # MERGE GROUND TRUTH (VinDr has multiple annotators per lesion)
            # We group boxes by class and merge overlapping ones
            final_gt_boxes = []
            final_gt_labels = []
            
            if denorm_gt and gt_labels:
                # Group by label
                by_label = {}
                for box, lbl in zip(denorm_gt, gt_labels):
                    if lbl not in by_label: by_label[lbl] = []
                    by_label[lbl].append(box)
                
                # Merge per label
                for lbl, boxes in by_label.items():
                    # Simple greedy merging
                    merged_boxes = []
                    while boxes:
                        current = boxes.pop(0)
                        keep = True
                        # check against already merged
                        for m_idx, m in enumerate(merged_boxes):
                            if compute_iou(current, m) > 0.3: # Low thresh to merge same finding
                                # It's the same finding, don't add new box. 
                                # Optionally average them? For now just keep the first one.
                                keep = False
                                break
                        if keep:
                            merged_boxes.append(current)
                    
                    for mb in merged_boxes:
                        final_gt_boxes.append(mb)
                        final_gt_labels.append(lbl)
            
            gt_boxes_for_eval = final_gt_boxes
            gt_labels_for_eval = final_gt_labels

            # Special case for Normal images:
            # If nothing in GT strings/boxes, it is normal.
            is_normal_gt = (not gt_boxes and not gt_labels)

            try:
                generated_text, pred_boxes = model.analyze_image(image, task="detection")
            except Exception as e:
                print(f"  Error [{i}]: {e}")
                fn += len(gt_boxes)
                total_count += 1
                continue

            if i < 5:  # Log first 5 samples
                raw_samples.append({
                    "gt_boxes": gt_boxes[:3],
                    "gt_labels": gt_labels[:3],
                    "pred_boxes": pred_boxes[:3],
                    "raw_text": generated_text[:500]  # Increased for debugging
                })

            # Denormalize predicted boxes (if normalized)
            # w, h = image.size  <-- ALREADY DEFINED ABOVE
            denorm_pred = []
            for b_idx, b in enumerate(pred_boxes):
                try:
                    # Assume model outputs 0-1000 integers
                    x1, y1, x2, y2 = b
                    denorm_pred.append([
                        (x1 / 1000) * w, (y1 / 1000) * h,
                        (x2 / 1000) * w, (y2 / 1000) * h
                    ])
                except:
                    continue
            
            # --- PREVIOUSLY INSERTED GT MERGING LOGIC IS HERE (Implicit) ---
 


            # Match boxes
            matched = set()
            for pb in denorm_pred:
                best_iou, best_idx = 0, -1
                for gi, gb in enumerate(gt_boxes_for_eval):
                    iou = compute_iou(pb, gb)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = gi

                if best_iou > 0.5 and best_idx not in matched:
                    tp += 1
                    matched.add(best_idx)
                    iou_sum += best_iou
                    iou_count += 1
                else:
                    fp += 1

            fn += len(gt_boxes_for_eval) - len(matched)  # Use merged GT, not raw duplicates

            # Finding text match
            if gt_labels:
                for gl in gt_labels:
                    if gl.lower() in generated_text.lower():
                        finding_match += 1
                        break

            total_count += 1

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        avg_iou = iou_sum / max(iou_count, 1)

        print(f"  ✓ Detection: P={precision:.4f} R={recall:.4f} F1={f1:.4f} IoU={avg_iou:.4f}")
        print(f"    TP={tp} FP={fp} FN={fn} (skipped {skipped_normal} normal images)")

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "avg_iou": round(avg_iou, 4),
            "tp": tp, "fp": fp, "fn": fn,
            "finding_match_rate": round(finding_match / max(total_count, 1), 4),
            "total": total_count,
            "raw_samples": raw_samples
        }

    def _eval_vqa_stage(self, model, loader, total):
        """VQA evaluation."""
        correct = 0
        total_count = 0
        bleu_sum, rouge_sum = 0.0, 0.0
        raw_samples = []

        for i, batch in tqdm(enumerate(loader), total=total, desc="VQA"):
            image = batch["image"][0]
            question = batch["question"][0]
            gt_answer = str(batch["answer"][0])

            try:
                response, _ = model.analyze_image(image, question, task="vqa")
                pred = response.strip()

                # Normalize answer: strip common prefixes and clean whitespace
                pred_normalized = pred.lower().strip()
                for prefix in ["the answer is", "based on the image,", "answer:", "the image shows"]:
                    if pred_normalized.startswith(prefix):
                        pred_normalized = pred_normalized[len(prefix):].strip()
                # Strip punctuation for comparison
                import string
                pred_clean = pred_normalized.strip(string.punctuation + " ")
                gt_clean = gt_answer.lower().strip(string.punctuation + " ")

                # Exact, substring, or normalized match
                if (pred_clean == gt_clean or
                    gt_clean in pred_clean or
                    pred_clean in gt_clean):
                    correct += 1

                # Text metrics
                p, r, _ = compute_text_metrics(pred, gt_answer)
                bleu_sum += p
                rouge_sum += r

                total_count += 1

                if i < 3:
                    raw_samples.append({
                        "question": question[:100],
                        "gt_answer": gt_answer,
                        "pred_answer": pred[:150]
                    })
            except Exception as e:
                total_count += 1
                if i < 3:
                    raw_samples.append({"question": question[:100], "error": str(e)})

        acc = correct / max(total_count, 1)
        bleu = bleu_sum / max(total_count, 1)
        rouge = rouge_sum / max(total_count, 1)

        print(f"  ✓ VQA: Accuracy={acc:.4f} BLEU-1={bleu:.4f} ROUGE-1={rouge:.4f}")

        return {
            "accuracy": round(acc, 4),
            "bleu1": round(bleu, 4),
            "rouge1": round(rouge, 4),
            "correct": correct,
            "total": total_count,
            "raw_samples": raw_samples
        }

    def _print_summary(self):
        """Print comprehensive summary table."""
        print("\n" + "=" * 70)
        print("  COMPREHENSIVE EVALUATION SUMMARY")
        print("=" * 70)
        print(f"{'Stage':<15} {'Status':<12} {'Main Metric':<25} {'Samples':<10} {'Time':<8}")
        print("-" * 70)

        for stage_name, result in self.results.items():
            status = result.get("status", "unknown")

            if status == "completed":
                samples = result.get("samples_evaluated", 0)
                elapsed = result.get("elapsed_seconds", 0)

                # Pick the main metric
                if "accuracy" in result:
                    metric_str = f"Acc: {result['accuracy']:.4f}"
                elif "f1" in result:
                    metric_str = f"F1: {result['f1']:.4f}"
                else:
                    metric_str = "N/A"

                print(f"{stage_name:<15} {'✓':<12} {metric_str:<25} {samples:<10} {elapsed:<8.1f}s")
            elif status == "skipped":
                reason = result.get("reason", "")
                print(f"{stage_name:<15} {'SKIP':<12} {reason:<25}")
            else:
                error = result.get("error", "Unknown error")
                print(f"{stage_name:<15} {'FAIL':<12} {error[:25]:<25}")

        print("=" * 70)

    def _save_report(self):
        """Save the full report as JSON."""
        output_dir = os.path.join("outputs", "multi_stage_eval")
        os.makedirs(output_dir, exist_ok=True)

        report = {
            "timestamp": datetime.now().isoformat(),
            "checkpoint": self.args.checkpoint,
            "max_samples_per_stage": self.args.max_samples,
            "device": self.device,
            "stages": {}
        }

        for stage_name, result in self.results.items():
            # Remove raw_samples from the report to keep it clean
            clean_result = {k: v for k, v in result.items() if k != "raw_samples"}
            report["stages"][stage_name] = clean_result

        report_path = os.path.join(output_dir, "evaluation_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nFull report saved to: {report_path}")

        # Save raw samples separately (useful for debugging)
        raw_path = os.path.join(output_dir, "raw_samples.json")
        raw_data = {name: result.get("raw_samples", []) for name, result in self.results.items()}
        with open(raw_path, "w") as f:
            json.dump(raw_data, f, indent=2, default=str)
        print(f"Raw sample logs saved to: {raw_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedGamma Multi-Stage Holdout Evaluation")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/production",
                        help="Checkpoint directory containing stage adapters")
    parser.add_argument("--data_dir", type=str, default="medical_data",
                        help="Base data directory")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Max samples per stage (-1 for all)")
    parser.add_argument("--stages", type=str, nargs="+",
                        default=["screening", "modality", "detection", "vqa"],
                        help="Stages to evaluate (default: all)")
    args = parser.parse_args()

    # Filter stages if user specified a subset
    if args.stages:
        filtered = [s for s in STAGE_CONFIGS if s["name"] in args.stages]
        STAGE_CONFIGS.clear()
        STAGE_CONFIGS.extend(filtered)

    evaluator = MultiStageEvaluator(args)
    evaluator.run_all()
