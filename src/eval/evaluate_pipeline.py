"""
Holdout Evaluation Pipeline for MedGamma.

Evaluates trained MedGemma adapters on held-out test splits for each training stage.
Supports: screening, modality, detection, vqa.

Automatically loads the correct stage-specific adapter based on --task.

Usage:
    # Detection evaluation (VinDr test set)
    python -m src.eval.evaluate_pipeline --checkpoint checkpoints/production --task detection --max_samples 50

    # VQA evaluation (SLAKE test set)
    python -m src.eval.evaluate_pipeline --checkpoint checkpoints/production --task vqa --dataset slake --max_samples 50

    # Screening evaluation
    python -m src.eval.evaluate_pipeline --checkpoint checkpoints/production --task screening --max_samples 50
"""

import os
import torch
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from peft import PeftModel
from src.medgemma_wrapper import MedGemmaWrapper
from src.data.factory import MedicalDatasetFactory
from src.eval.metrics import compute_iou, compute_text_metrics, ComprehensiveEvaluator
import argparse


# =====================================================================
# Adapter resolution: maps task -> correct stage adapter directory
# =====================================================================
TASK_TO_ADAPTER = {
    "detection": "final",  # User says final has lower loss
    "vqa": "vqa",
    "screening": "screening",
    "modality": "modality",
}

# Default dataset per task
TASK_DEFAULT_DATASET = {
    "detection": "vindr",
    "vqa": "slake",
    "screening": "kaggle_pneumonia",
    "modality": "brain_tumor_multimodal",
}


class PipelineEvaluator:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_raw_samples_logged = 0  # Counter for diagnostic logging
        self.max_raw_logs = args.log_raw_samples  # How many raw outputs to print

        # 1. Load MedGemma base model
        print("Loading MedGemma...")
        self.medgemma = MedGemmaWrapper()
        self.medgemma.load()

        # 2. Load the correct task-specific adapter
        adapter_path = self._resolve_adapter_path(args)
        if adapter_path and os.path.exists(adapter_path):
            print(f"Loading MedGemma Adapter from {adapter_path}")
            self.medgemma.model = PeftModel.from_pretrained(self.medgemma.model, adapter_path)
            self.medgemma.model.eval()
            print(f"  Adapter loaded. Model in eval mode.")
        else:
            print(f"(!) MedGemma Adapter not found at resolved path. Evaluating Base Model.")
            print(f"    Searched: {adapter_path}")

        # Ensure model is in eval mode
        self.medgemma.model.eval()

    def _resolve_adapter_path(self, args):
        """
        Resolve the correct adapter path based on task type.

        Priority order:
        1. Explicit --medgemma_adapter flag (user override)
        2. Task-specific adapter: checkpoint/medgemma/{task}/
        3. Fallback: checkpoint/medgemma/final/
        """
        # Priority 1: Explicit user override
        if args.medgemma_adapter and args.medgemma_adapter != "auto":
            if os.path.exists(args.medgemma_adapter):
                print(f"Using explicit adapter path: {args.medgemma_adapter}")
                return args.medgemma_adapter

        if not args.checkpoint:
            return args.medgemma_adapter  # Original fallback

        # Priority 2: Task-specific adapter
        task_adapter_name = TASK_TO_ADAPTER.get(args.task, args.task)
        
        # Check top-level first
        task_adapter_path = os.path.join(args.checkpoint, task_adapter_name)
        if not os.path.exists(os.path.join(task_adapter_path, "adapter_config.json")):
            # Check legacy medgemma subfolder
            task_adapter_path = os.path.join(args.checkpoint, "medgemma", task_adapter_name)

        if os.path.exists(task_adapter_path) and os.path.exists(os.path.join(task_adapter_path, "adapter_config.json")):
            print(f"Found task-specific adapter for '{args.task}': {task_adapter_path}")
            return task_adapter_path

        # Priority 3: Fallback to 'final'
        fallback_paths = [
            os.path.join(args.checkpoint, "medgemma", "final"),
            os.path.join(args.checkpoint, "final"),
            args.checkpoint
        ]
        for p in fallback_paths:
            if os.path.exists(p) and os.path.exists(os.path.join(p, "adapter_config.json")):
                print(f"(!) Task-specific adapter not found. Falling back to: {p}")
                return p

        return None

    def evaluate(self):
        """Run evaluation for the specified task."""
        factory = MedicalDatasetFactory(base_data_dir=self.args.data_dir)

        task_type = self.args.task

        # Determine dataset and split
        dataset_name = self.args.dataset
        if task_type in ("screening", "modality"):
            # Screening/modality datasets use train split subsets for holdout
            split = "train"
        elif dataset_name == "vindr":
            split = "test"
        else:
            split = "train"

        val_loader = factory.get_loader(dataset_name, task_type, batch_size=1, split=split)

        print(f"Evaluating {dataset_name} ({task_type}) on {len(val_loader.dataset)} images...")

        # Subset if requested
        if self.args.max_samples > 0:
            import random
            total = len(val_loader.dataset)
            all_indices = list(range(total))
            random.seed(42)  # Reproducible
            random.shuffle(all_indices)
            indices = all_indices[:min(total, self.args.max_samples)]
            print(f"Subsetting to {len(indices)} samples (randomized for class diversity).")
            collate_fn = val_loader.collate_fn
            val_loader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(val_loader.dataset, indices),
                batch_size=1,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_fn
            )

        # Initialize metrics
        metrics = {"Total": 0, "Task": task_type, "Dataset": dataset_name}
        if task_type == "detection":
            metrics.update({
                "TP": 0, "FP": 0, "FN": 0,
                "ExactMatch": 0,
                "Sum_IoU": 0.0, "TP_Count_For_IoU": 0,
                "Text_Precision": 0.0, "Text_Recall": 0.0, "Text_F1": 0.0
            })
        elif task_type in ("screening", "modality"):
            metrics.update({
                "Correct": 0, "ExactMatch": 0,
                "Text_Precision": 0.0, "Text_Recall": 0.0, "Text_F1": 0.0,
                "per_class": {}
            })
        else:  # vqa
            metrics.update({
                "Acc": 0, "ExactMatch": 0,
                "Text_Precision": 0.0, "Text_Recall": 0.0, "Text_F1": 0.0
            })

        results_log = []

        for i, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
            image = batch["image"][0]  # PIL Image

            if task_type == "detection":
                self._eval_detection(batch, image, metrics, results_log, i)
            elif task_type == "screening":
                self._eval_screening(batch, image, metrics, results_log, i)
            elif task_type == "modality":
                self._eval_modality(batch, image, metrics, results_log, i)
            else:
                self._eval_vqa(batch, image, metrics, results_log, i)

            if i % 100 == 0:
                self._print_progress(i, metrics)

        self.print_results(metrics, results_log)

    def _log_raw_output(self, sample_idx, raw_text, parsed_boxes=None):
        """Log raw model output for first N samples for diagnostic purposes."""
        if self.num_raw_samples_logged < self.max_raw_logs:
            print(f"\n--- RAW OUTPUT [Sample {sample_idx}] ---")
            print(f"  Text: {raw_text[:300]}{'...' if len(raw_text) > 300 else ''}")
            if parsed_boxes is not None:
                print(f"  Parsed Boxes: {parsed_boxes}")
            print(f"--- END ---\n")
            self.num_raw_samples_logged += 1

    def _eval_screening(self, batch, image, metrics, results_log, sample_idx):
        """Evaluate screening (binary: HEALTHY vs ABNORMAL)."""
        gt_label = batch.get("label", batch.get("class_name", ["unknown"]))[0]

        # Determine GT: is this healthy or abnormal?
        if isinstance(gt_label, str):
            gt_is_healthy = gt_label.lower() in ("normal", "healthy", "no finding")
        else:
            gt_is_healthy = (gt_label == 0)  # Assuming 0 = healthy

        try:
            response, _ = self.medgemma.analyze_image(image, task="screening")
            self._log_raw_output(sample_idx, response)

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

            correct = (pred_is_healthy == gt_is_healthy)
            if correct:
                metrics["Correct"] += 1
                metrics["ExactMatch"] += 1

            # Text metrics
            gt_text = "HEALTHY" if gt_is_healthy else "ABNORMAL"
            pred_text = "HEALTHY" if pred_is_healthy else "ABNORMAL"
            p, r, f1 = compute_text_metrics(pred_text, gt_text)
            metrics["Text_Precision"] += p
            metrics["Text_Recall"] += r
            metrics["Text_F1"] += f1

            metrics["Total"] += 1

            results_log.append({
                "gt_label": str(gt_label),
                "gt_is_healthy": gt_is_healthy,
                "pred_is_healthy": pred_is_healthy,
                "correct": correct,
                "response": response[:200]
            })

        except Exception as e:
            print(f"Error Screening [{sample_idx}]: {e}")
            metrics["Total"] += 1

    def _eval_modality(self, batch, image, metrics, results_log, sample_idx):
        """Evaluate modality detection (X-ray, CT, MRI, Ultrasound)."""
        gt_modality = batch.get("modality", batch.get("class_name", ["unknown"]))[0]

        try:
            response, _ = self.medgemma.analyze_image(image, task="modality")
            self._log_raw_output(sample_idx, response)

            response_lower = response.lower().strip()

            # Parse modality from response
            modality_keywords = {
                "x-ray": "xray", "xray": "xray", "radiograph": "xray",
                "ct": "ct", "computed tomography": "ct",
                "mri": "mri", "magnetic resonance": "mri",
                "ultrasound": "ultrasound", "sonograph": "ultrasound"
            }

            pred_modality = "unknown"
            for keyword, mod in modality_keywords.items():
                if keyword in response_lower:
                    pred_modality = mod
                    break

            gt_mod_normalized = gt_modality.lower().replace("-", "").replace(" ", "")
            pred_mod_normalized = pred_modality.lower().replace("-", "").replace(" ", "")

            correct = (gt_mod_normalized == pred_mod_normalized)
            if correct:
                metrics["Correct"] += 1
                metrics["ExactMatch"] += 1

            # Track per-class
            cls = gt_modality
            if cls not in metrics["per_class"]:
                metrics["per_class"][cls] = {"correct": 0, "total": 0}
            metrics["per_class"][cls]["total"] += 1
            if correct:
                metrics["per_class"][cls]["correct"] += 1

            # Text metrics
            p, r, f1 = compute_text_metrics(response, gt_modality)
            metrics["Text_Precision"] += p
            metrics["Text_Recall"] += r
            metrics["Text_F1"] += f1

            metrics["Total"] += 1

            results_log.append({
                "gt_modality": gt_modality,
                "pred_modality": pred_modality,
                "correct": correct,
                "response": response[:200]
            })

        except Exception as e:
            print(f"Error Modality [{sample_idx}]: {e}")
            metrics["Total"] += 1

    def _eval_vqa(self, batch, image, metrics, results_log, sample_idx):
        """Evaluate VQA task."""
        question = batch['question'][0]
        gt_answer = str(batch['answer'][0])
        prompt = question

        try:
            response, _ = self.medgemma.analyze_image(image, prompt, task="vqa")
            pred_answer = response.replace("Answer:", "").strip()

            self._log_raw_output(sample_idx, response)

            # Normalize answer: strip common prefixes and punctuation
            import string
            pred_normalized = pred_answer.lower().strip()
            for prefix in ["the answer is", "based on the image,", "answer:", "the image shows"]:
                if pred_normalized.startswith(prefix):
                    pred_normalized = pred_normalized[len(prefix):].strip()
            pred_clean = pred_normalized.strip(string.punctuation + " ")
            gt_clean = gt_answer.lower().strip(string.punctuation + " ")

            # Exact, substring, or normalized match
            if (pred_clean == gt_clean or
                gt_clean in pred_clean or
                pred_clean in gt_clean):
                metrics["Acc"] += 1
                metrics["ExactMatch"] += 1

            # Text Metrics
            p, r, f1 = compute_text_metrics(pred_answer, gt_answer)
            metrics["Text_Precision"] += p
            metrics["Text_Recall"] += r
            metrics["Text_F1"] += f1

            metrics["Total"] += 1

            results_log.append({
                "question": question,
                "gt_answer": gt_answer,
                "pred_answer": pred_answer,
                "full_text": response
            })
        except Exception as e:
            print(f"Error VQA [{sample_idx}]: {e}")
            metrics["Total"] += 1

    def _eval_detection(self, batch, image, metrics, results_log, sample_idx):
        """Evaluate detection task with bounding boxes."""
        gt_boxes = batch["boxes"][0]
        gt_labels = batch["labels"][0]

        try:
            generated_text, pred_boxes = self.medgemma.analyze_image(image, task="detection")
            pred_findings = [generated_text]

            self._log_raw_output(sample_idx, generated_text, pred_boxes)

        except Exception as e:
            print(f"Detection Error [{sample_idx}]: {e}")
            pred_boxes = []
            pred_findings = []
            generated_text = ""

        # 1. Text Metrics (Findings)
        gt_text = " ".join(gt_labels) if gt_labels else "No significant abnormalities"
        p, r, f1 = compute_text_metrics(generated_text, gt_text)
        metrics["Text_Precision"] += p
        metrics["Text_Recall"] += r
        metrics["Text_F1"] += f1

        # 2. Box Normalization / Formatting
        # CRITICAL: Both model output and GT boxes are in 0-1000 normalized format
        # (training used 0-1000, GT loader normalizes to 0-1000)
        # Compare directly in 0-1000 space — NO denormalization needed
        
        # Predicted boxes: model outputs coords in same format as training (0-1000 normalized)
        # Values like [100, 190, 130, 220] are 0-1000 normalized, NOT absolute pixels
        norm_pred_boxes = []
        for b in pred_boxes:
            try:
                x1, y1, x2, y2 = b
                # Skip "No significant abnormality" placeholder [0,0,0,0]
                if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
                    continue
                # Clamp to 0-1000 range
                x1 = max(0, min(1000, x1))
                y1 = max(0, min(1000, y1))
                x2 = max(0, min(1000, x2))
                y2 = max(0, min(1000, y2))
                # Skip degenerate boxes
                if x2 > x1 and y2 > y1:
                    norm_pred_boxes.append([x1, y1, x2, y2])
            except:
                continue
        
        # GT boxes: already in 0-1000 from factory loader
        norm_gt_boxes = []
        for b in gt_boxes:
            try:
                x1, y1, x2, y2 = b
                if x2 > x1 and y2 > y1:
                    norm_gt_boxes.append([x1, y1, x2, y2])
            except:
                continue
        gt_boxes_for_eval = norm_gt_boxes

        # 3. Match Boxes
        matched_gt = set()
        for pb in norm_pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            for gt_idx, gb in enumerate(gt_boxes_for_eval):
                iou = compute_iou(pb, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou > 0.3 and best_gt_idx not in matched_gt:
                metrics["TP"] += 1
                matched_gt.add(best_gt_idx)
                metrics["Sum_IoU"] += best_iou
                metrics["TP_Count_For_IoU"] += 1
            else:
                metrics["FP"] += 1

        metrics["FN"] += len(gt_boxes_for_eval) - len(matched_gt)

        # 4. Finding Match (Naive)
        normal_classes = {"no significant abnormality", "no finding", "normal"}
        if gt_labels and pred_findings:
            # Filter out normal class from pred_findings for matching
            real_pred_findings = [f for f in pred_findings if f.lower() not in normal_classes]
            hit = any(p_text.lower() in g.lower() or g.lower() in p_text.lower()
                      for p_text in real_pred_findings for g in gt_labels) if real_pred_findings else False
            if hit:
                metrics["ExactMatch"] += 1
        elif not gt_labels:
            # No GT findings - if model also predicted nothing (or only normal), that's correct
            real_pred_findings = [f for f in (pred_findings or []) if f.lower() not in normal_classes]
            if not real_pred_findings:
                metrics["ExactMatch"] += 1

        metrics["Total"] += 1

        results_log.append({
            "gt_boxes": gt_boxes,
            "gt_labels": gt_labels,
            "pred_boxes": norm_pred_boxes,
            "full_text": generated_text,
            "pred_findings": pred_findings
        })

    def _print_progress(self, i, metrics):
        """Print intermediate progress."""
        total = metrics["Total"] if metrics["Total"] > 0 else 1
        if "TP" in metrics:
            print(f"[{i}] TP: {metrics['TP']} | FP: {metrics['FP']} | FN: {metrics['FN']}")
        elif "Correct" in metrics:
            print(f"[{i}] Accuracy: {metrics['Correct'] / total:.4f}")
        elif "Acc" in metrics:
            print(f"[{i}] Accuracy: {metrics['Acc'] / total:.4f}")

    def print_results(self, metrics, results_log):
        total = metrics["Total"] if metrics["Total"] > 0 else 1
        task = metrics.get("Task", "unknown")
        dataset = metrics.get("Dataset", "unknown")

        print(f"\n{'=' * 50}")
        print(f"  FINAL EVALUATION RESULTS")
        print(f"  Task: {task.upper()} | Dataset: {dataset}")
        print(f"  Samples: {metrics['Total']}")
        print(f"{'=' * 50}")

        if task == "detection" and "TP" in metrics:
            precision = metrics["TP"] / (metrics["TP"] + metrics["FP"] + 1e-6)
            recall = metrics["TP"] / (metrics["TP"] + metrics["FN"] + 1e-6)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
            avg_iou = metrics["Sum_IoU"] / (metrics["TP_Count_For_IoU"] + 1e-6)

            print(f"Precision:      {precision:.4f}")
            print(f"Recall:         {recall:.4f}")
            print(f"F1 Score:       {f1:.4f}")
            print(f"Avg IoU (TP):   {avg_iou:.4f}")
            print(f"Finding Acc:    {metrics['ExactMatch'] / total:.4f}")
            print(f"TP: {metrics['TP']} | FP: {metrics['FP']} | FN: {metrics['FN']}")

        elif task in ("screening", "modality") and "Correct" in metrics:
            acc = metrics["Correct"] / total
            print(f"Accuracy:       {acc:.4f} ({metrics['Correct']}/{total})")
            print(f"Exact Match:    {metrics['ExactMatch'] / total:.4f}")

            # Per-class breakdown for modality
            if metrics.get("per_class"):
                print(f"\nPer-Class Accuracy:")
                for cls, vals in metrics["per_class"].items():
                    cls_acc = vals["correct"] / vals["total"] if vals["total"] > 0 else 0
                    print(f"  {cls}: {cls_acc:.4f} ({vals['correct']}/{vals['total']})")

        elif task == "vqa" and "Acc" in metrics:
            acc = metrics["Acc"] / total
            print(f"Accuracy:       {acc:.4f} ({metrics['Acc']}/{total})")
            print(f"Exact Match:    {metrics['ExactMatch'] / total:.4f}")

        # Text metrics (common to all)
        print("-" * 30)
        print("Clinical Text Metrics:")
        print(f"  BLEU-1 (Pre): {metrics.get('Text_Precision', 0) / total:.4f}")
        print(f"  ROUGE-1 (Rec):{metrics.get('Text_Recall', 0) / total:.4f}")
        print(f"  Text F1:      {metrics.get('Text_F1', 0) / total:.4f}")

        # Save results
        output_dir = os.path.join("outputs", f"eval_{task}_{dataset}")
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
            # Convert non-serializable items
            save_metrics = {k: v for k, v in metrics.items() if k != "per_class"}
            save_metrics["per_class"] = {k: v for k, v in metrics.get("per_class", {}).items()}
            json.dump(save_metrics, f, indent=2)
        print(f"\nSaved metrics to {output_dir}/evaluation_results.json")

        # Save detailed predictions
        try:
            df = pd.DataFrame(results_log)
            csv_path = os.path.join(output_dir, "predictions.csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved {len(df)} predictions to {csv_path}")
        except Exception as e:
            print(f"Failed to save predictions: {e}")

        # Generate Performance Plots
        try:
            self.plot_metrics(metrics, output_dir)
        except Exception as e:
            print(f"Failed to plot metrics: {e}")

    def plot_metrics(self, metrics, output_dir):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if "TP" in metrics:
            labels = ['True Positives', 'False Positives', 'False Negatives']
            values = [metrics['TP'], metrics['FP'], metrics['FN']]

            plt.figure(figsize=(8, 6))
            bars = plt.bar(labels, values, color=['#2ca02c', '#d62728', '#ff7f0e'])
            plt.title(f'Detection Performance (TP / FP / FN)')
            plt.ylabel('Count')

            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                         f'{int(height)}', ha='center', va='bottom')

            plt.savefig(os.path.join(output_dir, "detection_performance.png"))
            plt.close()
            print(f"Saved {output_dir}/detection_performance.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedGamma Holdout Evaluation Pipeline")
    parser.add_argument("--data_dir", type=str, default="medical_data")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["vindr", "slake", "vqa_rad", "kaggle_pneumonia",
                                 "brain_tumor_mri", "brain_tumor_multimodal"],
                        help="Dataset to evaluate on. Defaults based on task.")
    parser.add_argument("--task", type=str, default="detection",
                        choices=["detection", "vqa", "screening", "modality"],
                        help="Task type (determines adapter and evaluation logic)")
    parser.add_argument("--medgemma_adapter", type=str, default="auto",
                        help="Explicit adapter path (overrides automatic resolution)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint directory (auto-finds task-specific adapters)")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Max samples to evaluate. -1 for all.")
    parser.add_argument("--log_raw_samples", type=int, default=5,
                        help="Number of raw model outputs to print for diagnostics")
    args = parser.parse_args()

    # Auto-set dataset based on task if not specified
    if args.dataset is None:
        args.dataset = TASK_DEFAULT_DATASET.get(args.task, "vindr")
        print(f"Auto-selected dataset '{args.dataset}' for task '{args.task}'")

    evaluator = PipelineEvaluator(args)
    evaluator.evaluate()
