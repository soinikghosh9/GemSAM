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
from peft import PeftModel
from typing import Dict, Any, List, Optional, Union, Iterable
import torch
import numpy as np
import string
from datetime import datetime
from tqdm import tqdm
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.medgemma_wrapper import MedGemmaWrapper
from src.data.factory import MedicalDatasetFactory
from src.eval.metrics import (
    compute_iou, compute_text_metrics,
    compute_map_at_iou, compute_froc, compute_fp_per_image,
    bootstrap_f1_ci
)


# =====================================================================
# Stage Definitions
# =====================================================================

STAGE_CONFIGS = [
    {
        "name": "screening",
        "adapter_dir": "screening",
        "task": "screening",
        "dataset": "kaggle_pneumonia",
        "split": "test",  # FIX: Use holdout test set, not training data
        "description": "Binary screening: HEALTHY vs ABNORMAL"
    },
    {
        "name": "modality",
        "adapter_dir": "modality",
        "task": "modality",
        "dataset": "brain_tumor_multimodal",
        "split": "test",  # FIX: Use holdout test set, not training data
        "description": "Imaging modality classification: X-ray, CT, MRI, Ultrasound"
    },
    {
        "name": "detection",
        "adapter_dir": "detection",
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
        "split": "test",  # FIX: Use holdout test set, not training data
        "description": "Visual Question Answering"
    }
]


class MultiStageEvaluator:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.results: Dict[str, Any] = {}  # Store per-stage results
        self._base_wrapper = None  # Cached base model for adapter-swap

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
                stage_result: Dict[str, Any] = self._evaluate_stage(stage_config, adapter_path)
                self.results[stage_name] = stage_result
            except Exception as e:
                print(f"\n[ERROR] Stage '{stage_name}' failed: {e}")
                import traceback
                traceback.print_exc()
                error_result: Dict[str, Any] = {"status": "failed", "error": str(e)}
                self.results[stage_name] = error_result

        # Print and save unified report
        self._print_summary()
        self._save_report()

        # Final cleanup: unload the base model after ALL stages are done
        if self._base_wrapper is not None:
            self._base_wrapper.unload()
            self._base_wrapper = None

    def _load_model_with_adapter(self, adapter_path):
        """Load base MedGemma + stage-specific adapter.
        
        PRODUCTION OPTIMIZATION: Reuses the base model across stages.
        Only the lightweight LoRA adapter (~10MB) is swapped per stage,
        avoiding 4x reloads of the 7GB base model.
        """
        # Reuse base model if already loaded
        if not hasattr(self, '_base_wrapper') or self._base_wrapper is None or self._base_wrapper.model is None:
            print("  [INIT] Loading base MedGemma model (first stage)...")
            self._base_wrapper = MedGemmaWrapper()
            self._base_wrapper.load(use_base_model=True)
        else:
            print("  [CACHE] Reusing base model from previous stage")
            # --- ROBUST ADAPTER SWAP: Cleanly delete previous 'default' adapter ---
            try:
                model = self._base_wrapper.model
                
                # 1. If it's a PeftModel wrapper, unwrap it back to the base model
                # This prevents nesting PeftModel(PeftModel(...)) which causes OOM and warnings
                from peft import PeftModel
                if isinstance(model, PeftModel):
                    print("  [UNWRAP] Resetting PeftModel to base model for fresh adapter load")
                    # unload() is safe, but base_model.model is the direct path back
                    self._base_wrapper.model = model.base_model.model
                    model = self._base_wrapper.model
                
                # 2. Even if it's the base model class, it might have been patched by Transformers-PEFT
                # Attempt to delete 'default' adapter if it exists
                if hasattr(model, 'delete_adapter'):
                    try:
                        model.delete_adapter('default')
                        print("  [CLEANUP] Deleted existing 'default' adapter")
                    except Exception:
                        pass
                
                # 3. Aggressive cleanup of residual PEFT attributes that cause load_adapter to fail
                if hasattr(model, 'peft_config'):
                    try:
                        delattr(model, 'peft_config')
                        print("  [CLEANUP] Removed residual peft_config attribute")
                    except Exception:
                        pass
                
            except Exception as e:
                print(f"  [WARN] Adapter cleanup error: {e}")
        
        wrapper = self._base_wrapper

        if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
            print(f"  Loading adapter: {adapter_path}")
            try:
                # Use standard load_adapter on the (now clean) base model
                wrapper.model.load_adapter(adapter_path, adapter_name="default")
                wrapper.model.set_adapter("default")
            except Exception as e:
                print(f"  (!) Failed to load adapter via load_adapter: {e}")
                print(f"  Attempting PeftModel.from_pretrained fallback...")
                from peft import PeftModel
                # Fallback: manually wrap with PeftModel (will be unwrapped in next stage)
                wrapper.model = PeftModel.from_pretrained(wrapper.model, adapter_path, adapter_name="default")

            # CRITICAL FIX: Ensure LoRA weights are moved to GPU
            try:
                import torch
                device = getattr(wrapper.model, 'device', torch.device('cuda:0'))
                for name, param in wrapper.model.named_parameters():
                    if param.device.type == 'cpu':
                        param.data = param.data.to(device)
            except Exception as e:
                print(f"    Warning during device sync: {e}")
        else:
            print(f"  (!) No adapter_config.json found, using base model")

        wrapper.model.eval()
        # Update evaluator device based on the loaded model's device
        if hasattr(wrapper.model, 'device'):
            self.device = str(wrapper.model.device)
            print(f"  Model successfully loaded on device: {self.device}")
            
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
            # Pass abnormal_only flag (supported by VinDrCXRLoader now)
            loader = factory.get_loader(dataset_name, task, batch_size=1, split=split, 
                                      abnormal_only=getattr(self.args, 'abnormal_only', False))
        except Exception as e:
            print(f"  [ERROR] Stage '{stage_name}' dataset load failed: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "failed", "error": f"Dataset load failed: {e}"}

        total_samples = len(loader.dataset)

        # Subset with balanced ratio if requested for detection
        if task == "detection" and self.args.max_samples > 0:
            indices = self._get_balanced_indices(loader.dataset, self.args.max_samples)
            collate_fn = loader.collate_fn
            loader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(loader.dataset, indices),
                batch_size=1, shuffle=True, # Shuffle for variety if oversampling
                num_workers=0,
                collate_fn=collate_fn
            )
            total_samples = len(indices)
        elif self.args.max_samples > 0 and self.args.max_samples < total_samples:
            all_indices = list(range(total_samples))
            random.seed(42)
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
        final_result: Dict[str, Any] = dict(result)
        final_result["elapsed_seconds"] = round(float(elapsed), 1)
        final_result["samples_evaluated"] = int(total_samples)
        final_result["dataset"] = str(dataset_name)
        final_result["status"] = "completed"

        # NOTE: Do NOT call model.unload() here — it destroys the shared
        # _base_wrapper used across all stages. Adapter-swap is handled by
        # _load_model_with_adapter() which detaches the old LoRA adapter.
        # Only clear the CUDA cache to free intermediate tensors.

        return final_result

    def _get_balanced_indices(self, dataset, max_samples):
        """Get indices with 70% abnormal and 30% normal samples."""
        print(f"  Filtering for balanced sampling (70% abnormal, 30% normal)...")
        
        abnormal_indices = []
        normal_indices = []
        
        # We need to know which are abnormal. VinDrCXRLoader has image_ids and data_map.
        # If it's not a VinDrCXRLoader, we'll have to iterate (slow but safe).
        if hasattr(dataset, 'image_ids') and hasattr(dataset, 'data_map'):
            for i, img_id in enumerate(dataset.image_ids):
                group = dataset.data_map.get_group(img_id)
                if any(row['class_name'] != 'No finding' for _, row in group.iterrows()):
                    abnormal_indices.append(i)
                else:
                    normal_indices.append(i)
        else:
            # Fallback iteration
            for i in range(len(dataset)):
                sample = dataset[i]
                if sample.get("boxes") or sample.get("labels"):
                    abnormal_indices.append(i)
                else:
                    normal_indices.append(i)
        
        print(f"  Dataset has {len(abnormal_indices)} abnormal and {len(normal_indices)} normal samples.")
        
        # Target 70% abnormal
        n_abnormal = int(max_samples * 0.7)
        n_normal = max_samples - n_abnormal
        
        # If we don't have enough, we'll take what we have
        if len(abnormal_indices) < n_abnormal:
            print(f"  Warning: Only {len(abnormal_indices)} abnormal samples available. Using all.")
            selected_abnormal = abnormal_indices
            # Fill the rest with normal if possible
            n_normal = max_samples - len(selected_abnormal)
        else:
            random.seed(42)
            selected_abnormal = random.sample(abnormal_indices, n_abnormal)
            
        if len(normal_indices) < n_normal:
            selected_normal = normal_indices
        else:
            random.seed(42)
            selected_normal = random.sample(normal_indices, n_normal)
            
        print(f"  Selected {len(selected_abnormal)} abnormal and {len(selected_normal)} normal samples.")
        combined = selected_abnormal + selected_normal
        random.shuffle(combined)
        return combined

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
        print(f"  Screening Accuracy: {acc:.4f} ({correct}/{total_count})")

        return {
            "accuracy": round(float(acc), 4),
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

        print(f"  [DONE][DONE][DONE] Modality Accuracy: {acc:.4f}")
        for cls, cls_acc in per_class_acc.items():
            print(f"    {cls}: {cls_acc}")

        return {
            "accuracy": round(float(acc), 4),
            "correct": correct,
            "total": total_count,
            "per_class_accuracy": per_class_acc,
            "raw_samples": raw_samples
        }

    def _eval_detection_stage(self, model, loader, total):
        """Detection with bounding box evaluation + accuracy + confidence metrics."""
        tp, fp, fn = 0, 0, 0
        tn = 0  # True negatives: correctly predicted normal on normal images
        iou_sum, iou_count = 0.0, 0
        class_agnostic_iou_sum, class_agnostic_iou_count = 0.0, 0  # Localization-only metric
        finding_match = 0
        finding_total = 0  # FIX: Track total GT labels for proper finding_match_rate
        total_count = 0
        raw_samples = []

        # Per-image tracking for mAP, FROC, and bootstrap CI
        per_image_predictions = []  # List of dicts: {'boxes', 'scores', 'labels'}
        per_image_ground_truths = []  # List of dicts: {'boxes', 'labels'}
        per_image_tp = []
        per_image_fp = []
        per_image_fn = []

        # --- NEW: Per-image accuracy + confidence tracking ---
        image_correct = 0     # Correctly classified normal/abnormal
        image_total = 0
        confidence_scores = []  # Per-image confidence (0.0 to 1.0)
        per_class_stats = {}   # {class_name: {tp, fp, fn}}

        skipped_normal = 0

        # CLINICALLY CORRECT CLASS MATCHING GROUPS
        # FIX: Separated clinically distinct pathologies that were incorrectly grouped
        OPACITY_GROUP = {"lung opacity", "airspace opacity"}  # Truly equivalent terms only
        CONSOLIDATION_GROUP = {"consolidation", "pneumonia"}  # Related inflammatory process
        ATELECTASIS_GROUP = {"atelectasis", "collapse", "lung collapse"}  # Distinct from opacity
        INFILTRATION_GROUP = {"infiltration", "infiltrate"}  # Distinct radiologic pattern
        PLEURAL_EFFUSION_GROUP = {"pleural effusion", "hydrothorax"}  # FIX: Separate from thickening
        PLEURAL_THICKENING_GROUP = {"pleural thickening", "calcified pleura"}  # FIX: Separate from effusion
        NODULE_GROUP = {"nodule/mass", "nodule", "mass", "lung nodule", "mass/nodule"}
        CALCIFICATION_GROUP = {"calcification", "calcified nodule"}  # FIX: Separate from nodule/mass
        FIBROSIS_GROUP = {"pulmonary fibrosis", "fibrosis", "ild", "interstitial lung disease"}
        CARDIAC_GROUP = {"cardiomegaly", "enlarged heart", "cardiac enlargement"}  # FIX: Removed aortic enlargement
        AORTIC_GROUP = {"aortic enlargement", "aortic dilatation", "prominent aortic knob"}  # FIX: Separate from cardiomegaly
        EMPHYSEMA_GROUP = {"emphysema", "copd", "hyperinflation", "bullae"}
        FRACTURE_GROUP = {"rib fracture", "clavicle fracture", "fracture", "bone fracture"}
        ALL_GROUPS = [
            OPACITY_GROUP, CONSOLIDATION_GROUP, ATELECTASIS_GROUP, INFILTRATION_GROUP,
            PLEURAL_EFFUSION_GROUP, PLEURAL_THICKENING_GROUP, NODULE_GROUP, CALCIFICATION_GROUP,
            FIBROSIS_GROUP, CARDIAC_GROUP, AORTIC_GROUP, EMPHYSEMA_GROUP, FRACTURE_GROUP
        ]

        def _normalize_class(label):
            """Normalize class label for per-class tracking."""
            low = label.lower().strip()
            for group in ALL_GROUPS:
                if low in group:
                    return sorted(group)[0]  # Canonical name = alphabetically first
            return low

        def _compute_image_confidence(gt_boxes, gt_labels, pred_findings, matched_count, img_tp, img_fp, img_fn):
            """Compute a per-image confidence score (0.0 to 1.0).
            
            Components:
              - Classification confidence: Did we get normal/abnormal right?
              - Finding count agreement: |pred| close to |gt|?
              - Match rate: What fraction of GT findings were matched?
              - Reasoning quality: Does the output contain clinical reasoning?
            """
            gt_is_abnormal = len(gt_boxes) > 0
            pred_is_abnormal = len(pred_findings) > 0
            
            # 1. Classification component (40% weight)
            classification_score = 1.0 if (gt_is_abnormal == pred_is_abnormal) else 0.0
            
            # 2. Finding count agreement (20% weight)
            if gt_is_abnormal:
                gt_count = len(gt_labels)
                pred_count = len(pred_findings)
                count_ratio = min(gt_count, pred_count) / max(gt_count, pred_count, 1)
            else:
                count_ratio = 1.0 if not pred_is_abnormal else 0.0
            
            # 3. Match rate (30% weight)
            if gt_is_abnormal and len(gt_labels) > 0:
                match_rate = matched_count / len(gt_labels)
            elif not gt_is_abnormal and not pred_is_abnormal:
                match_rate = 1.0
            else:
                match_rate = 0.0
            
            # 4. Reasoning quality (10% weight) — does the finding have reasoning text?
            reasoning_score = 0.0
            if pred_findings:
                has_reasoning = sum(1 for f in pred_findings 
                                   if isinstance(f, dict) and 
                                   len(str(f.get("reasoning", f.get("r", "")))) > 10)
                reasoning_score = has_reasoning / len(pred_findings)
            elif not gt_is_abnormal:
                reasoning_score = 1.0  # Normal with no findings is fine
            
            # Weighted composite
            confidence = (
                0.40 * classification_score +
                0.20 * count_ratio +
                0.30 * match_rate +
                0.10 * reasoning_score
            )
            return round(confidence, 4)

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
            final_gt_boxes = []
            final_gt_labels = []
            
            if denorm_gt and gt_labels:
                by_label = {}
                for box, lbl in zip(denorm_gt, gt_labels):
                    if lbl not in by_label: by_label[lbl] = []
                    by_label[lbl].append(box)
                
                for lbl, boxes in by_label.items():
                    merged_boxes = []
                    while boxes:
                        current = boxes.pop(0)
                        keep = True
                        for m_idx, m in enumerate(merged_boxes):
                            if compute_iou(current, m) > 0.3:
                                keep = False
                                break
                        if keep:
                            merged_boxes.append(current)
                    
                    for mb in merged_boxes:
                        final_gt_boxes.append(mb)
                        final_gt_labels.append(lbl)
            
            gt_boxes_for_eval = final_gt_boxes
            gt_labels_for_eval = final_gt_labels

            try:
                modality = batch.get("modality", ["xray"])[0]
                generated_text, pred_boxes = model.analyze_image(
                    image, task="detection", modality=modality, include_reasoning=True
                )
            except Exception as e:
                print(f"  Error [{i}]: {e}")
                fn += len(gt_boxes_for_eval)
                total_count += 1
                image_total += 1
                # This image is wrong — 0 confidence
                confidence_scores.append(0.0)
                continue

            # Extract findings with labels
            findings = model._extract_findings(generated_text)
            reasoning_list = model._extract_reasoning(generated_text)
            
            denorm_pred = []
            for f in findings:
                if not isinstance(f, dict): continue
                lbl = f.get("c", f.get("class", "unknown"))
                if lbl.lower() == "no significant abnormality":
                    continue
                    
                b = f.get("b", f.get("box", []))
                if len(b) == 4:
                    try:
                        x1, y1, x2, y2 = b
                        if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
                            continue
                            
                        denorm_pred.append({
                            "box": [
                                round((x1 / 1000) * w), round((y1 / 1000) * h),
                                round((x2 / 1000) * w), round((y2 / 1000) * h)
                            ],
                            "label": lbl
                        })
                    except:
                        continue

            # Clean GT boxes
            rounded_gt = []
            for gob in gt_boxes_for_eval:
                rounded_gt.append([round(c) for c in gob])
            gt_boxes_for_eval = rounded_gt

            # --- PER-IMAGE ACCURACY: Normal/Abnormal classification ---
            gt_is_abnormal = len(gt_boxes_for_eval) > 0
            pred_is_abnormal = len(denorm_pred) > 0
            image_total += 1
            if gt_is_abnormal == pred_is_abnormal:
                image_correct += 1

            # Record raw sample
            raw_samples.append({
                "image_id": batch.get("image_id", ["unknown"])[0],
                "gt_boxes": gt_boxes_for_eval,
                "gt_labels": gt_labels_for_eval,
                "pred_boxes": denorm_pred,
                "reasoning": reasoning_list,
                "findings": findings,
                "raw_text": generated_text,
                "gt_is_abnormal": gt_is_abnormal,
                "pred_is_abnormal": pred_is_abnormal,
            })

            # TRUE NEGATIVE HANDLING
            if not gt_boxes_for_eval and not denorm_pred:
                tn += 1
                total_count += 1
                confidence_scores.append(
                    _compute_image_confidence(gt_boxes_for_eval, gt_labels_for_eval,
                                             findings, 0, 0, 0, 0)
                )
                continue

            # Match boxes and labels
            matched = set()
            img_tp, img_fp = 0, 0
            for pred in denorm_pred:
                pb = pred["box"]
                plbl = pred["label"]
                best_iou, best_idx = 0, -1
                best_class_agnostic_iou = 0
                
                for gi, (gb, glbl) in enumerate(zip(gt_boxes_for_eval, gt_labels_for_eval)):
                    p_low, g_low = plbl.lower(), glbl.lower()
                    
                    is_match = False
                    if p_low in g_low or g_low in p_low:
                        is_match = True
                    else:
                        for group in ALL_GROUPS:
                            if p_low in group and g_low in group:
                                is_match = True
                                break
                    if not is_match and (p_low == "other lesion" or g_low == "other lesion"):
                        is_match = True
                    
                    if is_match:
                        iou = compute_iou(pb, gb)
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = gi
                    
                    ca_iou = compute_iou(pb, gb)
                    if ca_iou > best_class_agnostic_iou:
                        best_class_agnostic_iou = ca_iou

                if best_class_agnostic_iou > 0.2:
                    class_agnostic_iou_sum += best_class_agnostic_iou
                    class_agnostic_iou_count += 1

                pred_class_norm = _normalize_class(plbl)

                if best_iou > 0.50 and best_idx not in matched:  # FIX: mAP@0.5 standard threshold
                    tp += 1
                    img_tp += 1
                    matched.add(best_idx)
                    iou_sum += best_iou
                    iou_count += 1
                    # Per-class TP
                    if pred_class_norm not in per_class_stats:
                        per_class_stats[pred_class_norm] = {"tp": 0, "fp": 0, "fn": 0}
                    per_class_stats[pred_class_norm]["tp"] += 1
                else:
                    fp += 1
                    img_fp += 1
                    # Per-class FP
                    if pred_class_norm not in per_class_stats:
                        per_class_stats[pred_class_norm] = {"tp": 0, "fp": 0, "fn": 0}
                    per_class_stats[pred_class_norm]["fp"] += 1
            
            # Count false negatives (missed ground truth)
            img_fn = len(gt_boxes_for_eval) - len(matched)
            fn += img_fn
            # Per-class FN
            for gi, glbl in enumerate(gt_labels_for_eval):
                if gi not in matched:
                    gt_class_norm = _normalize_class(glbl)
                    if gt_class_norm not in per_class_stats:
                        per_class_stats[gt_class_norm] = {"tp": 0, "fp": 0, "fn": 0}
                    per_class_stats[gt_class_norm]["fn"] += 1

            # Finding text match — FIX: Count per-label, not per-image
            if gt_labels_for_eval:
                for gl in gt_labels_for_eval:
                    if gl.lower() in generated_text.lower():
                        finding_match += 1
                finding_total += len(gt_labels_for_eval)

            # Compute per-image confidence
            conf = _compute_image_confidence(
                gt_boxes_for_eval, gt_labels_for_eval,
                findings, len(matched), img_tp, img_fp, img_fn
            )
            confidence_scores.append(conf)

            # Annotate in raw sample
            raw_samples[-1]["confidence"] = conf
            raw_samples[-1]["image_accuracy"] = 1 if (gt_is_abnormal == pred_is_abnormal) else 0

            # Track per-image data for mAP/FROC/bootstrap
            img_pred_boxes = [f.get("box", f.get("b", [])) for f in findings if isinstance(f, dict)]
            img_pred_labels = [f.get("class", f.get("c", "unknown")) for f in findings if isinstance(f, dict)]
            img_pred_scores = [f.get("confidence", 0.5) for f in findings if isinstance(f, dict)]
            per_image_predictions.append({
                'boxes': img_pred_boxes,
                'labels': img_pred_labels,
                'scores': img_pred_scores
            })
            per_image_ground_truths.append({
                'boxes': list(gt_boxes) if not isinstance(gt_boxes, list) else gt_boxes,
                'labels': list(gt_labels_for_eval) if gt_labels_for_eval else []
            })
            per_image_tp.append(img_tp)
            per_image_fp.append(img_fp)
            per_image_fn.append(img_fn)

            total_count += 1

        # --- Compute aggregate metrics ---
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        avg_iou = iou_sum / max(iou_count, 1)
        avg_ca_iou = class_agnostic_iou_sum / max(class_agnostic_iou_count, 1)

        # NEW METRICS
        image_accuracy = image_correct / max(image_total, 1)
        avg_confidence = sum(confidence_scores) / max(len(confidence_scores), 1)
        
        # Confidence buckets for clinical assessment
        high_conf = sum(1 for c in confidence_scores if c >= 0.7)
        med_conf = sum(1 for c in confidence_scores if 0.4 <= c < 0.7)
        low_conf = sum(1 for c in confidence_scores if c < 0.4)

        # Per-class accuracy
        per_class_accuracy = {}
        for cls, stats in sorted(per_class_stats.items()):
            cls_tp = stats["tp"]
            cls_total = stats["tp"] + stats["fn"]
            cls_precision = cls_tp / max(cls_tp + stats["fp"], 1)
            cls_recall = cls_tp / max(cls_total, 1)
            cls_f1 = 2 * cls_precision * cls_recall / max(cls_precision + cls_recall, 1e-6)
            per_class_accuracy[cls] = {
                "precision": round(cls_precision, 4),
                "recall": round(cls_recall, 4),
                "f1": round(cls_f1, 4),
                "tp": cls_tp, "fp": stats["fp"], "fn": stats["fn"]
            }

        # FIX: Clinical grade based on F1 alone (primary detection metric)
        # No misleading composite that masks low F1 with inflated image_accuracy
        fp_per_image = fp / max(image_total, 1)
        if f1 >= 0.60:
            clinical_grade = "A (Clinical-ready)"
        elif f1 >= 0.40:
            clinical_grade = "B (Acceptable)"
        elif f1 >= 0.20:
            clinical_grade = "C (Needs improvement)"
        else:
            clinical_grade = "D (Not ready)"

        print(f"    FP/image: {fp_per_image:.4f}")
        print(f"    Clinical Grade: {clinical_grade}")
        print(f"    Class-agnostic IoU: {avg_ca_iou:.4f} (n={class_agnostic_iou_count})")
        if per_class_accuracy:
            print(f"    Per-class breakdown:")
            for cls, stats in sorted(per_class_accuracy.items(), key=lambda x: x[1]['f1'], reverse=True):
                print(f"      {cls:.<30} P={stats['precision']:.3f} R={stats['recall']:.3f} F1={stats['f1']:.3f} (TP={stats['tp']} FP={stats['fp']} FN={stats['fn']})")

        # Compute advanced metrics: mAP@0.5, FROC, Bootstrap CIs
        map_result = compute_map_at_iou(per_image_predictions, per_image_ground_truths, iou_threshold=0.50)
        froc_result = compute_froc(per_image_predictions, per_image_ground_truths, iou_threshold=0.50)
        bootstrap_ci = bootstrap_f1_ci(per_image_tp, per_image_fp, per_image_fn, n_bootstrap=1000)

        print(f"    mAP@0.5: {map_result['mAP']:.4f}")
        print(f"    FROC mean sensitivity: {froc_result['mean_sensitivity']:.4f}")
        print(f"    FROC sensitivity @ FP/img: {froc_result['sensitivity_at_fp_rates']}")
        print(f"    Bootstrap 95% CI — F1: [{bootstrap_ci['f1_ci_lower']:.4f}, {bootstrap_ci['f1_ci_upper']:.4f}]")
        print(f"    Bootstrap 95% CI — Precision: [{bootstrap_ci['precision_ci_lower']:.4f}, {bootstrap_ci['precision_ci_upper']:.4f}]")
        print(f"    Bootstrap 95% CI — Recall: [{bootstrap_ci['recall_ci_lower']:.4f}, {bootstrap_ci['recall_ci_upper']:.4f}]")

        return {
            "image_accuracy": round(image_accuracy, 4),
            "avg_confidence": round(avg_confidence, 4),
            "clinical_grade": clinical_grade,
            "confidence_distribution": {
                "high_gte_0.7": high_conf,
                "medium_0.4_0.7": med_conf,
                "low_lt_0.4": low_conf
            },
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "avg_iou": round(avg_iou, 4),
            "class_agnostic_iou": round(avg_ca_iou, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "finding_match_rate": round(finding_match / max(finding_total, 1), 4),
            "fp_per_image": round(fp_per_image, 4),
            "mAP_at_0.5": map_result["mAP"],
            "mAP_per_class": map_result["per_class_AP"],
            "froc": froc_result,
            "bootstrap_95ci": bootstrap_ci,
            "per_class_accuracy": per_class_accuracy,
            "total": total_count,
            "raw_samples": raw_samples
        }

    def _eval_vqa_stage(self, model, loader, total):
        """VQA evaluation with enhanced answer normalization."""
        correct = 0
        total_count = 0
        bleu_sum, rouge_sum = 0.0, 0.0
        raw_samples = []

        # Medical synonym groups for normalization
        SYNONYM_GROUPS = [
            {"yes", "correct", "true", "affirmative", "positive"},
            {"no", "incorrect", "false", "negative", "none"},
            {"normal", "healthy", "no abnormality", "unremarkable", "no finding"},
            {"abnormal", "pathological", "disease", "pathology"},
            {"lung", "pulmonary", "respiratory"},
            {"heart", "cardiac", "cardiovascular"},
            {"brain", "cerebral", "intracranial"},
            {"liver", "hepatic"},
            {"kidney", "renal"},
        ]

        def normalize_answer(text):
            """Normalize answer for robust matching."""
            t = text.lower().strip()
            # Strip common prefixes
            for prefix in ["the answer is", "based on the image,", "answer:",
                           "the image shows", "this image shows", "the finding is",
                           "it shows", "yes,", "no,"]:
                if t.startswith(prefix):
                    t = t[len(prefix):].strip()
            # Strip punctuation
            t = t.strip(string.punctuation + " ")
            return t

        def synonym_match(pred, gt):
            """Check if pred and gt belong to the same synonym group."""
            for group in SYNONYM_GROUPS:
                if pred in group and gt in group:
                    return True
            return False

        def token_overlap(pred, gt):
            """Compute Jaccard word-token overlap."""
            pred_tokens = set(pred.split())
            gt_tokens = set(gt.split())
            if not pred_tokens or not gt_tokens:
                return 0.0
            intersection = pred_tokens & gt_tokens
            union = pred_tokens | gt_tokens
            return len(intersection) / len(union)

        for i, batch in tqdm(enumerate(loader), total=total, desc="VQA"):
            image = batch["image"][0]
            question = batch["question"][0]
            gt_answer = str(batch["answer"][0])

            try:
                response, _ = model.analyze_image(image, question, task="vqa")
                pred = response.strip()

                pred_clean = normalize_answer(pred)
                gt_clean = normalize_answer(gt_answer)

                # Multi-level matching:
                # 1. Exact match
                # 2. Substring match (either direction)
                # 3. Synonym match
                # 4. Token overlap >= 0.5 (Jaccard)
                is_correct = False
                if (pred_clean == gt_clean or
                    gt_clean in pred_clean or
                    pred_clean in gt_clean):
                    is_correct = True
                elif synonym_match(pred_clean, gt_clean):
                    is_correct = True
                elif token_overlap(pred_clean, gt_clean) >= 0.5:
                    is_correct = True

                if is_correct:
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

        print(f"  [DONE] VQA: Accuracy={acc:.4f} BLEU-1={bleu:.4f} ROUGE-1={rouge:.4f}")

        return {
            "accuracy": round(float(acc), 4),
            "bleu1": round(bleu, 4),
            "rouge1": round(rouge, 4),
            "correct": correct,
            "total": total_count,
            "raw_samples": raw_samples
        }

    def _print_summary(self):
        """Print comprehensive summary table."""
        print("\n" + "=" * 80)
        print("  COMPREHENSIVE EVALUATION SUMMARY")
        print("=" * 80)
        print(f"{'Stage':<15} {'Status':<10} {'Accuracy':<12} {'Main Metric':<25} {'Samples':<10} {'Time':<8}")
        print("-" * 80)

        for stage_name, result in self.results.items():
            status = result.get("status", "unknown")

            if status == "completed":
                samples = result.get("samples_evaluated", 0)
                elapsed = result.get("elapsed_seconds", 0)

                # Accuracy column (all stages now have accuracy)
                acc = result.get("accuracy", None)
                acc_str = f"{acc:.4f}" if acc is not None else "N/A"

                # Pick the detailed main metric
                if stage_name == "detection":
                    grade = result.get('clinical_grade', 'N/A')
                    conf = result.get('avg_confidence', 0)
                    metric_str = f"F1:{result.get('f1',0):.3f} Conf:{conf:.3f}"
                elif "bleu1" in result:
                    metric_str = f"BLEU:{result['bleu1']:.3f} R:{result.get('rouge1',0):.3f}"
                elif "per_class_accuracy" in result:
                    metric_str = f"Per-class: {len(result['per_class_accuracy'])} classes"
                else:
                    metric_str = "-"

                print(f"{stage_name:<15} {'OK':<10} {acc_str:<12} {metric_str:<25} {samples:<10} {elapsed:<8.1f}s")
            elif status == "skipped":
                reason = result.get("reason", "")
                print(f"{stage_name:<15} {'SKIP':<10} {'N/A':<12} {reason:<25}")
            else:
                error = result.get("error", "Unknown error")
                print(f"{stage_name:<15} {'FAIL':<10} {'N/A':<12} {error[:25]:<25}")

        # Detection clinical grade
        det_result = self.results.get("detection", {})
        if det_result.get("clinical_grade"):
            print(f"\n  Detection Clinical Grade: {det_result['clinical_grade']}")
            conf_dist = det_result.get('confidence_distribution', {})
            if conf_dist:
                print(f"  Confidence: High={conf_dist.get('high_gte_0.7',0)} Med={conf_dist.get('medium_0.4_0.7',0)} Low={conf_dist.get('low_lt_0.4',0)}")

        print("=" * 80)

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
    parser.add_argument("--abnormal_only", action="store_true",
                        help="Detection only: evaluate strictly on abnormal samples")
    args = parser.parse_args()

    # Filter stages if user specified a subset
    if args.stages:
        filtered = [s for s in STAGE_CONFIGS if s["name"] in args.stages]
        STAGE_CONFIGS.clear()
        STAGE_CONFIGS.extend(filtered)

    evaluator = MultiStageEvaluator(args)
    evaluator.run_all()
