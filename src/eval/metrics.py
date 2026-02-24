"""
Comprehensive evaluation metrics for medical image analysis.
Supports: Classification, Segmentation, Detection, VQA tasks.
"""

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from collections import defaultdict


def compute_iou(boxA, boxB):
    """
    Computes IoU between two bounding boxes [x1, y1, x2, y2].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def compute_mask_iou(mask1, mask2):
    """
    Computes IoU between two binary masks.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return (intersection + 1e-6) / (union + 1e-6)

def compute_dice_score(mask1, mask2):
    """
    Computes Dice Coefficient between two binary masks.
    Dice = 2 * Intersection / (Sum of pixels in both)
    """
    intersection = np.logical_and(mask1, mask2).sum()
    sum_pixels = mask1.sum() + mask2.sum()
    return (2. * intersection + 1e-6) / (sum_pixels + 1e-6)

class ClinicalEvaluator:
    def __init__(self):
        self.metrics = {
            "iou": [],
            "dice": [],
            "detection_acc": []
        }

    def update_segmentation(self, pred_mask, gt_mask):
        iou = compute_mask_iou(pred_mask, gt_mask)
        dice = compute_dice_score(pred_mask, gt_mask)
        self.metrics["iou"].append(iou)
        self.metrics["dice"].append(dice)

    def update_detection(self, pred_box, gt_box, iou_threshold=0.5):
        iou = compute_iou(pred_box, gt_box)
        self.metrics["detection_acc"].append(1 if iou > iou_threshold else 0)

    def summarize(self):
        return {
            "mIoU": np.mean(self.metrics["iou"]) if self.metrics["iou"] else 0.0,
            "Mean Dice": np.mean(self.metrics["dice"]) if self.metrics["dice"] else 0.0,
            "Detection Accuracy": np.mean(self.metrics["detection_acc"]) if self.metrics["detection_acc"] else 0.0
        }

def compute_text_metrics(pred_text, gt_text):
    """
    Computes simple Unigram Precision (BLEU-1 approx), Recall (ROUGE-1 approx), and F1.
    """
    pred_tokens = set(pred_text.lower().split())
    gt_tokens = set(gt_text.lower().split())

    if not pred_tokens:
        return 0.0, 0.0, 0.0

    intersection = len(pred_tokens.intersection(gt_tokens))

    precision = intersection / len(pred_tokens)
    recall = intersection / len(gt_tokens) if gt_tokens else 0.0
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    return precision, recall, f1


# =============================================================================
# ADVANCED SEGMENTATION METRICS
# =============================================================================

def compute_hausdorff_distance(pred_mask, gt_mask, percentile=95):
    """
    Computes Hausdorff Distance (HD95 by default) between two binary masks.
    HD measures the maximum distance from a point in one set to the closest point in the other.
    HD95 uses 95th percentile to reduce sensitivity to outliers.

    Returns: HD95 value (lower is better)
    """
    pred_mask = np.asarray(pred_mask, dtype=bool)
    gt_mask = np.asarray(gt_mask, dtype=bool)

    # Handle empty masks
    if not pred_mask.any() or not gt_mask.any():
        return float('inf')

    # Compute distance transforms (distance from background to nearest foreground)
    pred_dist = distance_transform_edt(~pred_mask)
    gt_dist = distance_transform_edt(~gt_mask)

    # Get distances at boundary points
    pred_surface = pred_dist[gt_mask]
    gt_surface = gt_dist[pred_mask]

    if len(pred_surface) == 0 or len(gt_surface) == 0:
        return float('inf')

    # Compute HD95 (95th percentile of combined distances)
    all_distances = np.concatenate([pred_surface, gt_surface])
    hd = np.percentile(all_distances, percentile)

    return float(hd)


def compute_boundary_f1(pred_mask, gt_mask, tolerance=2):
    """
    Computes Boundary F1 score (BF1) - measures boundary accuracy.
    Useful for evaluating segmentation precision at edges.

    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
        tolerance: Pixel tolerance for boundary matching

    Returns: Boundary F1 score (0-1, higher is better)
    """
    pred_mask = np.asarray(pred_mask, dtype=bool)
    gt_mask = np.asarray(gt_mask, dtype=bool)

    # Extract boundaries using erosion
    from scipy.ndimage import binary_erosion

    pred_boundary = pred_mask ^ binary_erosion(pred_mask)
    gt_boundary = gt_mask ^ binary_erosion(gt_mask)

    if not pred_boundary.any() or not gt_boundary.any():
        return 0.0

    # Compute distance transforms
    pred_dist = distance_transform_edt(~pred_boundary)
    gt_dist = distance_transform_edt(~gt_boundary)

    # Count boundary pixels within tolerance
    pred_correct = np.sum(gt_dist[pred_boundary] <= tolerance)
    gt_correct = np.sum(pred_dist[gt_boundary] <= tolerance)

    precision = pred_correct / np.sum(pred_boundary) if np.sum(pred_boundary) > 0 else 0
    recall = gt_correct / np.sum(gt_boundary) if np.sum(gt_boundary) > 0 else 0

    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return float(f1)


def compute_sensitivity_specificity(pred_mask, gt_mask):
    """
    Computes Sensitivity (True Positive Rate) and Specificity (True Negative Rate).
    Critical for medical imaging where false negatives can be dangerous.

    Returns: (sensitivity, specificity)
    """
    pred_mask = np.asarray(pred_mask, dtype=bool).flatten()
    gt_mask = np.asarray(gt_mask, dtype=bool).flatten()

    tp = np.sum(pred_mask & gt_mask)
    tn = np.sum(~pred_mask & ~gt_mask)
    fp = np.sum(pred_mask & ~gt_mask)
    fn = np.sum(~pred_mask & gt_mask)

    sensitivity = tp / (tp + fn + 1e-6)  # Recall / TPR
    specificity = tn / (tn + fp + 1e-6)  # TNR

    return float(sensitivity), float(specificity)


# =============================================================================
# CLASSIFICATION METRICS
# =============================================================================

class ClassificationMetrics:
    """Accumulates predictions for computing classification metrics."""

    def __init__(self, class_names):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.predictions = []
        self.ground_truths = []
        self.probabilities = []  # For AUC-ROC

    def update(self, pred_class, gt_class, probs=None):
        """Add a single prediction."""
        self.predictions.append(pred_class)
        self.ground_truths.append(gt_class)
        if probs is not None:
            self.probabilities.append(probs)

    def compute_confusion_matrix(self):
        """Compute confusion matrix."""
        cm = np.zeros((self.num_classes, self.num_classes), dtype=int)
        for pred, gt in zip(self.predictions, self.ground_truths):
            if 0 <= pred < self.num_classes and 0 <= gt < self.num_classes:
                cm[gt, pred] += 1
        return cm

    def compute_metrics(self):
        """Compute all classification metrics."""
        cm = self.compute_confusion_matrix()

        # Per-class metrics
        per_class = {}
        for i, name in enumerate(self.class_names):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn

            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)

            per_class[name] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(cm[i, :].sum())
            }

        # Overall metrics
        accuracy = np.trace(cm) / (cm.sum() + 1e-6)

        # Macro-averaged metrics
        macro_precision = np.mean([per_class[n]["precision"] for n in self.class_names])
        macro_recall = np.mean([per_class[n]["recall"] for n in self.class_names])
        macro_f1 = np.mean([per_class[n]["f1"] for n in self.class_names])

        # Weighted-averaged metrics
        supports = [per_class[n]["support"] for n in self.class_names]
        total_support = sum(supports)
        weighted_f1 = sum(per_class[n]["f1"] * per_class[n]["support"] for n in self.class_names) / (total_support + 1e-6)

        return {
            "accuracy": float(accuracy),
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "per_class": per_class,
            "confusion_matrix": cm.tolist(),
            "total_samples": int(cm.sum())
        }


# =============================================================================
# COMPREHENSIVE EVALUATOR FOR BLIND TESTING
# =============================================================================

class ComprehensiveEvaluator:
    """
    Comprehensive evaluator for medical imaging pipeline.
    Supports: Classification, Segmentation, Detection, VQA.
    """

    def __init__(self, class_names=None):
        self.class_names = class_names or []
        self.reset()

    def reset(self):
        """Reset all accumulated metrics."""
        # Segmentation metrics
        self.seg_metrics = {
            "dice": [],
            "iou": [],
            "hd95": [],
            "boundary_f1": [],
            "sensitivity": [],
            "specificity": []
        }

        # Classification metrics
        if self.class_names:
            self.clf_metrics = ClassificationMetrics(self.class_names)
        else:
            self.clf_metrics = None

        # Detection metrics
        self.det_metrics = {
            "tp": 0, "fp": 0, "fn": 0,
            "iou_sum": 0.0, "iou_count": 0
        }

        # VQA metrics
        self.vqa_metrics = {
            "correct": 0, "total": 0,
            "bleu_sum": 0.0, "rouge_sum": 0.0
        }

    def update_segmentation(self, pred_mask, gt_mask):
        """Update segmentation metrics with a single prediction."""
        pred_np = np.asarray(pred_mask) > 0.5
        gt_np = np.asarray(gt_mask) > 0.5

        # Skip if both empty (normal case)
        if not gt_np.any() and not pred_np.any():
            self.seg_metrics["dice"].append(1.0)
            self.seg_metrics["iou"].append(1.0)
            return

        dice = compute_dice_score(pred_np, gt_np)
        iou = compute_mask_iou(pred_np, gt_np)
        hd95 = compute_hausdorff_distance(pred_np, gt_np)
        bf1 = compute_boundary_f1(pred_np, gt_np)
        sens, spec = compute_sensitivity_specificity(pred_np, gt_np)

        self.seg_metrics["dice"].append(dice)
        self.seg_metrics["iou"].append(iou)
        if hd95 != float('inf'):
            self.seg_metrics["hd95"].append(hd95)
        self.seg_metrics["boundary_f1"].append(bf1)
        self.seg_metrics["sensitivity"].append(sens)
        self.seg_metrics["specificity"].append(spec)

    def update_classification(self, pred_class, gt_class, probs=None):
        """Update classification metrics."""
        if self.clf_metrics:
            self.clf_metrics.update(pred_class, gt_class, probs)

    def update_detection(self, pred_boxes, gt_boxes, iou_thresh=0.5):
        """Update detection metrics (box matching)."""
        matched_gt = set()

        for pb in pred_boxes:
            best_iou = 0
            best_idx = -1
            for i, gb in enumerate(gt_boxes):
                iou = compute_iou(pb, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_iou >= iou_thresh and best_idx not in matched_gt:
                self.det_metrics["tp"] += 1
                matched_gt.add(best_idx)
                self.det_metrics["iou_sum"] += best_iou
                self.det_metrics["iou_count"] += 1
            else:
                self.det_metrics["fp"] += 1

        self.det_metrics["fn"] += len(gt_boxes) - len(matched_gt)

    def update_vqa(self, pred_answer, gt_answer):
        """Update VQA metrics."""
        self.vqa_metrics["total"] += 1

        # Exact or partial match
        pred_lower = pred_answer.lower().strip()
        gt_lower = gt_answer.lower().strip()

        if pred_lower == gt_lower or gt_lower in pred_lower or pred_lower in gt_lower:
            self.vqa_metrics["correct"] += 1

        # Text similarity
        p, r, _ = compute_text_metrics(pred_answer, gt_answer)
        self.vqa_metrics["bleu_sum"] += p
        self.vqa_metrics["rouge_sum"] += r

    def compute_all_metrics(self):
        """Compute and return all accumulated metrics."""
        results = {}

        # Segmentation metrics
        if self.seg_metrics["dice"]:
            results["segmentation"] = {
                "dice_mean": float(np.mean(self.seg_metrics["dice"])),
                "dice_std": float(np.std(self.seg_metrics["dice"])),
                "iou_mean": float(np.mean(self.seg_metrics["iou"])),
                "iou_std": float(np.std(self.seg_metrics["iou"])),
                "hd95_mean": float(np.mean(self.seg_metrics["hd95"])) if self.seg_metrics["hd95"] else None,
                "boundary_f1_mean": float(np.mean(self.seg_metrics["boundary_f1"])),
                "sensitivity_mean": float(np.mean(self.seg_metrics["sensitivity"])),
                "specificity_mean": float(np.mean(self.seg_metrics["specificity"])),
                "num_samples": len(self.seg_metrics["dice"])
            }

        # Classification metrics
        if self.clf_metrics and self.clf_metrics.predictions:
            results["classification"] = self.clf_metrics.compute_metrics()

        # Detection metrics
        if self.det_metrics["tp"] + self.det_metrics["fp"] + self.det_metrics["fn"] > 0:
            tp, fp, fn = self.det_metrics["tp"], self.det_metrics["fp"], self.det_metrics["fn"]
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)
            avg_iou = self.det_metrics["iou_sum"] / (self.det_metrics["iou_count"] + 1e-6)

            results["detection"] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "avg_iou": float(avg_iou),
                "tp": tp, "fp": fp, "fn": fn
            }

        # VQA metrics
        if self.vqa_metrics["total"] > 0:
            total = self.vqa_metrics["total"]
            results["vqa"] = {
                "accuracy": self.vqa_metrics["correct"] / total,
                "bleu1_avg": self.vqa_metrics["bleu_sum"] / total,
                "rouge1_avg": self.vqa_metrics["rouge_sum"] / total,
                "total_samples": total
            }

        return results
