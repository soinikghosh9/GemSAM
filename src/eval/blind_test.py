"""
Comprehensive Blind Testing Evaluation for MedGamma Pipeline.

Tests trained models on completely unseen datasets (BUSI - Breast Ultrasound).
Evaluates:
1. Classification: benign/malignant/normal
2. Segmentation: Lesion boundary precision with SAM2
3. Full pipeline: MedGemma detection -> SAM2 segmentation

Usage:
    python -m src.eval.blind_test --checkpoint checkpoints/production
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.factory import MedicalDatasetFactory, BUSIDataset
from src.eval.metrics import (
    ComprehensiveEvaluator,
    compute_dice_score,
    compute_mask_iou,
    compute_hausdorff_distance,
    compute_boundary_f1
)


class BlindTestEvaluator:
    """
    Comprehensive blind testing evaluator.
    Tests MedGemma + SAM2 on unseen BUSI dataset.
    """

    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.results_dir = os.path.join(args.output_dir, "blind_test_results")
        os.makedirs(self.results_dir, exist_ok=True)

        print("=" * 60)
        print("MEDGAMMA BLIND TESTING EVALUATION")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Output: {self.results_dir}")
        print()

        # Initialize evaluator
        self.evaluator = ComprehensiveEvaluator(
            class_names=["normal", "benign", "malignant"]
        )

        # Load models
        self._load_models()

    def _load_models(self):
        """Load trained MedGemma and SAM2 models."""
        print("Loading models...")

        # Load MedGemma
        try:
            from src.medgemma_wrapper import MedGemmaWrapper
            from peft import PeftModel

            self.medgemma = MedGemmaWrapper()
            self.medgemma.load()

            # Load adapter if available
            adapter_paths = [
                os.path.join(self.args.checkpoint, "medgemma", "vqa"),
                os.path.join(self.args.checkpoint, "medgemma", "detection"),
                os.path.join(self.args.checkpoint, "vqa"),
            ]

            adapter_loaded = False
            for adapter_path in adapter_paths:
                if os.path.exists(adapter_path) and os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
                    print(f"  Loading MedGemma adapter from: {adapter_path}")
                    self.medgemma.model = PeftModel.from_pretrained(
                        self.medgemma.model, adapter_path
                    )
                    adapter_loaded = True
                    break

            if not adapter_loaded:
                print("  (!) No MedGemma adapter found, using base model")

            self.medgemma.model.eval()
            print("  MedGemma loaded successfully")

        except Exception as e:
            print(f"  Error loading MedGemma: {e}")
            self.medgemma = None

        # Load SAM2 - replicate training's loading approach
        try:
            sam2_checkpoint = os.path.join(self.args.checkpoint, "sam2", "best")
            if not os.path.exists(sam2_checkpoint):
                sam2_checkpoint = os.path.join(self.args.checkpoint, "sam2", "final")

            if os.path.exists(sam2_checkpoint):
                from omegaconf import OmegaConf
                from hydra.utils import instantiate
                from peft import PeftModel as SAM2PeftModel

                print("  Loading SAM2 base model...")

                # Load config and instantiate model (same as training)
                config_path = "sam2_hiera_t.yaml"
                checkpoint_path = "checkpoints/sam2_hiera_tiny.pt"

                cfg = OmegaConf.load(config_path)
                self.sam2_model = instantiate(cfg.model, _recursive_=True)

                # Load weights with strict=False (same as training)
                state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                if "model" in state_dict:
                    state_dict = state_dict["model"]

                missing, unexpected = self.sam2_model.load_state_dict(state_dict, strict=False)
                print(f"    Base model loaded (missing: {len(missing)}, unexpected: {len(unexpected)})")

                self.sam2_model.to(self.device)

                # Load trained LoRA adapters
                adapter_config_path = os.path.join(sam2_checkpoint, "adapter_config.json")
                if os.path.exists(adapter_config_path):
                    print(f"  Loading SAM2 LoRA adapters from: {sam2_checkpoint}")
                    self.sam2_model.image_encoder = SAM2PeftModel.from_pretrained(
                        self.sam2_model.image_encoder,
                        sam2_checkpoint
                    )

                self.sam2_model.eval()
                print(f"  SAM2 loaded successfully")
            else:
                print("  (!) SAM2 checkpoint not found")
                self.sam2_model = None

        except Exception as e:
            print(f"  Error loading SAM2: {e}")
            import traceback
            traceback.print_exc()
            self.sam2_model = None

        print()

    def run_classification_test(self, dataset):
        """
        Test MedGemma classification on BUSI dataset.
        Task: Classify breast ultrasound as normal/benign/malignant.
        """
        print("\n" + "=" * 50)
        print("CLASSIFICATION TEST (MedGemma)")
        print("=" * 50)

        if self.medgemma is None:
            print("Skipping: MedGemma not loaded")
            return {}

        class_mapping = {
            "normal": 0, "benign": 1, "malignant": 2,
            "healthy": 0, "abnormal": 1,  # Alternate mappings
        }

        results_log = []
        correct = 0
        total = 0

        # Classification prompt
        prompt = (
            "This is a breast ultrasound image. "
            "Classify this image as one of: normal, benign, or malignant. "
            "Respond with only the classification label."
        )

        for i, sample in enumerate(tqdm(dataset, desc="Classification")):
            if self.args.max_samples > 0 and i >= self.args.max_samples:
                break

            image = sample["image"]
            gt_class = sample["class_idx"]
            gt_name = sample["class_name"]

            try:
                # Get prediction - use task="vqa" to avoid detection JSON format
                response, _ = self.medgemma.analyze_image(image, prompt, task="vqa")
                pred_text = response.lower().strip()

                # Parse prediction
                pred_class = -1
                for key, idx in class_mapping.items():
                    if key in pred_text:
                        pred_class = idx
                        break

                # If no match, try to infer from response
                if pred_class == -1:
                    if "cancer" in pred_text or "tumor" in pred_text:
                        pred_class = 2  # malignant
                    elif "mass" in pred_text or "lesion" in pred_text:
                        pred_class = 1  # benign
                    else:
                        pred_class = 0  # normal

                # Update metrics
                self.evaluator.update_classification(pred_class, gt_class)

                if pred_class == gt_class:
                    correct += 1
                total += 1

                results_log.append({
                    "image": sample["image_path"],
                    "gt_class": gt_name,
                    "pred_class": ["normal", "benign", "malignant"][pred_class] if pred_class >= 0 else "unknown",
                    "response": response[:200],
                    "correct": pred_class == gt_class
                })

            except Exception as e:
                print(f"  Error on sample {i}: {e}")
                continue

        # Compute metrics
        acc = correct / total if total > 0 else 0
        print(f"\nClassification Accuracy: {acc:.4f} ({correct}/{total})")

        # Save detailed results
        with open(os.path.join(self.results_dir, "classification_predictions.json"), "w") as f:
            json.dump(results_log, f, indent=2)

        return {"accuracy": acc, "correct": correct, "total": total}

    def run_segmentation_test(self, dataset):
        """
        Test SAM2 segmentation on BUSI dataset.
        Uses ground truth bounding boxes as prompts.
        """
        print("\n" + "=" * 50)
        print("SEGMENTATION TEST (SAM2)")
        print("=" * 50)

        if self.sam2_model is None:
            print("Skipping: SAM2 not loaded")
            return {}

        results_log = []
        seg_evaluator = ComprehensiveEvaluator()

        for i, sample in enumerate(tqdm(dataset, desc="Segmentation")):
            if self.args.max_samples > 0 and i >= self.args.max_samples:
                break

            # Skip normal samples (no lesion to segment)
            if sample["class_name"] == "normal":
                continue

            image = sample["image"]
            gt_mask = sample["mask"]

            if gt_mask is None or not gt_mask.any():
                continue

            try:
                # Convert image to tensor with SAM2 normalization
                if isinstance(image, Image.Image):
                    image_resized = image.resize((1024, 1024))
                    img_np = np.array(image_resized).astype(np.float32)
                else:
                    img_np = np.array(image).astype(np.float32)

                # SAM2 normalization: ImageNet mean/std
                mean = np.array([123.675, 116.28, 103.53])
                std = np.array([58.395, 57.12, 57.375])
                img_normalized = (img_np - mean) / std

                # Convert to tensor [B, C, H, W]
                img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).float()
                img_tensor = img_tensor.unsqueeze(0).to(self.device)

                # Get bounding box from GT mask for prompt
                gt_np = gt_mask.squeeze().numpy()
                if gt_np.shape != (1024, 1024):
                    from PIL import Image as PILImage
                    gt_pil = PILImage.fromarray((gt_np * 255).astype(np.uint8))
                    gt_pil = gt_pil.resize((1024, 1024), resample=PILImage.NEAREST)
                    gt_np = np.array(gt_pil) / 255.0

                rows, cols = np.where(gt_np > 0.5)
                if len(rows) == 0:
                    continue

                y1, x1 = rows.min(), cols.min()
                y2, x2 = rows.max(), cols.max()
                box = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32, device=self.device)

                # Run SAM2 inference (matching training code approach)
                with torch.no_grad():
                    # Forward pass through image encoder
                    backbone_out = self.sam2_model.forward_image(img_tensor)
                    _, vision_feats, _, feat_sizes = self.sam2_model._prepare_backbone_features(backbone_out)

                    B = img_tensor.shape[0]

                    # Prepare high-res features (same as training code)
                    high_res_features = None
                    if self.sam2_model.use_high_res_features_in_sam and len(vision_feats) > 1:
                        high_res_features = [
                            f.permute(1, 2, 0).view(B, -1, *fs)
                            for f, fs in zip(vision_feats[:-1], feat_sizes[:-1])
                        ]

                    # Get image embeddings from last feature
                    image_embeddings = vision_feats[-1].permute(1, 2, 0).view(B, -1, *feat_sizes[-1])

                    # Prepare prompts (box only)
                    sparse_embeddings, dense_embeddings = self.sam2_model.sam_prompt_encoder(
                        points=None,
                        boxes=box,
                        masks=None
                    )

                    # Decode mask
                    low_res_masks, _, _, _ = self.sam2_model.sam_mask_decoder(
                        image_embeddings=image_embeddings,
                        image_pe=self.sam2_model.sam_prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                        repeat_image=False,
                        high_res_features=high_res_features
                    )

                    # Upsample and threshold
                    pred_mask = torch.nn.functional.interpolate(
                        low_res_masks,
                        size=(1024, 1024),
                        mode="bilinear",
                        align_corners=False
                    )
                    pred_mask = (torch.sigmoid(pred_mask) > 0.5).float().squeeze().cpu().numpy()

                # Compute metrics
                gt_binary = (gt_np > 0.5).astype(np.float32)
                dice = compute_dice_score(pred_mask, gt_binary)
                iou = compute_mask_iou(pred_mask, gt_binary)
                hd95 = compute_hausdorff_distance(pred_mask, gt_binary)
                bf1 = compute_boundary_f1(pred_mask, gt_binary)

                seg_evaluator.update_segmentation(pred_mask, gt_binary)

                results_log.append({
                    "image": sample["image_path"],
                    "class": sample["class_name"],
                    "dice": float(dice),
                    "iou": float(iou),
                    "hd95": float(hd95) if hd95 != float('inf') else None,
                    "boundary_f1": float(bf1)
                })

                # Save sample visualizations periodically
                if i < 10:
                    self._save_segmentation_viz(
                        sample["image"], gt_binary, pred_mask,
                        os.path.join(self.results_dir, f"seg_sample_{i}.png"),
                        dice, iou
                    )

            except Exception as e:
                print(f"  Error on sample {i}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Compute final metrics
        metrics = seg_evaluator.compute_all_metrics()

        if "segmentation" in metrics:
            print(f"\nSegmentation Results:")
            print(f"  Dice:        {metrics['segmentation']['dice_mean']:.4f} +/- {metrics['segmentation']['dice_std']:.4f}")
            print(f"  IoU:         {metrics['segmentation']['iou_mean']:.4f}")
            print(f"  HD95:        {metrics['segmentation']['hd95_mean']:.2f} px" if metrics['segmentation']['hd95_mean'] else "  HD95:        N/A")
            print(f"  Boundary F1: {metrics['segmentation']['boundary_f1_mean']:.4f}")
            print(f"  Sensitivity: {metrics['segmentation']['sensitivity_mean']:.4f}")
            print(f"  Specificity: {metrics['segmentation']['specificity_mean']:.4f}")

        # Save results
        with open(os.path.join(self.results_dir, "segmentation_predictions.json"), "w") as f:
            json.dump(results_log, f, indent=2)

        return metrics.get("segmentation", {})

    def run_full_pipeline_test(self, dataset):
        """
        Test full pipeline: MedGemma detection -> SAM2 segmentation.
        Simulates real clinical workflow.
        """
        print("\n" + "=" * 50)
        print("FULL PIPELINE TEST (MedGemma + SAM2)")
        print("=" * 50)

        if self.medgemma is None or self.sam2_model is None:
            print("Skipping: Models not fully loaded")
            return {}

        results_log = []
        detection_correct = 0
        segmentation_scores = []

        detection_prompt = (
            "Analyze this breast ultrasound image for any masses or lesions. "
            "If you detect any findings, provide the bounding box coordinates as [x1,y1,x2,y2]. "
            "If normal, respond with 'No lesions detected'."
        )

        for i, sample in enumerate(tqdm(dataset, desc="Full Pipeline")):
            if self.args.max_samples > 0 and i >= self.args.max_samples:
                break

            image = sample["image"]
            gt_class = sample["class_name"]
            gt_mask = sample["mask"]

            try:
                # Step 1: MedGemma detection
                response, pred_boxes = self.medgemma.analyze_image(image, detection_prompt)

                # Check detection correctness
                has_lesion_gt = gt_class != "normal"
                has_lesion_pred = len(pred_boxes) > 0 or "mass" in response.lower() or "lesion" in response.lower()

                if has_lesion_gt == has_lesion_pred:
                    detection_correct += 1

                # Step 2: SAM2 segmentation (if lesion detected and GT has mask)
                dice = None
                if has_lesion_pred and gt_mask is not None and gt_mask.any():
                    # Use predicted box or GT box as fallback
                    if pred_boxes:
                        # Denormalize box (assuming 1000-scale)
                        if isinstance(image, Image.Image):
                            w, h = image.size
                        else:
                            h, w = image.shape[-2:]
                        box = pred_boxes[0]
                        x1 = int(box[0] * w / 1000)
                        y1 = int(box[1] * h / 1000)
                        x2 = int(box[2] * w / 1000)
                        y2 = int(box[3] * h / 1000)
                    else:
                        # Use GT box
                        gt_np = gt_mask.squeeze().numpy()
                        rows, cols = np.where(gt_np > 0.5)
                        if len(rows) > 0:
                            y1, x1 = rows.min(), cols.min()
                            y2, x2 = rows.max(), cols.max()
                        else:
                            continue

                    # Run SAM2 segmentation with detected/GT box
                    if self.sam2_model is not None:
                        try:
                            # Prepare image for SAM2
                            if isinstance(image, Image.Image):
                                image_resized = image.resize((1024, 1024))
                                img_np = np.array(image_resized).astype(np.float32)
                            else:
                                img_np = np.array(image).astype(np.float32)

                            # SAM2 normalization
                            mean = np.array([123.675, 116.28, 103.53])
                            std = np.array([58.395, 57.12, 57.375])
                            img_normalized = (img_np - mean) / std

                            img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).float()
                            img_tensor = img_tensor.unsqueeze(0).to(self.device)

                            # Scale box coordinates to 1024x1024
                            if isinstance(image, Image.Image):
                                orig_w, orig_h = image.size
                            else:
                                orig_h, orig_w = image.shape[-2:]
                            scale_x = 1024.0 / orig_w
                            scale_y = 1024.0 / orig_h
                            box_1024 = torch.tensor(
                                [[x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]],
                                dtype=torch.float32, device=self.device
                            )

                            # Run SAM2 inference
                            with torch.no_grad():
                                backbone_out = self.sam2_model.forward_image(img_tensor)
                                _, vision_feats, _, feat_sizes = self.sam2_model._prepare_backbone_features(backbone_out)

                                B = img_tensor.shape[0]
                                high_res_features = None
                                if self.sam2_model.use_high_res_features_in_sam and len(vision_feats) > 1:
                                    high_res_features = [
                                        f.permute(1, 2, 0).view(B, -1, *fs)
                                        for f, fs in zip(vision_feats[:-1], feat_sizes[:-1])
                                    ]

                                image_embeddings = vision_feats[-1].permute(1, 2, 0).view(B, -1, *feat_sizes[-1])

                                sparse_embeddings, dense_embeddings = self.sam2_model.sam_prompt_encoder(
                                    points=None, boxes=box_1024, masks=None
                                )

                                low_res_masks, _, _, _ = self.sam2_model.sam_mask_decoder(
                                    image_embeddings=image_embeddings,
                                    image_pe=self.sam2_model.sam_prompt_encoder.get_dense_pe(),
                                    sparse_prompt_embeddings=sparse_embeddings,
                                    dense_prompt_embeddings=dense_embeddings,
                                    multimask_output=False,
                                    repeat_image=False,
                                    high_res_features=high_res_features
                                )

                                pred_mask = torch.nn.functional.interpolate(
                                    low_res_masks, size=(1024, 1024),
                                    mode="bilinear", align_corners=False
                                )
                                pred_mask = (torch.sigmoid(pred_mask) > 0.5).float().squeeze().cpu().numpy()

                            # Compute Dice against GT mask (resized to 1024x1024)
                            gt_np = gt_mask.squeeze().numpy()
                            if gt_np.shape != (1024, 1024):
                                gt_pil = Image.fromarray((gt_np * 255).astype(np.uint8))
                                gt_pil = gt_pil.resize((1024, 1024), resample=Image.NEAREST)
                                gt_np = np.array(gt_pil) / 255.0

                            gt_binary = (gt_np > 0.5).astype(np.float32)
                            dice = compute_dice_score(pred_mask, gt_binary)
                        except Exception as seg_err:
                            print(f"  SAM2 error on sample {i}: {seg_err}")
                            dice = 0.0
                    else:
                        dice = 0.0  # SAM2 not loaded

                results_log.append({
                    "image": sample["image_path"],
                    "gt_class": gt_class,
                    "detection_correct": has_lesion_gt == has_lesion_pred,
                    "boxes_detected": len(pred_boxes),
                    "dice": dice
                })

            except Exception as e:
                print(f"  Error on sample {i}: {e}")
                continue

        # Compute metrics
        total = len(results_log)
        det_acc = detection_correct / total if total > 0 else 0

        print(f"\nFull Pipeline Results:")
        print(f"  Detection Accuracy: {det_acc:.4f} ({detection_correct}/{total})")

        # Save results
        with open(os.path.join(self.results_dir, "pipeline_predictions.json"), "w") as f:
            json.dump(results_log, f, indent=2)

        return {"detection_accuracy": det_acc, "total": total}

    def _save_segmentation_viz(self, image, gt_mask, pred_mask, save_path, dice, iou):
        """Save segmentation visualization."""
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        # Original image
        if isinstance(image, Image.Image):
            axes[0].imshow(image)
        else:
            axes[0].imshow(image.permute(1, 2, 0).numpy())
        axes[0].set_title("Original")
        axes[0].axis("off")

        # GT mask
        axes[1].imshow(gt_mask, cmap="gray")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        # Predicted mask
        axes[2].imshow(pred_mask, cmap="gray")
        axes[2].set_title(f"Prediction\nDice={dice:.3f}, IoU={iou:.3f}")
        axes[2].axis("off")

        # Overlay
        if isinstance(image, Image.Image):
            img_np = np.array(image.resize((1024, 1024)))
        else:
            img_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        overlay = img_np.copy()
        overlay[pred_mask > 0.5] = [255, 0, 0]  # Red for prediction
        overlay[gt_mask > 0.5] = [0, 255, 0]  # Green for GT
        # Overlap is yellow
        overlap = (pred_mask > 0.5) & (gt_mask > 0.5)
        overlay[overlap] = [255, 255, 0]

        axes[3].imshow(overlay)
        axes[3].set_title("Overlay (R=Pred, G=GT, Y=Overlap)")
        axes[3].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def generate_report(self, clf_results, seg_results, pipeline_results):
        """Generate comprehensive evaluation report."""
        print("\n" + "=" * 60)
        print("COMPREHENSIVE BLIND TEST REPORT")
        print("=" * 60)

        report = {
            "timestamp": datetime.now().isoformat(),
            "checkpoint": self.args.checkpoint,
            "dataset": "BUSI (Breast Ultrasound Images)",
            "device": self.device,
            "results": {
                "classification": clf_results,
                "segmentation": seg_results,
                "full_pipeline": pipeline_results
            }
        }

        # Print summary
        print("\n[CLASSIFICATION]")
        if clf_results:
            print(f"  Accuracy: {clf_results.get('accuracy', 0):.4f}")

        print("\n[SEGMENTATION]")
        if seg_results:
            print(f"  Dice Score:  {seg_results.get('dice_mean', 0):.4f}")
            print(f"  IoU:         {seg_results.get('iou_mean', 0):.4f}")
            print(f"  Boundary F1: {seg_results.get('boundary_f1_mean', 0):.4f}")

        print("\n[FULL PIPELINE]")
        if pipeline_results:
            print(f"  Detection Accuracy: {pipeline_results.get('detection_accuracy', 0):.4f}")

        # Compute overall score
        scores = []
        if clf_results and 'accuracy' in clf_results:
            scores.append(clf_results['accuracy'])
        if seg_results and 'dice_mean' in seg_results:
            scores.append(seg_results['dice_mean'])

        if scores:
            overall = np.mean(scores)
            print(f"\n[OVERALL SCORE]: {overall:.4f}")
            report["overall_score"] = float(overall)

        # Comparison with benchmarks
        print("\n[BENCHMARK COMPARISON]")
        print("  BUSI Dataset Literature Benchmarks:")
        print("  - Classification: ~80-98% accuracy (various methods)")
        print("  - Segmentation:   ~0.89 Dice (state-of-the-art)")
        print()

        # Save report
        report_path = os.path.join(self.results_dir, "blind_test_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {report_path}")

        # Also compute and save classification confusion matrix
        if self.evaluator.clf_metrics and self.evaluator.clf_metrics.predictions:
            clf_full = self.evaluator.clf_metrics.compute_metrics()
            self._plot_confusion_matrix(
                clf_full["confusion_matrix"],
                self.evaluator.clf_metrics.class_names,
                os.path.join(self.results_dir, "confusion_matrix.png")
            )

        return report

    def _plot_confusion_matrix(self, cm, class_names, save_path):
        """Plot and save confusion matrix."""
        fig, ax = plt.subplots(figsize=(8, 6))

        cm_array = np.array(cm)
        im = ax.imshow(cm_array, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            ylabel='True Label',
            xlabel='Predicted Label',
            title='Classification Confusion Matrix'
        )

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add text annotations
        thresh = cm_array.max() / 2.
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, format(cm_array[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm_array[i, j] > thresh else "black")

        fig.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Confusion matrix saved to: {save_path}")

    def run_all_tests(self):
        """Run all blind tests."""
        print("\nLoading BUSI dataset for blind testing...")

        # Load dataset
        factory = MedicalDatasetFactory(base_data_dir=self.args.data_dir)

        # Use test split for blind evaluation
        busi_dataset = BUSIDataset(
            os.path.join(self.args.data_dir, "Dataset_BUSI_with_GT"),
            split="test",
            task="both",
            image_size=1024
        )

        print(f"Loaded {len(busi_dataset)} samples for blind testing\n")

        # Run tests
        clf_results = self.run_classification_test(busi_dataset)
        seg_results = self.run_segmentation_test(busi_dataset)
        pipeline_results = self.run_full_pipeline_test(busi_dataset)

        # Generate report
        report = self.generate_report(clf_results, seg_results, pipeline_results)

        return report


def main():
    parser = argparse.ArgumentParser(description="MedGamma Blind Testing Evaluation")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/production",
                        help="Path to checkpoint directory")
    parser.add_argument("--data_dir", type=str, default="medical_data",
                        help="Path to medical data directory")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory for results")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Maximum samples to evaluate (-1 for all)")

    args = parser.parse_args()

    evaluator = BlindTestEvaluator(args)
    report = evaluator.run_all_tests()

    print("\n" + "=" * 60)
    print("BLIND TESTING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
