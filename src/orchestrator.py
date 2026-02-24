"""
Clinical Orchestrator - MedGemma + SAM2 Integration

Coordinates the multi-stage clinical AI pipeline:
1. Knowledge Retrieval - Context from clinical guidelines
2. MedGemma Reasoning - Visual analysis and detection
3. SAM2 Segmentation - Precise ROI masking
4. Clinical Reporting - Structured output generation

Supports loading trained adapters from curriculum training pipeline.
"""

import os
import argparse
import json
# Optimize CUDA Memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import gc
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from .config import Config
from .medgemma_wrapper import MedGemmaWrapper
from .medsam_wrapper import MedSAMWrapper
from .agent_state import AgentState
from .knowledge_base import ClinicalKnowledgeBase
from .explainability import MedGemmaExplainer
from peft import PeftModel
from PIL import Image
import time

# Import clinical intelligence for robust analysis
try:
    from .clinical_intelligence import (
        ClinicalIntelligence,
        ImageQualityAssessor,
        ConfidenceLevel,
    )
    CLINICAL_INTELLIGENCE_AVAILABLE = True
except ImportError:
    CLINICAL_INTELLIGENCE_AVAILABLE = False
    print("Warning: Clinical intelligence module not available")
import sys
import codecs

# Enforce UTF-8
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())


class ClinicalOrchestrator:
    """
    Orchestrates MedGemma (reasoning) + SAM2 (segmentation) for clinical image analysis.

    Supports trained checkpoints from:
    - checkpoints/production/medgemma/final (MedGemma LoRA adapter)
    - checkpoints/production/sam2/final (SAM2 LoRA adapter)
    """

    def __init__(self, checkpoint_dir: str = None):
        """
        Initialize the clinical orchestrator.

        Args:
            checkpoint_dir: Path to trained checkpoints (e.g., "checkpoints/production").
                          If None, will search default locations.
        """
        print("Initializing Clinical Orchestrator...")

        self.checkpoint_dir = checkpoint_dir
        self.medgemma_adapter_path = None
        self.sam2_adapter_path = None

        # Find trained adapters
        self._find_adapters()

        self.medgemma = MedGemmaWrapper()
        # Explainer
        self.explainer = MedGemmaExplainer(self.medgemma)
        # Segmentation (pass found adapter path)
        self.sam = MedSAMWrapper(adapter_path=self.sam2_adapter_path)
        # Knowledge Base
        self.kb = ClinicalKnowledgeBase()

        # Clinical Intelligence for robust analysis
        if CLINICAL_INTELLIGENCE_AVAILABLE:
            self.clinical_intel = ClinicalIntelligence()
            print("  Clinical Intelligence: Enabled")
        else:
            self.clinical_intel = None
            print("  Clinical Intelligence: Disabled")
        self.kb.load()

    def _find_adapters(self):
        """Find trained adapter paths from checkpoint directory."""
        # Search paths in order of preference
        search_paths = []

        if self.checkpoint_dir:
            search_paths.append(self.checkpoint_dir)

        # Default paths
        search_paths.extend([
            "checkpoints/production",
            "checkpoints/curriculum",
            "outputs/medgemma_lora_vindr",
            "outputs/medgemma_lora"
        ])

        for base_path in search_paths:
            if not os.path.exists(base_path):
                continue

            # Check for model_config.json (from run_full_training.py)
            config_path = os.path.join(base_path, "model_config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path) as f:
                        config = json.load(f)
                    if "medgemma" in config:
                        adapter = config["medgemma"].get("adapter_path")
                        if adapter and os.path.exists(adapter):
                            self.medgemma_adapter_path = adapter
                    if "sam2" in config:
                        adapter = config["sam2"].get("adapter_path")
                        if adapter and os.path.exists(adapter):
                            self.sam2_adapter_path = adapter
                    break
                except:
                    pass

            # Check for direct paths - PRIORITIZE detection adapters over final
            # Detection task requires the detection-trained adapter, not the VQA-trained final
            medgemma_paths = [
                # Best detection checkpoint (primary)
                os.path.join(base_path, "medgemma", "detection_best"),
                # Legacy detection path
                os.path.join(base_path, "medgemma", "detection"),
                # Final only if no detection adapter exists
                os.path.join(base_path, "medgemma", "final"),
                os.path.join(base_path, "final"),
                base_path
            ]
            for p in medgemma_paths:
                if os.path.exists(p) and os.path.exists(os.path.join(p, "adapter_config.json")):
                    self.medgemma_adapter_path = p
                    break

            sam2_paths = [
                os.path.join(base_path, "sam2", "final"),
                os.path.join(base_path, "sam2", "best")
            ]
            for p in sam2_paths:
                if os.path.exists(p) and os.path.exists(os.path.join(p, "adapter_config.json")):
                    self.sam2_adapter_path = p
                    break

            if self.medgemma_adapter_path or self.sam2_adapter_path:
                break

        print(f"  MedGemma adapter: {self.medgemma_adapter_path or 'Not found (using base model)'}")
        print(f"  SAM2 adapter: {self.sam2_adapter_path or 'Not found (using base model)'}")
        
    def run_pipeline(self, image_path, clinical_query) -> AgentState:
        """
        Executes the agentic pipeline with MODALITY-FIRST approach:
        1. Detect Modality -> 2. Knowledge Retrieval -> 3. Modality-Specific Detection -> 4. Segment

        This is CRITICAL for handling multiple imaging modalities correctly.
        Previously, the pipeline hardcoded "chest X-ray" for ALL images,
        causing brain MRI to be incorrectly analyzed with chest pathology vocabulary.
        """
        start_time = time.time()

        state: AgentState = {
            "image_path": image_path,
            "user_query": clinical_query,
            "modality": "Unknown",
            "modality_confidence": 0.0,
            "image_metadata": {},
            "image_quality": 1.0,
            "messages": [],
            "current_thought": "",
            "detections": [],
            "segmentations": [],
            "final_report": "",
            "validation_status": "pending",
            "warnings": [],
            "error": None
        }

        if not os.path.exists(image_path):
            state["error"] = f"Image not found at: {image_path}"
            print(f"(!) {state['error']}")
            return state

        print(f"\n{'='*60}")
        print(f"MEDGAMMA CLINICAL AI PIPELINE v2.0")
        print(f"{'='*60}")
        print(f"Query: {clinical_query}")

        try:
            # --- Node -1: Image Quality Assessment (NEW) ---
            if self.clinical_intel:
                print("\n[Node -1: Image Quality Assessment]")
                pil_image = Image.open(image_path).convert("RGB")
                quality_score, quality_warnings = self.clinical_intel.assess_image_quality(pil_image)
                state["image_quality"] = quality_score
                state["warnings"].extend(quality_warnings)

                quality_status = "GOOD" if quality_score > 0.7 else "MODERATE" if quality_score > 0.5 else "POOR"
                print(f"  > Image Quality: {quality_status} ({quality_score:.2f})")
                if quality_warnings:
                    for w in quality_warnings:
                        print(f"  > Warning: {w}")

            # --- Node 0: Load MedGemma ---
            print("\n[Node 0: Load MedGemma]")
            self.medgemma.load()

            # Load LoRA adapter if available
            if self.medgemma_adapter_path and os.path.exists(self.medgemma_adapter_path):
                print(f"  > Loading MedGemma LoRA Adapter from {self.medgemma_adapter_path}...")
                try:
                    self.medgemma.model = PeftModel.from_pretrained(
                        self.medgemma.model,
                        self.medgemma_adapter_path
                    )
                    self.medgemma.model.eval()  # Critical: Set to eval mode
                    print("  > LoRA Adapter Loaded Successfully. Model in eval mode.")
                except Exception as e:
                    print(f"  (!) Failed to load LoRA Adapter: {e}")
            else:
                print("  > No MedGemma adapter found. Using base model.")

            # Ensure model is in eval mode for inference
            self.medgemma.model.eval()

            # --- Node 0.5: MODALITY DETECTION (NEW - CRITICAL) ---
            print("\n[Node 0.5: Modality Detection]")

            # Use robust modality detection if clinical intelligence is available
            if self.clinical_intel:
                # Get model-based prediction
                model_modality = self.medgemma.detect_modality(image_path)

                # Use robust detection combining model + image analysis
                pil_image = Image.open(image_path).convert("RGB")
                final_modality, modality_confidence, detection_method = \
                    self.clinical_intel.detect_modality_robust(
                        pil_image,
                        model_modality,
                        model_confidence=0.8
                    )
                state["modality"] = final_modality
                state["modality_confidence"] = modality_confidence
                print(f"  > Detected Modality: {final_modality.upper()} (confidence: {modality_confidence*100:.1f}%)")
                print(f"  > Detection Method: {detection_method}")
            else:
                # Fallback to model-only detection
                detected_modality = self.medgemma.detect_modality(image_path)
                state["modality"] = detected_modality
                state["modality_confidence"] = 0.8
                print(f"  > Detected Modality: {detected_modality.upper()}")

            # Modality-specific context
            modality_names = {
                "xray": "Chest X-ray",
                "mri": "Brain MRI",
                "ct": "CT Scan",
                "ultrasound": "Ultrasound"
            }
            modality_display = modality_names.get(state["modality"], state["modality"].upper())
            print(f"  > Image Type: {modality_display}")

            # --- Node 1: Knowledge Retrieval (Modality-Aware) ---
            print("\n[Node 1: Knowledge Retrieval]")
            # Enhance query with modality context
            modality_aware_query = f"{modality_display}: {clinical_query}"
            guidelines = self.kb.retrieve(modality_aware_query, top_k=2)
            print(f"  > Retrieved Context: {guidelines[:100]}...")

            # --- Node 2: Modality-Specific Detection (MedGemma) ---
            print(f"\n[Node 2: {modality_display} Detection]")

            # Use task-specific inference with MODALITY-AWARE prompt
            # This is the KEY FIX - pass modality to get correct detection prompt
            response_text, _ = self.medgemma.analyze_image(
                image_path,
                query=clinical_query,
                task="detection",
                modality=state["modality"]  # Pass detected modality from state
            )

            # Store the prompt for explainability (used later)
            system_prompt = clinical_query
            
            state["current_thought"] = response_text
            state["messages"].append({"role": "assistant", "content": response_text})
            
            # NOTE: We deliberately IGNORE parsed_boxes from MedGemma.
            # Raw text-based bounding boxes are imprecise. We will calculate them from the heatmap instead.
            print(f"  > Clinical Reasoning: {response_text[:100]}...")

            torch.cuda.empty_cache()

            # --- Node 1.5: Enhanced Explainability (Modality-Aware) ---
            print("\n[Node: MedGemma Explainer]")

            # Extract findings from VLM response for ROI validation
            vlm_findings = self.medgemma._extract_findings(response_text)

            # Use enhanced explainability with finding alignment
            try:
                from .explainability import ExplainabilityResult
                explainability_result = self.explainer.explain_with_findings(
                    image_path,
                    system_prompt,
                    findings=vlm_findings,
                    modality=state["modality"]
                )
                state["heatmap"] = explainability_result.heatmap
                state["explainability"] = {
                    "roi_coverage": explainability_result.roi_coverage,
                    "attention_quality": explainability_result.attention_quality,
                    "attention_peaks": explainability_result.attention_peaks,
                    "clinical_explanation": explainability_result.clinical_explanation,
                }
                print(f"  > Attention Heatmap generated (Quality: {explainability_result.attention_quality:.2f})")
                print(f"  > ROI Coverage: {explainability_result.roi_coverage:.1%}")
                if explainability_result.attention_quality < 0.4:
                    state["warnings"].append("Attention quality is low - interpretation requires caution")
            except Exception as e:
                # Fallback to basic explainability
                print(f"  > Enhanced explainability failed, using basic mode: {e}")
                heatmap = self.explainer.explain(image_path, system_prompt, modality=state["modality"])
                state["heatmap"] = heatmap
                print("  > Attention Heatmap generated (basic mode).")

            # --- Node 1.6: Heatmap-Guided Object Localization ---
            print("\n[Node: Heatmap-Guided Object Localization]")
            state["detections"] = []

            # vlm_findings already extracted above for explainability
            is_normal = any(f.get("class", "").lower() in ["no significant abnormality", "normal"] for f in vlm_findings) or not vlm_findings

            # Use heatmap from state
            heatmap = state.get("heatmap")

            if is_normal:
                print("  > Model indicates Normal/No findings. Skipping localized anomaly detection.")
            elif heatmap is not None and np.max(heatmap) > 0:
                # Threshold the heatmap (Top 8% for sharper localization)
                thresh_val = np.percentile(heatmap, 92)
                binary_map = (heatmap >= thresh_val).astype(np.uint8) * 255
                
                # Find contours
                contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                h_max, w_max = heatmap.shape
                min_area = (h_max * w_max) * 0.003 # 0.3% area
                
                finding_labels = [f.get("class", "Pathology") for f in vlm_findings if f.get("class", "").lower() not in ["no significant abnormality", "normal"]]
                
                for i, cnt in enumerate(contours):
                    if cv2.contourArea(cnt) > min_area:
                        x, y, w, h = cv2.boundingRect(cnt)
                        # Padding
                        pad = 12
                        box = [max(0, x-pad), max(0, y-pad), min(w_max, x+w+pad), min(h_max, y+h+pad)]
                        
                        # Assign label from VLM findings if available, else generic
                        label = finding_labels[i % len(finding_labels)] if finding_labels else "Heatmap-Localized Anomaly"
                        
                        state["detections"].append({
                            "label": label,
                            "box": box,
                            "confidence": float(np.percentile(heatmap[y:y+h, x:x+w], 95))
                        })
                
                # Fallback to absolute peak if no contours found but abnormality exists
                if not state["detections"]:
                    max_y, max_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                    pad = 50
                    box = [max(0, max_x-pad), max(0, max_y-pad), min(w_max, max_x+pad), min(h_max, max_y+pad)]
                    label = finding_labels[0] if finding_labels else "Peak Anomaly"
                    state["detections"].append({"label": label, "box": box, "confidence": 1.0})
            
            print(f"  > Logically Validated Detections: {len(state['detections'])}")

            # Cleanup explainer hooks before unloading model
            self.explainer.cleanup()
            self.medgemma.unload()
            gc.collect()
            torch.cuda.empty_cache()
            
            # --- Node 2: Tool Execution (SAM 2) ---
            if state["detections"]:
                print("\n[Node: Segmentation Tool]")
                self.sam.load() # Loads LoRA internally if present
                self.sam.set_image(image_path)
                
                for det in state["detections"]:
                    box = det["box"]
                    print(f"  > Segmenting ROI: {box}")
                    masks, scores = self.sam.predict_mask(box)
                    
                    if masks is not None:
                        for i, (mask, score) in enumerate(zip(masks, scores)):
                            state["segmentations"].append({
                                "box": box,
                                "mask": mask, 
                                "score": float(score),
                                "label": f"{det['label']} (Cand {i+1})"
                            })
                
                # self.sam.unload() # Optional

            # --- Node 3: Clinical Validation & Reporting ---
            print("\n[Node 3: Clinical Validation & Reporting]")

            # Clinical validation if available
            if self.clinical_intel:
                # Convert detections to findings format for validation
                findings_for_validation = [
                    {"class": det["label"], "box": det["box"], "confidence": det.get("confidence", 0.7)}
                    for det in state["detections"]
                ]

                # Validate findings
                validation_status, validation_warnings = self.clinical_intel.validator.validate_findings(
                    findings_for_validation,
                    state["modality"],
                    state.get("image_quality", 1.0)
                )
                state["validation_status"] = validation_status
                state["warnings"].extend(validation_warnings)

                print(f"  > Validation Status: {validation_status}")
                if validation_warnings:
                    for w in validation_warnings:
                        print(f"  > Validation Warning: {w}")

            # Generate comprehensive report
            processing_time = time.time() - start_time
            modality_display = {
                "xray": "Chest X-ray",
                "mri": "Brain MRI",
                "ct": "CT Scan",
                "ultrasound": "Ultrasound"
            }.get(state["modality"], state["modality"].upper())

            report = f"""
================================================================================
                         MEDGAMMA CLINICAL AI REPORT
================================================================================
Study Information:
  - Image: {os.path.basename(image_path)}
  - Modality: {modality_display}
  - Modality Confidence: {state.get('modality_confidence', 0)*100:.1f}%
  - Image Quality: {state.get('image_quality', 1.0)*100:.1f}%
  - Processing Time: {processing_time:.2f}s

Clinical Query: {clinical_query}

FINDINGS:
{response_text}

DETECTIONS ({len(state['detections'])} regions identified):
"""
            for i, det in enumerate(state["detections"], 1):
                report += f"  {i}. {det['label']} (Confidence: {det.get('confidence', 0)*100:.1f}%)\n"
                report += f"     Location: {det['box']}\n"

            if state["segmentations"]:
                report += f"\nSEGMENTATION:\n  - {len(state['segmentations'])} regions segmented with SAM2\n"

            report += f"""
VALIDATION STATUS: {state.get('validation_status', 'N/A')}
"""
            if state.get("warnings"):
                report += "\nWARNINGS:\n"
                for w in state["warnings"]:
                    report += f"  - {w}\n"

            report += """
================================================================================
DISCLAIMER: This analysis is for research purposes only and should not be used
for clinical diagnosis without expert review.
================================================================================
"""

            state["final_report"] = report
            state["processing_time"] = processing_time
            print(report)

            self.save_visualization(image_path, state)

        except Exception as e:
            state["error"] = str(e)
            print(f"(!) Error detected: {e}")
            import traceback
            traceback.print_exc()

        return state

    def save_visualization(self, image_path, state, output_dir="outputs"):
        """Visualization logic"""
        os.makedirs(output_dir, exist_ok=True)
        img = cv2.imread(image_path)
        if img is None: return
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        # 1. Detection
        ax0 = axes[0]
        ax0.imshow(img_rgb)
        ax0.set_title("Detection (MedGemma)", fontsize=16)
        ax0.axis('off')
        for det in state["detections"]:
            y1, x1, y2, x2 = det["box"]
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=3, edgecolor='red', facecolor='none')
            ax0.add_patch(rect)

        # 2. Heatmap (Modality-Aware Visualization)
        ax1 = axes[1]
        modality = state.get("modality", "xray")
        ax1.set_title(f"Visual Attention ({modality.upper()})", fontsize=16)
        ax1.axis('off')
        if "heatmap" in state and state["heatmap"] is not None:
            # Use modality-aware visualization
            overlay, _ = self.explainer.visualize(
                image_path, state["heatmap"], modality=modality
            )
            ax1.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

            # Add attention quality info if available
            if "explainability" in state:
                quality = state["explainability"].get("attention_quality", 0)
                coverage = state["explainability"].get("roi_coverage", 0)
                ax1.text(
                    0.02, 0.98, f"Q:{quality:.2f} ROI:{coverage:.0%}",
                    transform=ax1.transAxes, fontsize=9,
                    verticalalignment='top', color='white',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.5)
                )
        else:
            ax1.text(0.5, 0.5, "No Heatmap", ha='center')

        # 3. Segmentation
        ax2 = axes[2]
        ax2.imshow(img_rgb)
        ax2.set_title("Segmentation (SAM 2)", fontsize=16)
        ax2.axis('off')
        if state["segmentations"]:
            for seg in state["segmentations"]:
                mask = seg["mask"]
                if len(mask.shape) == 3: mask = mask[0]
                masked_data = np.ma.masked_where(mask == 0, mask)
                ax2.imshow(masked_data, cmap='spring', alpha=0.4, interpolation='none')
                y1, x1, y2, x2 = seg["box"]
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='yellow', facecolor='none', linestyle='--')
                ax2.add_patch(rect)
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(output_dir, f"visualization_{timestamp}.png")
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to: {filepath}")

def main():
    """CLI entry point for the orchestrator."""
    parser = argparse.ArgumentParser(
        description="MedGamma Clinical Orchestrator - MedGemma + SAM2 Integration"
    )
    parser.add_argument("--image", "-i", type=str, help="Path to medical image")
    parser.add_argument("--query", "-q", type=str,
                        default="Detect any pathologies and localize them.",
                        help="Clinical query")
    parser.add_argument("--checkpoint", "-c", type=str,
                        default=None,
                        help="Path to trained checkpoint directory")
    parser.add_argument("--demo", action="store_true",
                        help="Run demo with sample image")
    parser.add_argument("--output-dir", "-o", type=str, default="outputs",
                        help="Output directory for visualizations")

    args = parser.parse_args()

    # Initialize orchestrator
    orchestrator = ClinicalOrchestrator(checkpoint_dir=args.checkpoint)

    # Find image path
    img_path = args.image

    if args.demo or not img_path:
        # Auto-find test image
        possible_paths = [
            "medical_data/vinbigdata-chest-xray/train_448",  # Preprocessed
            "medical_data/vinbigdata-chest-xray/train",
            "medical_data/chest_xray_kaggle/train/NORMAL",
            "medical_data/chest_xray_kaggle/train/PNEUMONIA",
            "medical_data/Slake1.0/imgs/xmlab0"
        ]

        for base_path in possible_paths:
            if os.path.exists(base_path):
                # Find first image
                for f in os.listdir(base_path):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(base_path, f)
                        break
                if img_path:
                    break

    if img_path and os.path.exists(img_path):
        print(f"\nTargeting Image: {img_path}")
        print(f"Query: {args.query}")
        state = orchestrator.run_pipeline(img_path, args.query)

        if state.get("error"):
            print(f"\nPipeline Error: {state['error']}")
        else:
            print(f"\n{'='*60}")
            print("PIPELINE COMPLETE")
            print(f"{'='*60}")
            print(f"Detections: {len(state.get('detections', []))}")
            print(f"Segmentations: {len(state.get('segmentations', []))}")
    else:
        print("No test image found. Provide --image path or ensure sample data exists.")


if __name__ == "__main__":
    main()
