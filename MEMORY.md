# GemSAM Project Memory

## 1. Project Overview
**GemSAM** is an agentic framework for explainable multi-modal medical image analysis on the edge. It integrates **MedGemma-1.5** (LLM reasoning and detection) with **SAM2** (precise segmentation) to provide clinical-grade reliability, visual grounding, and structured reporting.

## 2. System Architecture
- **MedGemma-1.5 (4B):** Acts as the "Radiologist Agent." It performs initial screening, identifies imaging modalities, reasons about clinical findings, and provides rough bounding boxes.
- **SAM2 (Hiera-Tiny):** Acts as the "Visual Refiner." It takes rough boxes from MedGemma and generates precise pixel-level masks.
- **Visual Grounding:** A hybrid logic that "snaps" MedGemma's predicted boxes to the physical boundaries of the SAM2 segmentation masks, ensuring findings are grounded in literal pixels.
- **Clinical Intelligence:** Post-processing layers that filter hallucinations, validate vocabulary (VinDr/NIH standards), and generate structured reports.

## 3. Key Development Milestones & Fixes
- **v2.0: Multi-Stage Pipeline:** Integrated logic for Screening, Modality detection, Detection, and VQA.
- **v2.1: Visual Grounding & Box Snapping:**
    - Implemented `_refine_box_from_mask` in `orchestrator.py`.
    - Discarded "text-only" localizations in favor of "mask-derived" precise boxes.
- **v2.2: Anti-Memorization & Precision:**
    - **Prompt Redesign:** detection prompts in `modality_prompts.py` rewritten to force the model to "prove" its findings by describing actual image pixels.
    - **Hallucination Interceptor:** Added `TEMPLATE_COORDS` in `medgemma_wrapper.py` to filter out memorized training coordinates appearing on healthy images.
- **v2.2.1: Performance & Stability:**
    - **Screening Timeouts:** Fixed 60s/30s evaluation timeouts by adding explicit `<end_of_turn>` hints to prompts and reducing `max_new_tokens` to 20 for single-word tasks.
    - **Adapter Swapping:** Implemented robust PEFT unwrapping in `evaluate_all_stages.py` to prevent "Adapter already exists" conflicts during long evaluation runs.
- **v3.0: Clinical-Grade Metrics & Training Optimization:**
    - **Infrastructure:** Fixed data splits (all stages use `test` split now). Separated 13 clinically correct synonym groups.
    - **Metrics:** Implemented mAP@0.5, FROC (at 0.5, 1, 2, 4 FP/img), and bootstrap 95% CIs for robust performance assessment.
    - **Detection:** Reduced max tokens to 384 for eval; added box-level IoU matching to the blind benchmark.
    - **Training:** Switched to compressed JSON targets (`c/b/r` keys) for 40% token savings; added cosine LR scheduler; enabled augmentation.

## 4. Core Core Components
| Component | File | Responsibility |
| :--- | :--- | :--- |
| **Orchestrator** | `src/orchestrator.py` | Pipeline flow, box snapping, reporting. |
| **VLM Wrapper** | `src/medgemma_wrapper.py` | Token injection, hallucination filtering, JSON repair. |
| **SAM Wrapper** | `src/medsam_wrapper.py` | SAM2 model loading and mask prediction. |
| **Prompts** | `src/prompts/modality_prompts.py` | Centralized, modality-aware clinical prompts. |
| **Evaluation** | `src/eval/evaluate_all_stages.py` | Multi-adapter holdout evaluation loop. |
| **Stopping** | `src/utils/stopping_criteria.py` | Custom criteria to prevent infinite loops/timeouts. |

## 5. Deployment & Execution
- **Multi-Stage Evaluation:**
  ```bash
  python -m src.eval.evaluate_all_stages --checkpoint checkpoints/production --max_samples 100
  ```
- **End-to-End Inference:**
  ```bash
  python src/orchestrator.py --image path/to/image.jpg --query "Analyze this image"
  ```

## 6. Critical Technical Notes
- **Token Injection:** Always use direct token ID injection (255999) for images to avoid tokenizer inconsistencies.
- **Model Config:** `model.config.image_token_index` MUST be synchronized with the tokenizer ID (255999) for LoRA adapters to function.
- **JSON Repair:** `medgemma_wrapper.py` contains a stack-based JSON extractor to handle truncated outputs from small-context or timeout scenarios.
