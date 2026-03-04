# GemSAM: An Agentic Framework for Explainable Multi-Modal Medical Image Analysis on Edge using MedGemma-1.5 and SAM2

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-MedGemma%20Impact%20Challenge-20BEFF)](https://www.kaggle.com/competitions/medgemma-impact-challenge)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)

> **Kaggle MedGemma Impact Challenge Submission**
> Google DeepMind

---

## Abstract

**GemSAM** is an **agentic framework** for explainable multi-modal medical image analysis that combines vision-language understanding with precise image segmentation. The system integrates **MedGemma 1.5** (a 4B parameter medical vision-language model) as the cognitive reasoning engine ("Brain") with **SAM 2** (Segment Anything Model 2) as the precise segmentation module ("Hands"), orchestrated through an agentic architecture that mimics clinical decision-making workflows.

<img width="1401" height="1132" alt="Untitled Diagram drawio" src="https://github.com/user-attachments/assets/82a78fce-f0f3-4478-8cc7-2edcc55673c7" />


### Key Features (v2.2)

| Feature | Description |
|---------|-------------|
| **Multi-Modal Support** | Chest X-rays, Brain MRI, CT scans, Ultrasound with modality-specific prompts |
| **Modality-First Pipeline** | Auto-detects imaging modality → selects appropriate clinical vocabulary |
| **ROI-Driven Box Refinement** | **[NEW]** SAM2 masks "snap" bounding boxes to exact visual features (Visual Grounding) |
| **Anti-Memorization** | **[NEW]** Prompts redesigned to force pixel-level analysis over template recall |
| **Hallucination Interceptor**| **[NEW]** Filters out known training-set coordinate hallucinations from results |
| **Binary Screening** | **[NEW]** Strict "HEALTHY"/"ABNORMAL" mode to prevent 30s timeouts |
| **Detection Accuracy** | Per-image accuracy, per-class P/R/F1, confidence score (0–1), clinical grade (A–D) |
| **VQA with Prompt Masking** | Model learns only answer tokens; multi-level matching (exact, synonym, Jaccard) |
| **Enhanced Explainability** | Gradient-weighted attention maps with ROI validation |
| **JSON Robustness** | Depth-clamped brace/bracket counting handles malformed LLM output |
| **BICUBIC Consistency** | Train/inference resampling aligned (BICUBIC), ~20% preprocessing speedup |

### Current Training Status

| Model | Status | Final Loss | Training Time |
|-------|--------|------------|---------------|
| **MedGemma Detection** | Complete | 0.372 | ~23.5 hours |
| **SAM2 Segmentation** | Complete | 0.16 | ~15 min |

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Complete Installation Guide](#complete-installation-guide)
3. [Training Guide](#training-guide)
4. [End-to-End Training Command](#end-to-end-training-command)
5. [Fine-Tuning Details](#fine-tuning-details)
6. [Multi-Stage Pipeline](#multi-stage-pipeline)
7. [System Architecture](#system-architecture)
8. [Datasets](#datasets)
9. [Results & Evaluation](#results--evaluation)
10. [Edge Deployment](#edge-deployment-raspberry-pi-5)
11. [Citation](#citation)

---

## Quick Start

```bash
# 1. Setup environment
conda create -n medgamma python=3.10 && conda activate medgamma
pip install -r requirements.txt
huggingface-cli login

# 2. Validate pipeline (6 tests)
python test_pipeline.py

# 3. RUN EVERYTHING (train + eval + blind test)
python run_all.py --all --disk-cache-dir D:\medgamma_cache
# Quick test: python run_all.py --all --quick
# Eval only:  python run_all.py --eval-only

# 4. Run demo
python demo/gradio_app.py
```

---

## Complete Installation Guide

### Step 1: System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | 12GB VRAM | 16GB+ VRAM (RTX 4060 Ti, RTX 4080) |
| **RAM** | 24GB | 32GB+ DDR5 |
| **Storage** | 50GB | 100GB+ SSD (for disk cache) |
| **CUDA** | 11.8+ | 12.1+ |
| **Python** | 3.10 | 3.10+ |

### Step 2: Environment Setup

```bash
# Create conda environment
conda create -n medgamma python=3.10 -y
conda activate medgamma

# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
pip install -r requirements.txt

# Install SAM 2
pip install git+https://github.com/facebookresearch/segment-anything-2.git

# Download SAM 2 checkpoint
mkdir -p checkpoints/sam2
wget -O checkpoints/sam2/sam2_hiera_tiny.pt https://dl.fbaipublicfiles.com/segment_anything_2/sam2_hiera_tiny.pt

# Login to HuggingFace (required for MedGemma access)
huggingface-cli login
# Enter your HuggingFace token when prompted

# OR set as environment variable (Recommended for development)
export HF_TOKEN="your_token_here"  # Linux/Mac
set HF_TOKEN="your_token_here"     # Windows (Command Prompt)
$env:HF_TOKEN="your_token_here"    # Windows (PowerShell)
```

### Step 3: Dataset Preparation

```bash
# Create data directory
mkdir -p medical_data

# Required datasets (download from respective sources):
# 1. VinDr-CXR: https://vindr.ai/datasets/cxr
#    → Extract to: medical_data/vinbigdata-chest-xray/

# 2. Brain Tumor MRI: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
#    → Extract to: medical_data/Brain Tumor MRI Dataset/

# 3. SLAKE (for VQA): https://www.med-vqa.com/slake/
#    → Extract to: medical_data/Slake1.0/

# Directory structure after setup:
medical_data/
├── vinbigdata-chest-xray/
│   ├── train/
│   ├── test/
│   └── train.csv
├── Brain Tumor MRI Dataset/
│   ├── Training/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   └── Testing/
└── Slake1.0/
    ├── imgs/
    ├── train.json
    └── test.json
```

### Step 4: Validate Installation

```bash
python test_pipeline.py

# Expected output:
# ============================================================
# MedGamma Pipeline Validation
# ============================================================
# TEST 1: Modality-Specific Prompts
#   [PASS] XRAY: 22 classes, prompt length: ...
#   [PASS] MRI: 16 classes, prompt length: ...
#   [PASS] CT: 11 classes, prompt length: ...
#   [PASS] ULTRASOUND: 8 classes, prompt length: ...
#   [PASS] Modality normalization
# TEST 2: Clinical Intelligence Module
#   [PASS] Clinical intelligence imports
#   [PASS] ClinicalIntelligence instantiated
#   [PASS] ConfidenceLevel enum
# TEST 3: Enhanced Explainability Module
#   [PASS] Explainability imports
#   [PASS] Modality visualization configs
#   [PASS] AttentionAggregation enum
#   [PASS] ExplainabilityResult dataclass
# TEST 4: Data Factory Modality Support
#   [PASS] MedicalDatasetFactory imports
#   [PASS] MedicalDatasetFactory instantiated
#   [PASS] Dataset loaders available
# TEST 5: MedGemma Wrapper Modality Support
#   [PASS] MedGemmaWrapper imports
#   [PASS] detect_modality method exists
# TEST 6: Orchestrator Integration
#   [PASS] ClinicalOrchestrator imports
#   [PASS] Key methods exist
# ============================================================
# Results: 6/6 tests passed
# ============================================================
```

---

## Training Guide

### Training Modes Overview

| Mode | Command | Images | Epochs | LoRA Rank | Time | Quality |
|------|---------|--------|--------|-----------|------|---------|
| **TURBO** | `--turbo` | 1,000 | 1 | 4 | ~45-60 min | Demo only |
| **BALANCED** | `--balanced` | 3,000 | 2 | 8 | ~2-3 hours | Good |
| **FULL** | `--full` | 10,500 | 1 | 16 | ~14-24 hours | **Best** |
| **EXTENDED** | `--extended` | 10,500 | 3 | 16 | ~40-45 hours | Maximum |

### Memory Management Options

| Option | Description | RAM Usage | Speed | Stability |
|--------|-------------|-----------|-------|-----------|
| `--disk-cache` | Cache processed images to SSD | ~12GB | Fast | **Excellent** |
| `--stream` | No caching, load-use-discard | ~10GB | Slower | Excellent |
| `--stable` | Auto-configure for crash-free training | ~12GB | Fast | **Excellent** |
| Default (RAM cache) | Pre-load images to RAM | ~28-30GB | Fastest | **Risky on 32GB** |

> **IMPORTANT**: On systems with 32GB RAM, use `--disk-cache` or `--stream` to prevent memory exhaustion. RAM caching can cause 10x slowdowns due to memory pressure.

### Training Commands by Use Case

```bash
# 1. KAGGLE COMPETITION (Recommended)
# Full training with disk cache for stability
python retrain_detection_optimized.py \
    --full \
    --disk-cache \
    --disk-cache-dir D:\medgamma_cache

# 2. LIMITED RAM (32GB or less)
# Stream mode - most stable, no caching
python retrain_detection_optimized.py \
    --full \
    --stream

# 3. QUICK ITERATION / TESTING
# Balanced mode - good quality in 2-3 hours
python retrain_detection_optimized.py --balanced

# 4. DEMO ONLY
# Turbo mode - fast but poor detection quality
python retrain_detection_optimized.py --turbo

# 5. MAXIMUM QUALITY (if you have time)
# Extended mode - 3 epochs on all data
python retrain_detection_optimized.py --extended --disk-cache
```

---

## End-to-End Training, Evaluation & Blind Test

### **Complete 6-Step Pipeline**

```bash
# ============================================================
# STEP 0: VALIDATE PIPELINE (run once)
# ============================================================
python test_pipeline.py

# ============================================================
# STEP 1: CURRICULUM TRAINING — Screening, Modality, VQA (~2h)
# Trains LoRA adapters for screening (binary), modality
# classification, and VQA stages via curriculum learning
# ============================================================
python run_curriculum.py --data-dir medical_data --output-dir checkpoints/production
# Quick test: python run_curriculum.py --quick
# Single stage: python run_curriculum.py --stage detection

# ============================================================
# STEP 2: PRE-BUILD DISK CACHE (run once, ~30-45 min)
# Preprocesses all images to SSD for fast detection training
# ============================================================
python retrain_detection_optimized.py \
    --preprocess-cache \
    --disk-cache-dir D:\medgamma_cache

# ============================================================
# STEP 3: DETECTION TRAINING (~14-24 hours)
# Full detection retraining with modality-aware prompts,
# class-balanced sampling, and prompt length caching
# ============================================================
python retrain_detection_optimized.py \
    --full \
    --disk-cache \
    --disk-cache-dir D:\medgamma_cache \
    --epochs 1 \
    --batch_size 2 \
    --grad_accum 8 \
    --lr 2e-5 \
    --lora_r 16

# ============================================================
# STEP 4: SAM2 SEGMENTATION TRAINING (~10-15 min)
# Trains on combined BUSI + SLAKE (~1228 samples)
# Uses epoch-level early stopping on validation Dice
# ============================================================
python -m src.train.train_sam2_medical \
    --data-dir medical_data \
    --dataset all \
    --epochs 5 \
    --batch-size 1 \
    --lr 1e-4

# ============================================================
# STEP 5: MULTI-STAGE EVALUATION (all stages)
# Reports: accuracy, F1, confidence, clinical grade per stage
# ============================================================
python -m src.eval.evaluate_all_stages \
    --checkpoint checkpoints/production \
    --max_samples 200

# ============================================================
# STEP 6: BLIND BENCHMARK (multi-modal, unseen data)
# Evaluates all stages on held-out test splits + MedMNIST v2
# Sections: screening, modality, detection, VQA, segmentation,
#           external benchmark (6 MedMNIST subsets)
# ============================================================
python blind_benchmark.py \
    --checkpoint checkpoints/production \
    --data-dir medical_data \
    --max-samples 200
# Results saved to: outputs/blind_benchmark/benchmark_report.json

# ============================================================
# STEP 7: LAUNCH DEMO
# ============================================================
python demo/gradio_app.py
```

### **Single Command (all-in-one)**

```bash
# Runs: validation -> curriculum -> cache -> detection -> SAM2 -> eval -> blind test
python run_all.py --all --disk-cache-dir D:\medgamma_cache

# Quick test (~1h total, turbo mode)
python run_all.py --all --quick

# Evaluation + blind test only (no training)
python run_all.py --eval-only

# Full pipeline but skip curriculum (detection already uses separate adapters)
python run_all.py --all --skip-curriculum

# Custom detection mode
python run_all.py --all --detection-mode balanced --no-demo
```

---

## Fine-Tuning Details

### MedGemma LoRA Fine-Tuning

```python
# Configuration in retrain_detection_optimized.py
LoRA_CONFIG = {
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"       # MLP
    ],
    "r": 16,              # Rank (FULL mode)
    "lora_alpha": 32,     # Scaling factor
    "lora_dropout": 0.1,  # Regularization
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

# Trainable Parameters: ~32.8M (0.76% of 4B total)
```

### Dataset-Specific Fine-Tuning

| Dataset | Task | Modality | Prompt Type | Output Format |
|---------|------|----------|-------------|---------------|
| **VinDr-CXR** | Detection | X-ray | X-ray specific | JSON with boxes |
| **NIH ChestX-ray** | Detection | X-ray | X-ray specific | JSON with boxes |
| **Brain Tumor MRI** | Detection | MRI | MRI specific | JSON with boxes |
| **SLAKE** | VQA | Multi-modal | Question-based | Free text |
| **VQA-RAD** | VQA | Multi-modal | Question-based | Free text |
| **BUSI** | Segmentation | Ultrasound | SAM2 prompts | Binary masks |

### Modality-Specific Prompts

```python
# src/prompts/modality_prompts.py

# X-RAY DETECTION PROMPT (22 pathology classes)
XRAY_PROMPT = """
Analyze this chest X-ray for pathological findings.
Look for: cardiomegaly, pneumonia, pleural effusion, consolidation,
atelectasis, pulmonary nodules, masses, pneumothorax, rib fractures,
aortic enlargement, emphysema, fibrosis, infiltration, ILD, edema,
pleural thickening, calcification, and other abnormalities.
For each finding, provide the class name and bounding box coordinates
[x1,y1,x2,y2] normalized to 0-1000.
Output in JSON format: {"findings": [...]}
"""

# MRI DETECTION PROMPT (16 pathology classes)
MRI_PROMPT = """
Analyze this brain MRI for pathological findings.
Look for: glioma, meningioma, pituitary adenoma, metastasis,
hemorrhage, infarction, white matter lesion, hydrocephalus,
mass effect, midline shift, edema, cyst, abscess, brain atrophy.
For each finding, provide class name and bounding box [x1,y1,x2,y2].
Include tumor grade (if applicable) and anatomical location
(frontal, temporal, parietal, occipital, cerebellum).
Output in JSON format: {"findings": [...]}
"""

# CT DETECTION PROMPT (11 pathology classes)
CT_PROMPT = """
Analyze this CT scan for pathological findings.
Look for: tumors, hemorrhage, infarction, fractures, calcifications,
masses, effusions, abscesses, edema, and other lesions.
Include Hounsfield unit assessment and size estimation.
Output in JSON format: {"findings": [...]}
"""

# ULTRASOUND DETECTION PROMPT (8 pathology classes)
ULTRASOUND_PROMPT = """
Analyze this ultrasound image for pathological findings.
Look for: benign masses, malignant masses, cysts, calcifications,
abnormal echogenicity, fluid collections, and other lesions.
Describe echogenicity (hypoechoic, hyperechoic, anechoic, mixed)
and margins (well-defined, irregular).
Output in JSON format: {"findings": [...]}
"""
```

### SAM2 Fine-Tuning

```python
# Configuration in src/train/train_sam2_medical.py
SAM2_LORA_CONFIG = {
    "target_modules": ["qkv", "proj"],  # Hiera attention blocks
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
}

# Loss Function
loss = 0.5 * dice_loss + 0.5 * bce_loss

# Trainable Parameters: ~238K (0.87% of encoder)
```

---

## Multi-Stage Pipeline

### Pipeline Stages Explained

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          MEDGAMMA MULTI-STAGE PIPELINE                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  STAGE -1: IMAGE QUALITY ASSESSMENT                                             │
│  ├── Input: Raw medical image                                                   │
│  ├── Process: Check resolution, contrast, brightness, artifacts                │
│  ├── Output: Quality score (0-1), warnings list                                │
│  └── Action: Flag low-quality images for cautious interpretation               │
│                                                                                  │
│  STAGE 0: MODEL LOADING                                                         │
│  ├── Load MedGemma 1.5 4B (base model)                                         │
│  ├── Apply LoRA adapter from checkpoints/production/medgemma/                   │
│  └── Set model to eval mode                                                     │
│                                                                                  │
│  STAGE 0.5: MODALITY DETECTION (CRITICAL)                                       │
│  ├── Input: Medical image                                                       │
│  ├── Method 1: MedGemma prediction (VLM-based)                                 │
│  ├── Method 2: Image characteristic analysis (intensity, contrast)             │
│  ├── Output: Modality (xray/mri/ct/ultrasound), confidence (0-1)              │
│  └── Purpose: Select correct detection prompt and vocabulary                   │
│                                                                                  │
│  STAGE 1: KNOWLEDGE RETRIEVAL (RAG)                                             │
│  ├── Input: Modality + clinical query                                          │
│  ├── Process: Retrieve relevant clinical protocols from knowledge base         │
│  └── Output: Context for detection prompt                                       │
│                                                                                  │
│  STAGE 2: MODALITY-SPECIFIC DETECTION                                           │
│  ├── Input: Image + modality-specific prompt                                   │
│  ├── Model: MedGemma 1.5 4B with LoRA                                          │
│  ├── Output: JSON with findings, bounding boxes, descriptions                  │
│  └── Example: {"findings": [{"class": "Glioma", "box": [200,150,600,500]}]}   │
│                                                                                  │
│  STAGE 3: ENHANCED EXPLAINABILITY                                               │
│  ├── Input: Image + prompt + findings                                          │
│  ├── Process: Generate gradient-weighted attention maps                        │
│  ├── Validation: Calculate ROI coverage (attention vs detected boxes)          │
│  ├── Output: Heatmap, quality score, clinical explanation                      │
│  └── Purpose: Verify model is "looking at" correct regions                     │
│                                                                                  │
│  STAGE 4: HEATMAP-GUIDED LOCALIZATION                                           │
│  ├── Input: Attention heatmap + VLM findings                                   │
│  ├── Process: Threshold heatmap, find contours, extract refined boxes          │
│  └── Output: Refined detection boxes from attention peaks                      │
│                                                                                  │
│  STAGE 5: PRECISE SEGMENTATION (SAM2)                                           │
│  ├── Input: Image + detection boxes                                            │
│  ├── Model: SAM2 Hiera-Tiny with LoRA                                          │
│  ├── Output: Binary masks [3, H, W] with IoU scores                           │
│  └── Purpose: Pixel-level delineation of pathologies                          │
│                                                                                  │
│  STAGE 6: CLINICAL VALIDATION                                                   │
│  ├── Input: Findings + modality + image quality                                │
│  ├── Checks: Anatomical plausibility, confidence thresholds                   │
│  └── Output: Validation status, warnings                                       │
│                                                                                  │
│  STAGE 7: REPORT GENERATION                                                     │
│  ├── Input: All pipeline outputs                                               │
│  ├── Format: Structured clinical report                                        │
│  └── Output: Findings, detections, segmentations, recommendations             │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Stage-by-Stage Training Requirements

| Stage | Training Script | Dataset | Epochs | Time |
|-------|-----------------|---------|--------|------|
| Screening | `run_curriculum.py` | Kaggle Pneumonia, Brain Tumor | 2 | ~30 min |
| Modality | `run_curriculum.py` | Brain Tumor Multimodal | 2 | ~20 min |
| Detection | `retrain_detection_optimized.py` | VinDr-CXR, NIH | 1-3 | ~14-24h |
| VQA | `run_curriculum.py` | SLAKE, VQA-RAD | 2 | ~1 hour |
| Segmentation | `train_sam2_medical.py` | BUSI, SLAKE | 5 | ~15 min |

---

## System Architecture

### High-Level Architecture

```
                                    ┌─────────────────────────────────────┐
                                    │         USER INTERFACE              │
                                    │   (Image Upload + Clinical Query)   │
                                    └───────────────┬─────────────────────┘
                                                    │
                                                    ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                              ORCHESTRATION LAYER (v2.1)                                │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                    │
│  │  State Manager  │◄──►│ Decision Engine │◄──►│  Tool Executor  │                    │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘                    │
└───────────────────────────────────────────────────────────────────────────────────────┘
                    │                                               │
                    ▼                                               ▼
    ┌───────────────────────────────┐           ┌───────────────────────────────┐
    │        REASONING MODULE        │           │      SEGMENTATION MODULE       │
    │         (MedGemma 4B)          │           │          (SAM 2)               │
    │  ┌─────────────────────────┐  │           │  ┌─────────────────────────┐  │
    │  │    Vision Encoder       │  │           │  │   Hiera Image Encoder   │  │
    │  │      (SigLIP)           │  │           │  │      (LoRA Adapted)     │  │
    │  ├─────────────────────────┤  │           │  ├─────────────────────────┤  │
    │  │  Multi-Modal Projector  │  │           │  │    Prompt Encoder       │  │
    │  ├─────────────────────────┤  │           │  ├─────────────────────────┤  │
    │  │   Language Model        │  │           │  │     Mask Decoder        │  │
    │  │   (Gemma 3 + LoRA)      │  │           │  └─────────────────────────┘  │
    │  └─────────────────────────┘  │           └───────────────────────────────┘
    └───────────────────────────────┘
```

### Core Module Structure

```
src/
├── __init__.py
├── orchestrator.py           # Main pipeline coordinator
├── medgemma_wrapper.py       # MedGemma VLM wrapper with modality detection
├── medsam_wrapper.py         # SAM2 segmentation wrapper
├── explainability.py         # Enhanced attention visualization with ROI validation
├── clinical_intelligence.py  # Quality assessment, validation, confidence levels
├── knowledge_base.py         # RAG knowledge retrieval
├── agent_state.py            # State management for multi-stage pipeline
├── config.py                 # Configuration management
├── prompts/
│   ├── __init__.py
│   └── modality_prompts.py   # Modality-specific detection prompts (22+16+11+8 classes)
├── data/
│   ├── factory.py            # Dataset loaders with modality support
│   ├── preprocess_images.py  # Image preprocessing for training
│   └── split_dataset.py      # Train/val/test splitting
├── train/
│   ├── train_medgemma.py     # MedGemma LoRA training
│   ├── train_sam2_medical.py # SAM2 LoRA training
│   └── curriculum_trainer.py # Multi-stage curriculum learning
└── eval/
    ├── evaluate_all_stages.py
    └── metrics.py
```

---

## Datasets

### Dataset Details

| Dataset | Samples | Task | Modality | Classes | Training Use |
|---------|---------|------|----------|---------|--------------|
| **VinDr-CXR** | 15,000 | Detection | Chest X-ray | 22 | Primary detection |
| **NIH ChestX-ray14** | 112,120 | Classification | Chest X-ray | 14 | Supplementary |
| **Brain Tumor MRI** | 5,712 | Classification/Detection | MRI | 4 | MRI detection |
| **Brain Tumor Multimodal** | 7,694 | Modality Detection | CT + MRI | 2 | Modality classification |
| **SLAKE** | 9,835 | VQA | Multi-modal | - | VQA reasoning |
| **VQA-RAD** | 1,798 | VQA | Multi-modal | - | VQA reasoning |
| **BUSI** | 780 | Segmentation | Ultrasound | 3 | SAM2 fine-tuning |

### Pathology Classes by Modality

**X-ray (22 classes)**:
```
Aortic enlargement, Atelectasis, Calcification, Cardiomegaly,
Clavicle fracture, Consolidation, Edema, Emphysema, Fibrosis,
ILD, Infiltration, Lung opacity, Mass, Nodule, Pleural effusion,
Pleural thickening, Pneumonia, Pneumothorax, Pulmonary fibrosis,
Rib fracture, Other lesion, No significant abnormality
```

**MRI (16 classes)**:
```
Glioma, Meningioma, Pituitary adenoma, Metastasis, Hemorrhage,
Infarction, White matter lesion, Hydrocephalus, Mass effect,
Midline shift, Edema, Cyst, Abscess, Brain atrophy, Other lesion,
No significant abnormality
```

**CT (11 classes)**:
```
Tumor, Hemorrhage, Infarction, Fracture, Calcification, Mass,
Effusion, Abscess, Edema, Other lesion, No significant abnormality
```

**Ultrasound (8 classes)**:
```
Benign mass, Malignant mass, Cyst, Calcification,
Abnormal echogenicity, Fluid collection, Other lesion,
No significant abnormality
```

---

## Results & Evaluation

### Multi-Stage Evaluation

| Stage | Primary Metric | Value | Secondary Metrics | Dataset |
|-------|---------------|-------|-------------------|---------|
| Screening | Accuracy | 89% | — | Kaggle Pneumonia |
| Modality | Accuracy | 100% | Per-class acc | Brain Tumor Multimodal |
| Detection | Overall Acc | — | P/R/F1, IoU, Confidence, Clinical Grade | VinDr-CXR |
| VQA | Accuracy | 66% | BLEU-1, ROUGE-1 | SLAKE |
| Segmentation | Dice | 0.878 | — | BUSI |

### Detection Metrics (v2.2)

| Metric | Description |
|--------|-------------|
| `image_accuracy` | % images correctly classified as normal/abnormal |
| `precision` / `recall` / `f1` | Standard detection metrics at IoU > 0.35 |
| `avg_confidence` | Per-image confidence score (0.0–1.0) |
| `clinical_grade` | A (≥0.7 clinical-ready), B (≥0.5), C (≥0.3), D (<0.3) |
| `per_class_accuracy` | P/R/F1 per pathology class (cardiomegaly, consolidation, etc.) |
| `confidence_distribution` | Count of high (≥0.7) / medium (0.4–0.7) / low (<0.4) images |

### Training Convergence

| Stage | Initial Loss | Final Loss | Improvement |
|-------|--------------|------------|-------------|
| Screening | 4.06 | 2.38 | -41.4% |
| Modality | 0.96 | 0.53 | -44.8% |
| Detection | 0.89 | 0.37 | -58.4% |
| VQA | 0.46 | 0.35 | -23.9% |
| SAM 2 | 0.35 | 0.16 | -54.3% |

### Explainability Validation

| Modality | Avg ROI Coverage | Avg Attention Quality |
|----------|------------------|----------------------|
| X-ray | 0.68 | 0.72 |
| MRI | 0.71 | 0.68 |
| CT | 0.65 | 0.70 |
| Ultrasound | 0.62 | 0.65 |

---

## Edge Deployment (Raspberry Pi 5)

### Target: Edge AI Prize ($5,000)

MedGamma supports edge deployment on Raspberry Pi 5 + AI HAT+2 (Hailo-10H 40 TOPS):

| Component | Performance |
|-----------|-------------|
| **MedGemma 4B (INT4)** | ~30-60s per image (CPU) |
| **SAM2-Tiny (Hailo-10H)** | ~25-50ms per mask (accelerated) |

### Edge Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RASPBERRY PI 5 (8GB)                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐    ┌──────────────────────────────┐   │
│  │   MedGemma 4B    │    │       SAM2-Tiny              │   │
│  │   (CPU + INT4)   │    │   (Hailo-10H Accelerated)    │   │
│  │                  │    │                              │   │
│  │  - Screening     │    │  - 40 TOPS INT8              │   │
│  │  - Detection     │───►│  - 25-50ms per mask          │   │
│  │  - Box output    │    │  - Real-time segmentation    │   │
│  │                  │    │                              │   │
│  │  ~30-60s/image   │    │  Near real-time capable      │   │
│  └──────────────────┘    └──────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Edge Export Commands

```bash
# Export quantized MedGemma
python edge/export_models.py --model medgemma --format int4

# Export SAM2 for Hailo
python edge/export_models.py --model sam2 --target hailo10h

# Run edge inference
python edge/edge_inference.py --image test.jpg
```

---

## Citation

```bibtex
@software{medgamma2026,
    title={GemSAM: An Agentic Framework for Explainable Multi-Modal Medical Image Analysis on Edge using MedGemma-1.5 and SAM2},
    author={[Soinik Ghosh]},
    year={2026},
    url={https://github.com/soinikghosh9/GemSAM},
    note={Kaggle MedGemma Impact Challenge Submission}
}
```

### Acknowledgments

- **MedGemma** - Google DeepMind
- **SAM 2** - Meta AI
- **VinDr-CXR** - VinBigData
- **NIH ChestX-ray14** - NIH Clinical Center

---

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

**Disclaimer**: This system is intended for research purposes only. It is not FDA-approved and should not be used for clinical diagnosis without expert oversight.

---

*Last Updated: February 28, 2026*
*MedGamma Framework Version: 2.2*

### **Anti-Memorization & Precision Engineering (v2.2)**

To achieve clinical-grade reliability, GemSAM v2.2 implements:

*   **ROI-Driven Box Refinement:** SAM2 masks are used to "snap" bounding boxes to literal pixel boundaries (Visual Grounding), ensuring precision.
*   **Anti-Memorization Prompts:** All detection prompts explicitly command the model to disregard memorized training templates and "prove" findings via direct pixel description.
*   **Coordinate Filtering (Hallucination Interceptor):** Automatically intercepts and discards known training-set "hallucinated" coordinates (e.g., [738, 459, 861, 521]).
*   **Binary Screening Mode:** Strict "HEALTHY/ABNORMAL" mode to stop the 30-second timeout failures. 
*   **Extended Evaluation Thresholds:** Complex JSON generation tasks now allow up to **180 seconds** for thorough processing.
