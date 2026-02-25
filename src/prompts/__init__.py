"""
MedGamma Modality-Aware Prompts Module.

This module provides centralized, modality-specific prompt templates
for all stages of the clinical AI pipeline.
"""

from .modality_prompts import (
    MODALITY_DETECTION_PROMPTS,
    MODALITY_SCREENING_PROMPTS,
    MODALITY_VQA_CONTEXT,
    MODALITY_CLASSES,
    get_detection_prompt,
    get_screening_prompt,
    get_vqa_context,
    get_modality_classes,
    SUPPORTED_MODALITIES,
)

__all__ = [
    "MODALITY_DETECTION_PROMPTS",
    "MODALITY_SCREENING_PROMPTS",
    "MODALITY_VQA_CONTEXT",
    "MODALITY_CLASSES",
    "get_detection_prompt",
    "get_screening_prompt",
    "get_vqa_context",
    "get_modality_classes",
    "SUPPORTED_MODALITIES",
]
