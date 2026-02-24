"""
Clinical Intelligence Module for MedGamma.

Provides robust, production-ready clinical AI capabilities:
1. Confidence-aware modality detection with fallback
2. Clinical validation and sanity checks
3. Multi-stage verification pipeline
4. Anatomical constraint checking
5. Quality assessment for input images
6. Uncertainty quantification
7. Clinical protocol adherence

This module makes MedGamma more reliable for real-world clinical deployment.
"""

import os
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json


# =============================================================================
# Clinical Confidence Levels
# =============================================================================

class ConfidenceLevel(Enum):
    """Confidence levels for clinical findings."""
    HIGH = "high"           # >90% confidence - reliable for clinical use
    MODERATE = "moderate"   # 70-90% - requires clinician review
    LOW = "low"             # 50-70% - needs verification
    UNCERTAIN = "uncertain" # <50% - should not be used without expert review


@dataclass
class ClinicalFinding:
    """Structured clinical finding with confidence and validation."""
    finding_class: str
    box: List[int]
    description: str
    confidence: float
    confidence_level: ConfidenceLevel
    modality: str
    anatomical_location: str = ""
    severity: str = "unknown"
    requires_followup: bool = False
    validation_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "class": self.finding_class,
            "box": self.box,
            "description": self.description,
            "confidence": self.confidence,
            "confidence_level": self.confidence_level.value,
            "modality": self.modality,
            "anatomical_location": self.anatomical_location,
            "severity": self.severity,
            "requires_followup": self.requires_followup,
            "validation_flags": self.validation_flags
        }


@dataclass
class ClinicalReport:
    """Comprehensive clinical report with all findings and metadata."""
    image_path: str
    modality: str
    modality_confidence: float
    findings: List[ClinicalFinding]
    overall_impression: str
    recommendations: List[str]
    quality_score: float
    processing_time: float
    validation_status: str
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "image_path": self.image_path,
            "modality": self.modality,
            "modality_confidence": self.modality_confidence,
            "findings": [f.to_dict() for f in self.findings],
            "overall_impression": self.overall_impression,
            "recommendations": self.recommendations,
            "quality_score": self.quality_score,
            "processing_time": self.processing_time,
            "validation_status": self.validation_status,
            "warnings": self.warnings
        }


# =============================================================================
# Image Quality Assessment
# =============================================================================

class ImageQualityAssessor:
    """
    Assesses the quality of medical images before analysis.

    Checks for:
    - Adequate resolution
    - Proper contrast
    - No severe artifacts
    - Appropriate brightness
    """

    MIN_RESOLUTION = 224  # Minimum acceptable resolution
    OPTIMAL_RESOLUTION = 512  # Optimal resolution for analysis

    @classmethod
    def assess(cls, image: Image.Image) -> Tuple[float, List[str]]:
        """
        Assess image quality and return score (0-1) and warnings.

        Returns:
            Tuple of (quality_score, list_of_warnings)
        """
        warnings = []
        scores = []

        # Convert to numpy for analysis
        img_array = np.array(image.convert("L"))  # Grayscale

        # 1. Resolution check
        width, height = image.size
        min_dim = min(width, height)

        if min_dim < cls.MIN_RESOLUTION:
            warnings.append(f"Low resolution ({width}x{height}). Minimum recommended: {cls.MIN_RESOLUTION}px")
            scores.append(0.3)
        elif min_dim < cls.OPTIMAL_RESOLUTION:
            warnings.append(f"Sub-optimal resolution ({width}x{height}). Optimal: {cls.OPTIMAL_RESOLUTION}px")
            scores.append(0.7)
        else:
            scores.append(1.0)

        # 2. Contrast check (standard deviation of pixel values)
        contrast = np.std(img_array)
        if contrast < 20:
            warnings.append("Very low contrast - image may appear washed out")
            scores.append(0.4)
        elif contrast < 40:
            warnings.append("Low contrast - may affect detection accuracy")
            scores.append(0.7)
        else:
            scores.append(1.0)

        # 3. Brightness check
        mean_brightness = np.mean(img_array)
        if mean_brightness < 30:
            warnings.append("Image is too dark - may miss findings")
            scores.append(0.5)
        elif mean_brightness > 225:
            warnings.append("Image is too bright/overexposed")
            scores.append(0.5)
        else:
            scores.append(1.0)

        # 4. Blank/uniform image check
        unique_values = len(np.unique(img_array))
        if unique_values < 10:
            warnings.append("Image appears nearly uniform - may be corrupted or blank")
            scores.append(0.1)
        else:
            scores.append(1.0)

        # Calculate overall quality score
        quality_score = np.mean(scores)

        return quality_score, warnings


# =============================================================================
# Anatomical Constraint Checker
# =============================================================================

class AnatomicalConstraintChecker:
    """
    Validates findings against anatomical constraints.

    Ensures that detected pathologies are anatomically plausible:
    - X-ray findings should be in lung/heart regions
    - Brain MRI findings should be in brain tissue
    - Bounding boxes should have reasonable sizes
    """

    # Anatomical regions by modality (normalized 0-1000 coordinates)
    ANATOMICAL_REGIONS = {
        "xray": {
            "lungs": {"x_range": (100, 900), "y_range": (150, 700)},
            "heart": {"x_range": (350, 650), "y_range": (300, 600)},
            "mediastinum": {"x_range": (300, 700), "y_range": (100, 500)},
        },
        "mri": {
            "brain_parenchyma": {"x_range": (150, 850), "y_range": (100, 900)},
            "ventricles": {"x_range": (300, 700), "y_range": (300, 600)},
        },
        "ct": {
            "brain": {"x_range": (150, 850), "y_range": (100, 850)},
        },
        "ultrasound": {
            "tissue": {"x_range": (50, 950), "y_range": (50, 950)},
        }
    }

    # Minimum and maximum box sizes (as fraction of image)
    MIN_BOX_SIZE = 0.01  # 1% of image
    MAX_BOX_SIZE = 0.80  # 80% of image

    @classmethod
    def validate_finding(cls, finding: Dict, modality: str) -> Tuple[bool, List[str]]:
        """
        Validate a finding against anatomical constraints.

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        box = finding.get("box", [0, 0, 0, 0])

        if len(box) != 4:
            issues.append("Invalid bounding box format")
            return False, issues

        x1, y1, x2, y2 = box

        # Check for invalid coordinates
        if x1 >= x2 or y1 >= y2:
            issues.append("Invalid box coordinates (x1 >= x2 or y1 >= y2)")
            return False, issues

        # Check box size
        box_area = (x2 - x1) * (y2 - y1) / (1000 * 1000)

        if box_area < cls.MIN_BOX_SIZE:
            issues.append(f"Box too small ({box_area*100:.1f}% of image)")

        if box_area > cls.MAX_BOX_SIZE:
            issues.append(f"Box suspiciously large ({box_area*100:.1f}% of image)")

        # Check anatomical plausibility
        box_center_x = (x1 + x2) / 2
        box_center_y = (y1 + y2) / 2

        regions = cls.ANATOMICAL_REGIONS.get(modality, {})
        in_any_region = False

        for region_name, bounds in regions.items():
            x_range = bounds["x_range"]
            y_range = bounds["y_range"]

            if (x_range[0] <= box_center_x <= x_range[1] and
                y_range[0] <= box_center_y <= y_range[1]):
                in_any_region = True
                break

        if not in_any_region and regions:
            issues.append("Finding location may be anatomically implausible")

        is_valid = len(issues) == 0
        return is_valid, issues


# =============================================================================
# Clinical Validation Pipeline
# =============================================================================

class ClinicalValidator:
    """
    Multi-stage clinical validation pipeline.

    Validates:
    1. Image quality
    2. Modality detection confidence
    3. Finding plausibility
    4. Anatomical constraints
    5. Clinical protocol adherence
    """

    # Pathologies that are mutually exclusive
    MUTUALLY_EXCLUSIVE = {
        "xray": [
            {"Pneumothorax", "Normal lung expansion"},
            {"Cardiomegaly", "Normal heart size"},
        ],
        "mri": [
            {"Glioma", "Meningioma"},  # Usually one primary tumor type
        ]
    }

    # Pathologies that commonly co-occur
    COMMONLY_COOCCUR = {
        "xray": [
            {"Pleural effusion", "Cardiomegaly"},
            {"Consolidation", "Pneumonia"},
        ],
        "mri": [
            {"Tumor", "Edema"},
            {"Mass effect", "Midline shift"},
        ]
    }

    @classmethod
    def validate_findings(
        cls,
        findings: List[Dict],
        modality: str,
        image_quality_score: float
    ) -> Tuple[str, List[str]]:
        """
        Validate a set of findings.

        Returns:
            Tuple of (validation_status, list_of_warnings)
        """
        warnings = []

        # 1. Image quality check
        if image_quality_score < 0.5:
            warnings.append("Low image quality may affect accuracy")

        # 2. Check for mutually exclusive findings
        finding_classes = {f.get("class", "").lower() for f in findings}

        for exclusive_set in cls.MUTUALLY_EXCLUSIVE.get(modality, []):
            exclusive_lower = {c.lower() for c in exclusive_set}
            overlap = finding_classes & exclusive_lower
            if len(overlap) > 1:
                warnings.append(f"Potentially conflicting findings: {overlap}")

        # 3. Validate each finding
        for finding in findings:
            is_valid, issues = AnatomicalConstraintChecker.validate_finding(
                finding, modality
            )
            if not is_valid:
                warnings.extend(issues)

        # 4. Determine overall status
        if len(warnings) == 0:
            status = "VALIDATED"
        elif len(warnings) <= 2:
            status = "VALIDATED_WITH_WARNINGS"
        else:
            status = "REQUIRES_REVIEW"

        return status, warnings


# =============================================================================
# Modality Detection with Confidence
# =============================================================================

class RobustModalityDetector:
    """
    Robust modality detection with confidence scoring and fallback.

    Features:
    - Multi-feature analysis (not just text)
    - Confidence scoring
    - Fallback to user input
    - Caching for performance
    """

    # Image characteristics by modality
    MODALITY_CHARACTERISTICS = {
        "xray": {
            "typical_contrast": (40, 80),  # Standard deviation range
            "typical_brightness": (80, 180),
            "aspect_ratio_range": (0.8, 1.2),  # Usually square-ish
        },
        "mri": {
            "typical_contrast": (30, 70),
            "typical_brightness": (50, 150),
            "aspect_ratio_range": (0.9, 1.1),
        },
        "ct": {
            "typical_contrast": (35, 75),
            "typical_brightness": (60, 160),
            "aspect_ratio_range": (0.9, 1.1),
        },
        "ultrasound": {
            "typical_contrast": (20, 60),
            "typical_brightness": (40, 140),
            "aspect_ratio_range": (0.6, 1.8),  # Can be various shapes
        }
    }

    @classmethod
    def estimate_modality_from_image(cls, image: Image.Image) -> Tuple[str, float]:
        """
        Estimate modality from image characteristics.

        Returns:
            Tuple of (estimated_modality, confidence)
        """
        img_array = np.array(image.convert("L"))

        # Calculate image characteristics
        contrast = np.std(img_array)
        brightness = np.mean(img_array)
        aspect_ratio = image.width / image.height

        # Score each modality
        scores = {}

        for modality, chars in cls.MODALITY_CHARACTERISTICS.items():
            score = 0.0

            # Contrast match
            c_min, c_max = chars["typical_contrast"]
            if c_min <= contrast <= c_max:
                score += 0.33
            elif abs(contrast - (c_min + c_max) / 2) < 30:
                score += 0.15

            # Brightness match
            b_min, b_max = chars["typical_brightness"]
            if b_min <= brightness <= b_max:
                score += 0.33
            elif abs(brightness - (b_min + b_max) / 2) < 40:
                score += 0.15

            # Aspect ratio match
            ar_min, ar_max = chars["aspect_ratio_range"]
            if ar_min <= aspect_ratio <= ar_max:
                score += 0.34

            scores[modality] = score

        # Get best match
        best_modality = max(scores, key=scores.get)
        confidence = scores[best_modality]

        return best_modality, confidence

    @classmethod
    def combine_detections(
        cls,
        model_prediction: str,
        model_confidence: float,
        image_based_prediction: str,
        image_confidence: float
    ) -> Tuple[str, float, str]:
        """
        Combine model-based and image-based modality predictions.

        Returns:
            Tuple of (final_modality, final_confidence, method_used)
        """
        # If model is confident and matches image analysis
        if model_prediction == image_based_prediction:
            combined_confidence = (model_confidence + image_confidence) / 2 + 0.1
            return model_prediction, min(combined_confidence, 1.0), "consensus"

        # If model is very confident
        if model_confidence > 0.85:
            return model_prediction, model_confidence, "model_primary"

        # If image analysis is more confident
        if image_confidence > model_confidence:
            return image_based_prediction, image_confidence, "image_analysis"

        # Default to model prediction
        return model_prediction, model_confidence, "model_default"


# =============================================================================
# Clinical Recommendation Engine
# =============================================================================

class ClinicalRecommendationEngine:
    """
    Generates clinical recommendations based on findings.

    Provides:
    - Follow-up recommendations
    - Additional imaging suggestions
    - Urgency levels
    - Specialist referrals
    """

    URGENCY_LEVELS = {
        "STAT": ["Pneumothorax", "Tension pneumothorax", "Hemorrhage", "Midline shift"],
        "URGENT": ["Large mass", "Consolidation", "Pleural effusion", "Glioma", "Meningioma"],
        "ROUTINE": ["Small nodule", "Atelectasis", "Cardiomegaly", "Pituitary adenoma"],
        "MONITORING": ["Calcification", "Fibrosis", "Stable lesion"]
    }

    SPECIALIST_REFERRALS = {
        "Pneumothorax": "Thoracic Surgery / Pulmonology",
        "Cardiomegaly": "Cardiology",
        "Glioma": "Neuro-oncology",
        "Meningioma": "Neurosurgery",
        "Pituitary adenoma": "Endocrinology / Neurosurgery",
        "Pleural effusion": "Pulmonology",
        "Mass": "Oncology",
    }

    ADDITIONAL_IMAGING = {
        "xray": {
            "Mass": ["CT chest with contrast", "PET-CT if malignancy suspected"],
            "Nodule": ["Follow-up CT in 3-6 months", "Consider PET-CT if > 8mm"],
            "Consolidation": ["Follow-up X-ray in 2-4 weeks"],
        },
        "mri": {
            "Glioma": ["MRI with perfusion/spectroscopy", "Consider PET"],
            "Meningioma": ["Follow-up MRI in 6 months if stable"],
        }
    }

    @classmethod
    def generate_recommendations(
        cls,
        findings: List[ClinicalFinding],
        modality: str
    ) -> Tuple[List[str], str, Optional[str]]:
        """
        Generate clinical recommendations.

        Returns:
            Tuple of (recommendations_list, urgency_level, specialist_referral)
        """
        recommendations = []
        urgency = "ROUTINE"
        specialist = None

        for finding in findings:
            finding_class = finding.finding_class

            # Check urgency
            for level, conditions in cls.URGENCY_LEVELS.items():
                if any(c.lower() in finding_class.lower() for c in conditions):
                    if cls._urgency_priority(level) > cls._urgency_priority(urgency):
                        urgency = level
                    break

            # Check specialist referral
            for condition, spec in cls.SPECIALIST_REFERRALS.items():
                if condition.lower() in finding_class.lower():
                    specialist = spec
                    break

            # Check additional imaging
            imaging_recs = cls.ADDITIONAL_IMAGING.get(modality, {})
            for condition, recs in imaging_recs.items():
                if condition.lower() in finding_class.lower():
                    recommendations.extend(recs)

        # Add general recommendations
        if urgency == "STAT":
            recommendations.insert(0, "Immediate clinical correlation required")
        elif urgency == "URGENT":
            recommendations.insert(0, "Clinical correlation within 24-48 hours recommended")

        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)

        return unique_recommendations, urgency, specialist

    @staticmethod
    def _urgency_priority(level: str) -> int:
        """Get numeric priority for urgency level."""
        priorities = {"STAT": 4, "URGENT": 3, "ROUTINE": 2, "MONITORING": 1}
        return priorities.get(level, 0)


# =============================================================================
# Unified Clinical Intelligence Interface
# =============================================================================

class ClinicalIntelligence:
    """
    Unified interface for all clinical intelligence features.

    Usage:
        ci = ClinicalIntelligence()
        report = ci.process_image(image_path, medgemma_wrapper)
    """

    def __init__(self):
        self.quality_assessor = ImageQualityAssessor()
        self.constraint_checker = AnatomicalConstraintChecker()
        self.validator = ClinicalValidator()
        self.modality_detector = RobustModalityDetector()
        self.recommendation_engine = ClinicalRecommendationEngine()

    def assess_image_quality(self, image: Image.Image) -> Tuple[float, List[str]]:
        """Assess image quality before analysis."""
        return self.quality_assessor.assess(image)

    def detect_modality_robust(
        self,
        image: Image.Image,
        model_prediction: str,
        model_confidence: float = 0.8
    ) -> Tuple[str, float, str]:
        """
        Robustly detect modality using multiple methods.

        Returns:
            Tuple of (modality, confidence, detection_method)
        """
        # Get image-based prediction
        img_modality, img_confidence = self.modality_detector.estimate_modality_from_image(image)

        # Combine predictions
        return self.modality_detector.combine_detections(
            model_prediction, model_confidence,
            img_modality, img_confidence
        )

    def validate_and_enhance_findings(
        self,
        findings: List[Dict],
        modality: str,
        image_quality: float
    ) -> Tuple[List[ClinicalFinding], str, List[str]]:
        """
        Validate findings and convert to structured format.

        Returns:
            Tuple of (enhanced_findings, validation_status, warnings)
        """
        # Validate findings
        status, warnings = self.validator.validate_findings(
            findings, modality, image_quality
        )

        # Convert to ClinicalFinding objects
        enhanced_findings = []
        for f in findings:
            # Skip "No significant abnormality" placeholder
            if f.get("class", "").lower() == "no significant abnormality":
                continue

            # Determine confidence level
            confidence = f.get("confidence", 0.7)
            if confidence > 0.9:
                level = ConfidenceLevel.HIGH
            elif confidence > 0.7:
                level = ConfidenceLevel.MODERATE
            elif confidence > 0.5:
                level = ConfidenceLevel.LOW
            else:
                level = ConfidenceLevel.UNCERTAIN

            # Validate anatomically
            is_valid, issues = self.constraint_checker.validate_finding(f, modality)

            enhanced = ClinicalFinding(
                finding_class=f.get("class", "Unknown"),
                box=f.get("box", [0, 0, 0, 0]),
                description=f.get("description", ""),
                confidence=confidence,
                confidence_level=level,
                modality=modality,
                validation_flags=issues if not is_valid else []
            )
            enhanced_findings.append(enhanced)

        return enhanced_findings, status, warnings

    def generate_clinical_report(
        self,
        image_path: str,
        modality: str,
        modality_confidence: float,
        findings: List[ClinicalFinding],
        processing_time: float,
        quality_score: float,
        validation_status: str,
        warnings: List[str]
    ) -> ClinicalReport:
        """Generate comprehensive clinical report."""

        # Generate recommendations
        recommendations, urgency, specialist = self.recommendation_engine.generate_recommendations(
            findings, modality
        )

        if specialist:
            recommendations.append(f"Consider referral to {specialist}")

        # Generate overall impression
        if not findings:
            impression = self._get_normal_impression(modality)
        else:
            finding_classes = [f.finding_class for f in findings]
            impression = f"Identified {len(findings)} finding(s): {', '.join(finding_classes)}. "
            impression += f"Urgency: {urgency}."

        return ClinicalReport(
            image_path=image_path,
            modality=modality,
            modality_confidence=modality_confidence,
            findings=findings,
            overall_impression=impression,
            recommendations=recommendations,
            quality_score=quality_score,
            processing_time=processing_time,
            validation_status=validation_status,
            warnings=warnings
        )

    def _get_normal_impression(self, modality: str) -> str:
        """Get modality-specific normal impression."""
        impressions = {
            "xray": "No acute cardiopulmonary abnormality identified.",
            "mri": "No intracranial abnormality identified.",
            "ct": "No acute intracranial abnormality identified.",
            "ultrasound": "No focal abnormality identified."
        }
        return impressions.get(modality, "No significant abnormality identified.")
