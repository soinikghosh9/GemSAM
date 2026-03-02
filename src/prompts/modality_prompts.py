"""
Modality-Specific Prompt Templates for MedGamma.

CRITICAL: This module centralizes all prompts to ensure consistency between
training and inference. All prompt changes should be made HERE and nowhere else.

Supported Modalities:
- xray: Chest X-ray, Radiograph
- mri: Brain MRI, Spine MRI
- ct: CT scan (head, chest, abdomen)
- ultrasound: Breast, Abdominal, Cardiac ultrasound

Usage:
    from src.prompts import get_detection_prompt
    prompt = get_detection_prompt("mri")
"""

# =============================================================================
# Supported Modalities
# =============================================================================

SUPPORTED_MODALITIES = ["xray", "mri", "ct", "ultrasound"]

# =============================================================================
# Detection Prompts - Task: Localize and classify pathologies
# =============================================================================

MODALITY_DETECTION_PROMPTS = {
    "xray": (
        "You are a senior radiologist analyzing this specific chest X-ray.\n"
        "Task: Disregard generalized templates. Instead, carefully examine the actual image pixels for TRUE visual abnormalities, regions of interest (ROI), and image-specific features.\n\n"
        "Valid classes to map your findings to: Aortic enlargement, Atelectasis, Calcification, Cardiomegaly, "
        "Consolidation, Edema, Emphysema, Infiltration, ILD, Lung Opacity, Nodule/Mass, "
        "Pleural effusion, Pleural thickening, Pneumothorax, Pulmonary fibrosis, Rib fracture, Other lesion.\n\n"
        "Guidelines:\n"
        "- Do not hallucinate. Prove what you see by describing exactly what the visual pixels show in THIS specific image.\n"
        "- Provide precise bounding box coordinates that tightly wrap the actual ROI in the image.\n"
        "- If the image is healthy, return class 'No significant abnormality' with box [0,0,0,0].\n\n"
        "Output ONLY valid JSON in this exact structure:\n"
        "{\"findings\": [{\"class\": \"<pathology>\", \"box\": [x1, y1, x2, y2], \"reasoning\": \"<describe the exact visual features/pixels you see in this image>\"}]}\n\n"
        "Coordinate system: [x1, y1, x2, y2] in 0-1000 scale where (0,0) is top-left corner.\n"
    ),
    "mri": (
        "You are a senior neuroradiologist analyzing this specific brain MRI.\n"
        "Task: Disregard generalized templates. Instead, carefully examine the actual image pixels for TRUE visual abnormalities, regions of interest (ROI), and image-specific features.\n\n"
        "Valid classes to map your findings to: Brain tumor, Glioma, Meningioma, Metastasis, Hemorrhage, Infarction, "
        "Hydrocephalus, Mass effect, Midline shift, White matter lesion, Edema, Cyst, Abscess, "
        "Brain atrophy, Other lesion.\n\n"
        "Guidelines:\n"
        "- Do not hallucinate. Prove what you see by describing exactly what the visual pixels show in THIS specific image.\n"
        "- Provide precise bounding box coordinates that tightly wrap the actual ROI in the image.\n"
        "- If the image is healthy, return class 'No significant abnormality' with box [0,0,0,0].\n\n"
        "Output ONLY valid JSON in this exact structure:\n"
        "{\"findings\": [{\"class\": \"<pathology>\", \"box\": [x1, y1, x2, y2], \"reasoning\": \"<describe the exact visual features/pixels you see in this image>\"}]}\n\n"
        "Coordinate system: [x1, y1, x2, y2] in 0-1000 scale where (0,0) is top-left corner.\n"
    ),
    "ct": (
        "You are a senior radiologist analyzing this specific CT scan.\n"
        "Task: Disregard generalized templates. Instead, carefully examine the actual image pixels for TRUE visual abnormalities, regions of interest (ROI), and image-specific features.\n\n"
        "Valid classes to map your findings to: Tumor, Hemorrhage, Infarction, Fracture, Calcification, Mass, Nodule, "
        "Consolidation, Effusion, Abscess, Edema, Lytic lesion, Other lesion.\n\n"
        "Guidelines:\n"
        "- Do not hallucinate. Prove what you see by describing exactly what the visual pixels show in THIS specific image.\n"
        "- Provide precise bounding box coordinates that tightly wrap the actual ROI in the image.\n"
        "- If the image is healthy, return class 'No significant abnormality' with box [0,0,0,0].\n\n"
        "Output ONLY valid JSON in this exact structure:\n"
        "{\"findings\": [{\"class\": \"<pathology>\", \"box\": [x1, y1, x2, y2], \"reasoning\": \"<describe the exact visual features/pixels you see in this image>\"}]}\n\n"
        "Coordinate system: [x1, y1, x2, y2] in 0-1000 scale where (0,0) is top-left corner.\n"
    ),
    "ultrasound": (
        "You are a senior sonographer analyzing this specific ultrasound image.\n"
        "Task: Disregard generalized templates. Instead, carefully examine the actual image pixels for TRUE visual abnormalities, regions of interest (ROI), and image-specific features.\n\n"
        "Valid classes to map your findings to: Benign mass, Malignant mass, Cyst, Calcification, "
        "Abnormal echogenicity, Fluid collection, Other lesion.\n\n"
        "Guidelines:\n"
        "- Do not hallucinate. Prove what you see by describing exactly what the visual pixels show in THIS specific image.\n"
        "- Provide precise bounding box coordinates that tightly wrap the actual ROI in the image.\n"
        "- If the image is healthy, return class 'No significant abnormality' with box [0,0,0,0].\n\n"
        "Output ONLY valid JSON in this exact structure:\n"
        "{\"findings\": [{\"class\": \"<pathology>\", \"box\": [x1, y1, x2, y2], \"reasoning\": \"<describe the exact visual features/pixels you see in this image>\"}]}\n\n"
        "Coordinate system: [x1, y1, x2, y2] in 0-1000 scale where (0,0) is top-left corner.\n"
    ),
}

# =============================================================================
# Screening Prompts - Task: Binary classification (normal vs abnormal)
# =============================================================================

MODALITY_SCREENING_PROMPTS = {
    "xray": (
        "Examine this chest X-ray and determine if it shows any abnormalities. "
        "Respond ONLY with the single word 'HEALTHY' if the image appears normal, or 'ABNORMAL' if you detect any pathology. Do not provide any reasoning or extra text."
    ),
    "mri": (
        "Examine this brain MRI and determine if it shows any abnormalities. "
        "Respond ONLY with the single word 'HEALTHY' if the image appears normal, or 'ABNORMAL' if you detect any pathology. Do not provide any reasoning or extra text."
    ),
    "ct": (
        "Examine this CT scan and determine if it shows any abnormalities. "
        "Respond ONLY with the single word 'HEALTHY' if the image appears normal, or 'ABNORMAL' if you detect any pathology. Do not provide any reasoning or extra text."
    ),
    "ultrasound": (
        "Examine this ultrasound image and determine if it shows any abnormalities. "
        "Respond ONLY with the single word 'HEALTHY' if the image appears normal, or 'ABNORMAL' if you detect any pathology. Do not provide any reasoning or extra text."
    ),
}

# =============================================================================
# VQA Context Prompts - Provides modality context for question answering
# =============================================================================

MODALITY_VQA_CONTEXT = {
    "xray": "This is a chest X-ray (radiograph). ",
    "mri": "This is a brain MRI (magnetic resonance imaging). ",
    "ct": "This is a CT scan (computed tomography). ",
    "ultrasound": "This is an ultrasound image. ",
}

# =============================================================================
# Modality-Specific Pathology Classes
# =============================================================================

MODALITY_CLASSES = {
    "xray": [
        "No significant abnormality",
        "Aortic enlargement",
        "Atelectasis",
        "Calcification",
        "Cardiomegaly",
        "Clavicle fracture",
        "Consolidation",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Infiltration",
        "ILD",  # Interstitial Lung Disease
        "Lung Opacity",
        "Mass",
        "Nodule",
        "Pleural effusion",
        "Pleural thickening",
        "Pneumonia",
        "Pneumothorax",
        "Pulmonary fibrosis",
        "Rib fracture",
        "Other lesion",
    ],
    "mri": [
        "No significant abnormality",
        "Glioma",
        "Meningioma",
        "Pituitary adenoma",
        "Metastasis",
        "Hemorrhage",
        "Infarction",
        "White matter lesion",
        "Hydrocephalus",
        "Mass effect",
        "Midline shift",
        "Edema",
        "Cyst",
        "Abscess",
        "Brain atrophy",
        "Other lesion",
    ],
    "ct": [
        "No significant abnormality",
        "Tumor",
        "Hemorrhage",
        "Infarction",
        "Fracture",
        "Calcification",
        "Mass",
        "Effusion",
        "Abscess",
        "Edema",
        "Other lesion",
    ],
    "ultrasound": [
        "No significant abnormality",
        "Benign mass",
        "Malignant mass",
        "Cyst",
        "Calcification",
        "Abnormal echogenicity",
        "Fluid collection",
        "Other lesion",
    ],
}

# =============================================================================
# Normal Findings - Clinical language for healthy/normal cases
# =============================================================================

MODALITY_NORMAL_FINDINGS = {
    "xray": {
        "findings": [{
            "class": "No significant abnormality",
            "box": [0, 0, 0, 0],
            "reasoning": "No acute cardiopulmonary abnormality identified. "
                        "Heart size is within normal limits. Lungs are clear bilaterally. "
                        "No pleural effusion or pneumothorax. Bony structures appear intact."
        }]
    },
    "mri": {
        "findings": [{
            "class": "No significant abnormality",
            "box": [0, 0, 0, 0],
            "reasoning": "No intracranial mass, hemorrhage, or acute infarction identified. "
                        "Ventricular system is normal in size and configuration. "
                        "No midline shift or mass effect. Brain parenchyma appears normal."
        }]
    },
    "ct": {
        "findings": [{
            "class": "No significant abnormality",
            "box": [0, 0, 0, 0],
            "reasoning": "No acute intracranial abnormality identified. "
                        "No evidence of hemorrhage, mass, or midline shift. "
                        "Ventricles and sulci are within normal limits for age."
        }]
    },
    "ultrasound": {
        "findings": [{
            "class": "No significant abnormality",
            "box": [0, 0, 0, 0],
            "reasoning": "No focal lesion or mass identified. "
                        "Normal echogenicity pattern throughout the examined region. "
                        "No suspicious findings requiring further evaluation."
        }]
    },
}

# =============================================================================
# Modality Detection Prompt
# =============================================================================

MODALITY_DETECTION_CLASSIFICATION_PROMPT = (
    "What imaging modality was used to acquire this medical image? "
    "Analyze the image characteristics (contrast, resolution, anatomical structures visible) "
    "and classify as one of: X-ray, MRI, CT, or Ultrasound. "
    "Respond with only the modality name."
)

# =============================================================================
# Helper Functions
# =============================================================================

def get_detection_prompt(modality: str) -> str:
    """
    Get the detection prompt for a specific imaging modality.

    Args:
        modality: One of 'xray', 'mri', 'ct', 'ultrasound'

    Returns:
        Detection prompt string for the specified modality
    """
    modality = modality.lower().strip()

    # Handle common aliases
    modality_aliases = {
        "x-ray": "xray",
        "xray": "xray",
        "radiograph": "xray",
        "chest x-ray": "xray",
        "brain mri": "mri",
        "mri": "mri",
        "magnetic resonance": "mri",
        "ct": "ct",
        "ct scan": "ct",
        "computed tomography": "ct",
        "us": "ultrasound",
        "ultrasound": "ultrasound",
        "sono": "ultrasound",
    }

    normalized = modality_aliases.get(modality, modality)
    return MODALITY_DETECTION_PROMPTS.get(normalized, MODALITY_DETECTION_PROMPTS["xray"])


def get_screening_prompt(modality: str) -> str:
    """Get the screening prompt for a specific imaging modality."""
    modality = modality.lower().strip()
    modality_aliases = {"x-ray": "xray", "ct scan": "ct", "us": "ultrasound"}
    normalized = modality_aliases.get(modality, modality)
    return MODALITY_SCREENING_PROMPTS.get(normalized, MODALITY_SCREENING_PROMPTS["xray"])


def get_vqa_context(modality: str) -> str:
    """Get the VQA context prefix for a specific imaging modality."""
    modality = modality.lower().strip()
    modality_aliases = {"x-ray": "xray", "ct scan": "ct", "us": "ultrasound"}
    normalized = modality_aliases.get(modality, modality)
    return MODALITY_VQA_CONTEXT.get(normalized, MODALITY_VQA_CONTEXT["xray"])


def get_modality_classes(modality: str) -> list:
    """Get the pathology classes for a specific imaging modality."""
    modality = modality.lower().strip()
    modality_aliases = {"x-ray": "xray", "ct scan": "ct", "us": "ultrasound"}
    normalized = modality_aliases.get(modality, modality)
    return MODALITY_CLASSES.get(normalized, MODALITY_CLASSES["xray"])


def get_normal_findings(modality: str) -> dict:
    """Get the normal findings JSON for a specific imaging modality."""
    modality = modality.lower().strip()
    modality_aliases = {"x-ray": "xray", "ct scan": "ct", "us": "ultrasound"}
    normalized = modality_aliases.get(modality, modality)
    return MODALITY_NORMAL_FINDINGS.get(normalized, MODALITY_NORMAL_FINDINGS["xray"])


def normalize_modality(modality_text: str) -> str:
    """
    Normalize modality text from model output to standard format.

    Args:
        modality_text: Raw modality text from model (e.g., "X-ray", "Brain MRI", "CT Scan")

    Returns:
        Normalized modality key (e.g., "xray", "mri", "ct")
    """
    text = modality_text.lower().strip()

    if any(x in text for x in ["x-ray", "xray", "radiograph", "chest"]):
        return "xray"
    elif any(x in text for x in ["mri", "magnetic", "brain mri"]):
        return "mri"
    elif any(x in text for x in ["ct", "computed", "tomography"]):
        return "ct"
    elif any(x in text for x in ["ultrasound", "us", "sono", "echo"]):
        return "ultrasound"
    else:
        return "xray"  # Default fallback
