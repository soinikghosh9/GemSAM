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
        "Analyze this chest X-ray for pathological findings. "
        "For each finding, provide the class name and bounding box coordinates. "
        "Additionally, provide a comprehensive clinical description of the anomaly, its severity, and its exact anatomical location. "
        "Output in JSON format: {\"findings\": [{\"class\": \"...\", \"box\": [x1,y1,x2,y2], \"description\": \"...\"}]}"
    ),
    "mri": (
        "Analyze this brain MRI for pathological findings. "
        "Look for: brain tumors (glioma, meningioma, pituitary adenoma), metastases, "
        "intracranial hemorrhage, white matter lesions, cerebral infarction, hydrocephalus, "
        "mass effect, midline shift, or any other structural abnormalities. "
        "For each finding, provide the class name and bounding box coordinates [x1,y1,x2,y2] normalized to 0-1000. "
        "Additionally, provide a comprehensive clinical description including tumor grade (if applicable), "
        "location (frontal, temporal, parietal, occipital, cerebellum), and surrounding edema assessment. "
        'Output in JSON format: {"findings": [{"class": "...", "box": [x1,y1,x2,y2], "description": "..."}]}'
    ),
    "ct": (
        "Analyze this CT scan for pathological findings. "
        "Look for: tumors, masses, hemorrhage (epidural, subdural, intracerebral, subarachnoid), "
        "fractures, calcifications, abnormal densities, effusions, or structural abnormalities. "
        "For each finding, provide the class name and bounding box coordinates [x1,y1,x2,y2] normalized to 0-1000. "
        "Additionally, provide a comprehensive clinical description including Hounsfield unit assessment "
        "(if relevant), size, and location. "
        'Output in JSON format: {"findings": [{"class": "...", "box": [x1,y1,x2,y2], "description": "..."}]}'
    ),
    "ultrasound": (
        "Analyze this ultrasound image for pathological findings. "
        "Look for: masses (solid vs cystic), abnormal echogenicity patterns, calcifications, "
        "fluid collections, structural abnormalities, or any suspicious lesions. "
        "For each finding, provide the class name and bounding box coordinates [x1,y1,x2,y2] normalized to 0-1000. "
        "Additionally, provide a comprehensive clinical description including echogenicity (hypoechoic, "
        "hyperechoic, anechoic, mixed), margins (well-defined, irregular), and size estimation. "
        'Output in JSON format: {"findings": [{"class": "...", "box": [x1,y1,x2,y2], "description": "..."}]}'
    ),
}

# =============================================================================
# Screening Prompts - Task: Binary classification (normal vs abnormal)
# =============================================================================

MODALITY_SCREENING_PROMPTS = {
    "xray": (
        "Examine this chest X-ray and determine if it shows any abnormalities. "
        "Consider cardiac silhouette, lung fields, pleural spaces, bony structures, and soft tissues. "
        "Respond with 'HEALTHY' if the image appears normal, or 'ABNORMAL' if you detect any pathology."
    ),
    "mri": (
        "Examine this brain MRI and determine if it shows any abnormalities. "
        "Consider brain parenchyma, ventricular system, extra-axial spaces, and any focal lesions. "
        "Respond with 'HEALTHY' if the image appears normal, or 'ABNORMAL' if you detect any pathology such as tumors, lesions, or structural abnormalities."
    ),
    "ct": (
        "Examine this CT scan and determine if it shows any abnormalities. "
        "Consider tissue densities, structural integrity, and any focal abnormalities. "
        "Respond with 'HEALTHY' if the image appears normal, or 'ABNORMAL' if you detect any pathology."
    ),
    "ultrasound": (
        "Examine this ultrasound image and determine if it shows any abnormalities. "
        "Consider echogenicity patterns, structural integrity, and any focal lesions. "
        "Respond with 'HEALTHY' if the image appears normal, or 'ABNORMAL' if you detect any pathology."
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
        "Lung opacity",
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
            "description": "No acute cardiopulmonary abnormality identified. "
                          "Heart size is within normal limits. Lungs are clear bilaterally. "
                          "No pleural effusion or pneumothorax. Bony structures appear intact."
        }]
    },
    "mri": {
        "findings": [{
            "class": "No significant abnormality",
            "box": [0, 0, 0, 0],
            "description": "No intracranial mass, hemorrhage, or acute infarction identified. "
                          "Ventricular system is normal in size and configuration. "
                          "No midline shift or mass effect. Brain parenchyma appears normal."
        }]
    },
    "ct": {
        "findings": [{
            "class": "No significant abnormality",
            "box": [0, 0, 0, 0],
            "description": "No acute intracranial abnormality identified. "
                          "No evidence of hemorrhage, mass, or midline shift. "
                          "Ventricles and sulci are within normal limits for age."
        }]
    },
    "ultrasound": {
        "findings": [{
            "class": "No significant abnormality",
            "box": [0, 0, 0, 0],
            "description": "No focal lesion or mass identified. "
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
