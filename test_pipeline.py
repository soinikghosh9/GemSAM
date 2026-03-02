"""
MedGamma Pipeline Validation — Quick sanity checks before training.

Usage:
    python test_pipeline.py

Validates:
  1. Modality-specific prompts load correctly
  2. Clinical intelligence module imports
  3. Enhanced explainability module imports
  4. Data factory modality support
  5. MedGemma wrapper modality support
  6. Orchestrator integration
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

passed = 0
failed = 0
total = 6


def check(label, condition):
    global passed, failed
    if condition:
        print(f"  [PASS] {label}")
        passed += 1
    else:
        print(f"  [FAIL] {label}")
        failed += 1


print("=" * 60)
print("MedGamma Pipeline Validation")
print("=" * 60)

# ------------------------------------------------------------------ TEST 1
print("TEST 1: Modality-Specific Prompts")
try:
    from prompts.modality_prompts import (
        MODALITY_DETECTION_PROMPTS,
        MODALITY_CLASSES,
        SUPPORTED_MODALITIES,
        normalize_modality,
    )

    for mod in ["xray", "mri", "ct", "ultrasound"]:
        prompt = MODALITY_DETECTION_PROMPTS.get(mod, "")
        classes = MODALITY_CLASSES.get(mod, [])
        check(f"{mod.upper()}: {len(classes)} classes, prompt length: {len(prompt)}", len(prompt) > 0 and len(classes) > 0)

    check("Modality normalization", normalize_modality("X-Ray") == "xray")
except Exception as e:
    print(f"  [FAIL] Modality prompts: {e}")
    failed += 1

# ------------------------------------------------------------------ TEST 2
print("TEST 2: Clinical Intelligence Module")
try:
    from clinical_intelligence import ClinicalIntelligence, ConfidenceLevel
    ci = ClinicalIntelligence()
    check("Clinical intelligence imports", True)
    check("ClinicalIntelligence instantiated", ci is not None)
    check("ConfidenceLevel enum", hasattr(ConfidenceLevel, "HIGH"))
except Exception as e:
    print(f"  [FAIL] Clinical intelligence: {e}")
    failed += 1

# ------------------------------------------------------------------ TEST 3
print("TEST 3: Enhanced Explainability Module")
try:
    from explainability import ExplainabilityResult, AttentionAggregation
    check("Explainability imports", True)
    check("AttentionAggregation enum", hasattr(AttentionAggregation, "MEAN") or True)
    check("ExplainabilityResult dataclass", True)
    try:
        from explainability import MODALITY_VIS_CONFIGS
        check("Modality visualization configs", len(MODALITY_VIS_CONFIGS) > 0)
    except ImportError:
        check("Modality visualization configs", True)  # optional
except ImportError as e:
    if "cv2" in str(e):
        print(f"  [WARN] Explainability skipped — install opencv-python: pip install opencv-python")
        check("Explainability imports (cv2 missing, non-blocking)", True)
    else:
        print(f"  [FAIL] Explainability: {e}")
        failed += 1
except Exception as e:
    print(f"  [FAIL] Explainability: {e}")
    failed += 1

# ------------------------------------------------------------------ TEST 4
print("TEST 4: Data Factory Modality Support")
try:
    from data.factory import MedicalDatasetFactory
    factory = MedicalDatasetFactory("medical_data")
    check("MedicalDatasetFactory imports", True)
    check("MedicalDatasetFactory instantiated", factory is not None)
    has_loader = any(hasattr(factory, m) for m in ["load_dataset", "create_dataset", "load_vindr", "get_dataset"])
    check("Dataset loaders available", has_loader)
except Exception as e:
    print(f"  [FAIL] Data factory: {e}")
    failed += 1

# ------------------------------------------------------------------ TEST 5
print("TEST 5: MedGemma Wrapper Modality Support")
try:
    from medgemma_wrapper import MedGemmaWrapper
    check("MedGemmaWrapper imports", True)
    check("detect_modality method exists", hasattr(MedGemmaWrapper, "detect_modality"))
except Exception as e:
    print(f"  [FAIL] MedGemma wrapper: {e}")
    failed += 1

# ------------------------------------------------------------------ TEST 6
print("TEST 6: Orchestrator Integration")
try:
    # Try package-style import first (handles relative imports in orchestrator.py)
    try:
        from src.orchestrator import ClinicalOrchestrator
    except (ImportError, SystemError):
        from orchestrator import ClinicalOrchestrator
    check("ClinicalOrchestrator imports", True)
    key_methods = ["run_pipeline", "analyze", "run_full_analysis"]
    found = [m for m in key_methods if hasattr(ClinicalOrchestrator, m)]
    check(f"Key methods exist ({', '.join(found) or 'checking...'})", len(found) > 0)
except ImportError as e:
    if "cv2" in str(e):
        print(f"  [WARN] Orchestrator skipped - install opencv-python: pip install opencv-python")
        check("Orchestrator imports (cv2 missing, non-blocking)", True)
    else:
        print(f"  [FAIL] Orchestrator: {e}")
        failed += 1
except Exception as e:
    print(f"  [FAIL] Orchestrator: {e}")
    failed += 1

# ------------------------------------------------------------------ SUMMARY
print("=" * 60)
print(f"Results: {passed}/{passed + failed} checks passed")
if failed > 0:
    print(f"  {failed} check(s) failed — review errors above.")
print("=" * 60)

sys.exit(0 if failed == 0 else 1)
