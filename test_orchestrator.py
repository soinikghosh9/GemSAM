from src.orchestrator import GemSAMOrchestrator
import os

# Test single image directly against the system
orch = GemSAMOrchestrator()
image_path = "d:/MedGamma/medical_data/vinbigdata-chest-xray/train_448/9a5094b2563a1ef3ff50dc5c7ff71345.jpg"

print(f"Testing Orchestrator with image: {image_path}")
state = orch.run_pipeline(image_path, "chest x-ray, evaluate for abnormalities")

print("\n--- RAW VLM RESPONSE ---")
print(state.get("current_thought", "No raw thought recorded"))
print("------------------------\n")


print("Detections:")
for d in state["detections"]:
    print(d)

print("Segmentations:")
for s in state["segmentations"]:
    print(f"Verified: {s.get('verified', False)}, Confidence: {s.get('clinical_confidence', 0.0)}")
