"""
Custom Stopping Criteria for MedGemma Generation.

Prevents infinite loops, repeated outputs, and ensures valid JSON output.
"""

import time
import re
import torch
from transformers import StoppingCriteria, StoppingCriteriaList


class StopStringCriteria(StoppingCriteria):
    """
    Stop generation when a specific string is detected in the output.
    Essential for Gemma models which use <end_of_turn> token.

    IMPORTANT: Only checks GENERATED tokens, not the input prompt.
    The input_len parameter tracks where the prompt ends.
    """

    def __init__(self, tokenizer, stop_strings: list = None, input_len: int = 0):
        self.tokenizer = tokenizer
        # Expanded stop strings for diverse model versions and formats
        self.stop_strings = stop_strings or ["<end_of_turn>", "```\n\n", "</s>", "<|endoftext|>", "[END]", "\n\n\n"]
        self.input_len = input_len  # Track where prompt ends
        # Pre-encode stop strings for faster matching
        self.stop_token_ids = []
        for s in self.stop_strings:
            tokens = tokenizer.encode(s, add_special_tokens=False)
            if tokens:
                self.stop_token_ids.append(tokens)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        current_len = input_ids.shape[1]

        # Calculate how many tokens have been generated (exclude prompt)
        generated_len = current_len - self.input_len

        # Need at least a few generated tokens before checking
        if generated_len < 3:
            return False

        # Only check GENERATED tokens (after the prompt)
        # Check last 20 generated tokens, or all generated if fewer
        check_start = max(self.input_len, current_len - 20)
        generated_tokens = input_ids[0, check_start:]
        decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)

        for stop_str in self.stop_strings:
            if stop_str in decoded:
                return True

        return False

    def set_input_len(self, input_len: int):
        """Set the input length after creation (for reuse)."""
        self.input_len = input_len


class JSONCompleteStoppingCriteria(StoppingCriteria):
    """
    Stop generation when a complete JSON object is detected.
    Tracks brace/bracket matching to determine when JSON is complete.

    IMPORTANT: Only checks GENERATED tokens, not the input prompt.
    The input_len parameter tracks where the prompt ends.
    """

    def __init__(self, tokenizer, max_objects: int = 1, input_len: int = 0):
        self.tokenizer = tokenizer
        self.max_objects = max_objects
        self.input_len = input_len  # Track where prompt ends
        self.last_check_len = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        current_len = input_ids.shape[1]

        # Calculate how many tokens have been generated (exclude prompt)
        generated_len = current_len - self.input_len

        # Need at least 10 generated tokens before checking for complete JSON
        # (a minimal JSON like {"a":1} is ~5-10 tokens)
        if generated_len < 10:
            return False

        # Only check every 5 NEW tokens for efficiency
        if generated_len - self.last_check_len < 5:
            return False
        self.last_check_len = generated_len

        # Only check GENERATED tokens (after prompt)
        # Use window within generated tokens only
        generated_tokens = input_ids[0, self.input_len:]
        decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Strip and check if it's a valid JSON start/end
        stripped = decoded.strip()
        if not stripped.startswith('{') and not stripped.startswith('['):
            return False

        # Look for complete JSON objects in generated text only
        json_complete = self._count_complete_json(decoded)

        return json_complete >= self.max_objects
    
    def _count_complete_json(self, text: str) -> int:
        """Count number of complete JSON objects in text."""
        count = 0
        brace_depth = 0
        bracket_depth = 0
        in_string = False
        escape = False
        found_start = False
        
        for char in text:
            if escape:
                escape = False
                continue
            
            if char == '\\':
                escape = True
                continue
            
            if char == '"' and not escape:
                in_string = not in_string
                continue
            
            if in_string:
                continue
            
            if char == '{':
                if brace_depth == 0 and bracket_depth == 0:
                    found_start = True
                brace_depth += 1
            elif char == '}':
                brace_depth = max(0, brace_depth - 1)
                if found_start and brace_depth == 0 and bracket_depth == 0:
                    count += 1
                    found_start = False
            elif char == '[':
                if brace_depth == 0 and bracket_depth == 0:
                    found_start = True
                bracket_depth += 1
            elif char == ']':
                bracket_depth = max(0, bracket_depth - 1)
                if found_start and brace_depth == 0 and bracket_depth == 0:
                    count += 1
                    found_start = False
        
        return count


class MaxTimeCriteria(StoppingCriteria):
    """
    Stop generation after a maximum time limit.
    Safety net to prevent indefinitely long generations.
    """
    
    def __init__(self, max_seconds: float = 30.0):
        self.max_seconds = max_seconds
        self.start_time = None
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_time is None:
            self.start_time = time.time()
        
        elapsed = time.time() - self.start_time
        if elapsed > self.max_seconds:
            print(f"[STOP] Max time ({self.max_seconds}s) exceeded")
            return True
        
        return False
    
    def reset(self):
        """Reset timer for next generation."""
        self.start_time = None


class RepetitionDetectionCriteria(StoppingCriteria):
    """
    Stop generation when repetitive patterns are detected.
    Catches cases like "Other Other Other..." or repeated class names.

    IMPORTANT: Only checks GENERATED tokens, not the input prompt.
    The input_len parameter tracks where the prompt ends.
    """

    def __init__(self, tokenizer, max_repetitions: int = 3, window_size: int = 50, input_len: int = 0):
        self.tokenizer = tokenizer
        self.max_repetitions = max_repetitions
        self.window_size = window_size
        self.input_len = input_len  # Track where prompt ends
        self.last_check_len = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        current_len = input_ids.shape[1]

        # Calculate how many tokens have been generated (exclude prompt)
        generated_len = current_len - self.input_len

        # Need at least 15 generated tokens before checking for repetition
        if generated_len < 15:
            return False

        # Only check every 10 NEW generated tokens
        if generated_len - self.last_check_len < 10:
            return False
        self.last_check_len = generated_len

        # Only check GENERATED tokens (after prompt)
        # Use window within generated tokens only
        window = min(self.window_size, generated_len)
        generated_tokens = input_ids[0, -window:]
        decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Check for word-level repetitions
        words = decoded.split()
        if len(words) >= self.max_repetitions:
            # Check last N words for all same
            last_words = words[-self.max_repetitions:]
            if len(set(last_words)) == 1:
                print(f"[STOP] Repetition detected: '{last_words[0]}' repeated {self.max_repetitions} times")
                return True

        # Check for pattern repetition (e.g., "class name, class name, class name")
        pattern = r'(\b\w+(?:\s+\w+)?\s*,?\s*)\1{2,}'
        if re.search(pattern, decoded):
            print(f"[STOP] Pattern repetition detected")
            return True

        return False


def create_stopping_criteria(tokenizer, task: str = "detection", input_len: int = 0) -> StoppingCriteriaList:
    """
    Create a list of stopping criteria appropriate for the task.

    Args:
        tokenizer: The tokenizer being used
        task: One of "detection", "vqa", "screening", "modality"
        input_len: Length of input prompt in tokens (to exclude from checking prompt content)

    Returns:
        StoppingCriteriaList with appropriate criteria

    CRITICAL: All text-checking criteria MUST receive input_len to avoid
    triggering on prompt content (e.g., JSON examples in the prompt).
    """
    criteria = []

    # Always add stop string criteria (with input_len to avoid checking prompt)
    criteria.append(StopStringCriteria(tokenizer, input_len=input_len))

    # Always add time limit
    # Increased to allow for slower generation and model warmup during evaluation
    max_time = 180.0 if task == "detection" else 60.0
    criteria.append(MaxTimeCriteria(max_seconds=max_time))

    # Add repetition detection  but NOT for detection task
    # Detection JSON has legitimate repeated patterns like {"class":...,"box":...}
    # that falsely trigger the regex-based repetition detector
    if task != "detection":
        criteria.append(RepetitionDetectionCriteria(tokenizer, input_len=input_len))

    # Add JSON completion for detection task (with input_len - CRITICAL!)
    # The prompt contains JSON examples that would trigger false positives
    if task == "detection":
        criteria.append(JSONCompleteStoppingCriteria(tokenizer, input_len=input_len))

    return StoppingCriteriaList(criteria)


class VocabularyConstraint:
    """
    Constrains model output to valid medical vocabulary.
    Used for detection to ensure only valid class names are generated.
    """
    
    # VinDr-CXR valid classes
    VINDR_CLASSES = [
        "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
        "Clavicle fracture", "Consolidation", "Edema", "Emphysema",
        "Enlarged PA", "ILD", "Infiltration", "Lung Opacity", "Lung cavity",
        "Lung cyst", "Mediastinal shift", "Nodule/Mass", "Other lesion",
        "Pleural effusion", "Pleural thickening", "Pneumothorax",
        "Pulmonary fibrosis", "Rib fracture", "No finding"
    ]
    
    # NIH ChestX-ray14 valid classes  
    NIH_CLASSES = [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
        "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
        "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia", "No Finding"
    ]
    
    # Combined vocabulary (normalized to lowercase)
    ALL_CLASSES = set([c.lower() for c in VINDR_CLASSES + NIH_CLASSES])
    
    @classmethod
    def normalize_class(cls, class_name: str) -> str:
        """Normalize a class name to standard format."""
        if not class_name:
            return ""
        
        name_lower = class_name.lower().strip()
        
        # Direct match
        if name_lower in cls.ALL_CLASSES:
            # Find original casing
            for c in cls.VINDR_CLASSES + cls.NIH_CLASSES:
                if c.lower() == name_lower:
                    return c
        
        # Fuzzy matching for common variations
        mappings = {
            "pleural effusion": "Pleural effusion",
            "effusion": "Pleural effusion",
            "nodule": "Nodule/Mass",
            "mass": "Nodule/Mass",
            "nodule/mass": "Nodule/Mass",
            "opacity": "Lung Opacity",
            "lung opacity": "Lung Opacity",
            "lung opacities": "Lung Opacity",
            "opacities": "Lung Opacity",
            "parenchymal opacity": "Lung Opacity",
            "no finding": "No finding",
            "no significant abnormality": "No finding",
            "normal": "No finding",
            "healthy": "No finding",
            "fibrosis": "Pulmonary fibrosis",
            "pulmonary fibrosis": "Pulmonary fibrosis",
            "scarring": "Pulmonary fibrosis",
            "ild": "ILD",
            "interstitial lung disease": "ILD",
            "reticular markings": "ILD",
            "pneumothorax": "Pneumothorax",
            "cardiomegaly": "Cardiomegaly",
            "enlarged heart": "Cardiomegaly",
            "enlarged cardiomediastinum": "Cardiomegaly",
            "atelectasis": "Atelectasis",
            "consolidation": "Consolidation",
            "infiltration": "Infiltration",
            "infiltrate": "Infiltration",
            "infiltrates": "Infiltration",
            "emphysema": "Emphysema",
            "edema": "Edema",
            "pulmonary edema": "Edema",
            "pleural thickening": "Pleural thickening",
            "pleural extension": "Pleural thickening",
            "pleural scarring": "Pleural thickening",
            "calcification": "Calcification",
            "calcified": "Calcification",
            "calcified nodule": "Calcification",
            "aortic enlargement": "Aortic enlargement",
            "aortic knob": "Aortic enlargement",
            "dilated aorta": "Aortic enlargement",
            "prominent aortic knob": "Aortic enlargement",
            "cavity": "Lung cavity",
            "cyst": "Lung cyst",
            "other lesion": "Other lesion",
            "bone lesion": "Other lesion",
            "soft tissue lesion": "Other lesion",
            "pleural plaque": "Pleural thickening",
            "lung opacities": "Lung Opacity",
            "foreign object": "Other lesion",
            "consolidation": "Consolidation",
            "pneumonia": "Consolidation",
        }
        
        if name_lower in mappings:
            return mappings[name_lower]
        
        # Partial match
        for standard in cls.ALL_CLASSES:
            if name_lower in standard or standard in name_lower:
                for c in cls.VINDR_CLASSES + cls.NIH_CLASSES:
                    if c.lower() == standard:
                        return c
        
        # Unknown class - return as "Other lesion"
        print(f"Warning: Unknown class '{class_name}' mapped to 'Other lesion'")
        return "Other lesion"
    
    @classmethod
    def validate_findings(cls, findings: list) -> list:
        """Validate, normalize, and deduplicate a list of findings."""
        validated = []
        seen_boxes = []
        
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            
            box = finding.get("box", finding.get("b", []))
            class_name = finding.get("class", finding.get("c", ""))
            
            # Normal finding check
            if class_name.lower() in ["no significant abnormality", "no finding", "healthy", "normal"] or box == [0, 0, 0, 0]:
                if not validated:
                    validated.append({"class": "No significant abnormality", "box": [0, 0, 0, 0]})
                continue

            normalized = cls.normalize_class(class_name)
            
            if normalized and len(box) == 4:
                # Deduplicate identical or near-identical boxes
                # Convert to ints for robust comparison
                box_int = [int(float(b)) for b in box]
                is_duplicate = False
                for seen in seen_boxes:
                    if box_int == seen:
                        is_duplicate = True
                        break
                    # Simple IoU check for near-duplicates
                    try:
                        # Inline simple IoU for speed
                        xA = max(box_int[0], seen[0])
                        yA = max(box_int[1], seen[1])
                        xB = min(box_int[2], seen[2])
                        yB = min(box_int[3], seen[3])
                        inter = max(0, xB - xA) * max(0, yB - yA)
                        areaA = (box_int[2] - box_int[0]) * (box_int[3] - box_int[1])
                        areaB = (seen[2] - seen[0]) * (seen[3] - seen[1])
                        iou = inter / float(areaA + areaB - inter + 1e-6)
                        if iou > 0.95:
                            is_duplicate = True
                            break
                    except:
                        continue
                
                if not is_duplicate:
                    # PRESERVE all information and use standard keys
                    cleaned_finding = finding.copy()
                    cleaned_finding["class"] = normalized
                    cleaned_finding["box"] = box_int
                    # Remove compressed keys if present
                    cleaned_finding.pop("c", None)
                    cleaned_finding.pop("b", None)
                    cleaned_finding.pop("r", None) # Prefer 'reasoning'
                    if "reasoning" not in cleaned_finding and "r" in finding:
                        cleaned_finding["reasoning"] = finding["r"]
                    validated.append(cleaned_finding)
                    seen_boxes.append(box_int)
        
        return validated
