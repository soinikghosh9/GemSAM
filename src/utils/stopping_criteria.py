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
        self.stop_strings = stop_strings or ["<end_of_turn>", "```\n\n", "</s>"]
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

        # CRITICAL: Don't stop on empty findings like {"findings": []}
        # Only count as complete if JSON contains actual content
        stripped = decoded.strip()
        if stripped in ('{"findings": []}', '{"findings":[]}', '{ "findings": [] }'):
            return False  # Force model to keep generating

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
                brace_depth -= 1
                if found_start and brace_depth == 0 and bracket_depth == 0:
                    count += 1
                    found_start = False
            elif char == '[':
                if brace_depth == 0 and bracket_depth == 0:
                    found_start = True
                bracket_depth += 1
            elif char == ']':
                bracket_depth -= 1
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

    # Always add time limit (30s for short tasks, 90s for complex detection JSON)
    # Increased from 15s/60s to allow for model warmup during evaluation
    max_time = 90.0 if task == "detection" else 30.0
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
            "no finding": "No finding",
            "no significant abnormality": "No finding",  # Training uses this phrase
            "normal": "No finding",
            "healthy": "No finding",
            "fibrosis": "Pulmonary fibrosis",
            "pulmonary fibrosis": "Pulmonary fibrosis",
            "ild": "ILD",
            "interstitial lung disease": "ILD",
            "pneumothorax": "Pneumothorax",
            "cardiomegaly": "Cardiomegaly",
            "enlarged heart": "Cardiomegaly",
            "atelectasis": "Atelectasis",
            "consolidation": "Consolidation",
            "infiltration": "Infiltration",
            "infiltrate": "Infiltration",
            "emphysema": "Emphysema",
            "edema": "Edema",
            "pulmonary edema": "Edema",
            "pleural thickening": "Pleural thickening",
            "calcification": "Calcification",
            "cavity": "Lung cavity",
            "cyst": "Lung cyst",
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
        """Validate and normalize a list of findings."""
        validated = []
        seen_classes = set()
        
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            
            class_name = finding.get("class", "")
            normalized = cls.normalize_class(class_name)
            
            # Allow multiple findings of same class (e.g. multiple nodules)
            # if normalized.lower() in seen_classes:
            #    continue
            seen_classes.add(normalized.lower())
            
            # Keep valid findings
            if normalized and normalized.lower() != "other lesion":
                validated.append({
                    "class": normalized,
                    "box": finding.get("box", [])
                })
        
        return validated
