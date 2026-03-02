"""
Unit tests for stopping_criteria.py JSON counting logic.

Tests the _count_complete_json static method directly to avoid
heavy dependency mocking issues with torch/transformers.

Run: python tests/test_stopping_criteria.py
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Mock heavy dependencies before importing
mock_torch = MagicMock()
mock_transformers = MagicMock()
# Make StoppingCriteria a real base class so __init__ works normally
mock_transformers.StoppingCriteria = type('StoppingCriteria', (), {'__init__': lambda self: None})
sys.modules['torch'] = mock_torch
sys.modules['transformers'] = mock_transformers

# Now safe to import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.stopping_criteria import JSONCompleteStoppingCriteria


class TestCountCompleteJSON(unittest.TestCase):
    """Tests for JSONCompleteStoppingCriteria._count_complete_json"""
    
    def setUp(self):
        """Create a criteria instance with mocked tokenizer."""
        mock_tokenizer = MagicMock()
        self.criteria = JSONCompleteStoppingCriteria(mock_tokenizer)
    
    def _count(self, text: str) -> int:
        return self.criteria._count_complete_json(text)
    
    # --- Basic Cases ---
    
    def test_empty_string(self):
        self.assertEqual(self._count(""), 0)
    
    def test_simple_object(self):
        self.assertEqual(self._count('{"key": "value"}'), 1)
    
    def test_simple_array(self):
        self.assertEqual(self._count('[1, 2, 3]'), 1)
        
    def test_empty_object(self):
        self.assertEqual(self._count('{}'), 1)
        
    def test_empty_array(self):
        self.assertEqual(self._count('[]'), 1)
    
    # --- Multiple Top-Level Structures ---
    
    def test_two_objects(self):
        self.assertEqual(self._count('{"a":1}{"b":2}'), 2)
        
    def test_object_then_array(self):
        self.assertEqual(self._count('{"a":1}[1,2]'), 2)
    
    # --- Nested Structures ---
    
    def test_nested_object(self):
        self.assertEqual(self._count('{"a": {"b": {"c": 1}}}'), 1)
        
    def test_nested_array(self):
        self.assertEqual(self._count('[[1, [2]], [3]]'), 1)
        
    def test_mixed_nesting(self):
        self.assertEqual(self._count('{"findings": [{"class": "X", "box": [1,2,3,4]}]}'), 1)
    
    # --- String Content (should not affect counting) ---
    
    def test_braces_in_string(self):
        self.assertEqual(self._count('{"key": "value with { and }"}'), 1)
    
    def test_brackets_in_string(self):
        self.assertEqual(self._count('{"key": "array [1,2] here"}'), 1)
    
    def test_escaped_quotes(self):
        self.assertEqual(self._count('{"key": "value \\\\"with\\\\" quotes"}'), 1)
        
    def test_escaped_backslash(self):
        self.assertEqual(self._count('{"path": "C:\\\\\\\\Users\\\\\\\\test"}'), 1)
    
    # --- Incomplete / Malformed JSON ---
    
    def test_incomplete_object(self):
        self.assertEqual(self._count('{"key": "value"'), 0)
    
    def test_incomplete_array(self):
        self.assertEqual(self._count('[1, 2, 3'), 0)
        
    def test_no_json(self):
        self.assertEqual(self._count('hello world'), 0)
    
    # --- Edge Cases (robustness — PR #1 fix) ---
    
    def test_leading_closing_brace(self):
        """Stray '}' before real JSON — must not crash or go negative."""
        self.assertEqual(self._count('}{"a": 1}'), 1)
    
    def test_leading_closing_bracket(self):
        """Stray ']' before real JSON — must not crash or go negative."""
        self.assertEqual(self._count('][1, 2]'), 1)
    
    def test_multiple_stray_closers(self):
        """Multiple stray '}]' before real JSON."""
        self.assertEqual(self._count('}]}{"valid": true}'), 1)
    
    def test_text_then_json(self):
        """Common LLM output: text then JSON."""
        self.assertEqual(self._count('Here are the findings: {"findings": []}'), 1)
    
    # --- Medical Pipeline Specific ---
    
    def test_detection_output(self):
        """Full detection output with findings array."""
        text = '{"findings": [{"class": "Cardiomegaly", "box": [120, 340, 580, 720], "reasoning": "Enlarged cardiac silhouette"}]}'
        self.assertEqual(self._count(text), 1)
    
    def test_normal_findings(self):
        """Normal case detection output."""
        text = '{"findings": [{"class": "No significant abnormality", "box": [0,0,0,0], "reasoning": "Normal"}]}'
        self.assertEqual(self._count(text), 1)
    
    def test_multi_finding(self):
        """Multiple findings in one JSON."""
        text = '{"findings": [{"class": "A", "box": [1,2,3,4]}, {"class": "B", "box": [5,6,7,8]}]}'
        self.assertEqual(self._count(text), 1)


if __name__ == "__main__":
    unittest.main()
