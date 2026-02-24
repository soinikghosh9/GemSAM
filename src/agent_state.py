from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    """
    Represents the state of the clinical agent as it processes a case.
    """
    # Inputs
    image_path: str
    user_query: str
    
    # Context
    modality: str  # e.g., 'CXR', 'CT', 'MRI'
    image_metadata: Dict[str, Any]
    
    # Working Memory
    messages: List[Dict[str, str]] # History of thoughts/actions
    current_thought: str
    
    # Spatial Findings
    # List of detections: [{'label': 'nodule', 'box': [y1, x1, y2, x2], 'confidence': 0.9}]
    detections: List[Dict[str, Any]] 
    
    # Segmentation Results
    # List of masks (paths or RLE) and measurements
    segmentations: List[Dict[str, Any]]
    
    # Final Output
    final_report: str
    error: Optional[str]
