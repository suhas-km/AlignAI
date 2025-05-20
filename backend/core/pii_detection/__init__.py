# PII detection module initialization
from typing import Dict, Any, Optional
from .pii_detector import PIIDetectionEngine

# Create a singleton instance of the PII detection engine
_pii_detector = PIIDetectionEngine(use_ml=True)

def detect_pii(text: str, language: Optional[str] = None) -> Dict[str, Any]:
    """Detect personally identifiable information (PII) in the provided text.
    
    Args:
        text: The text to analyze for PII
        language: Optional language code for language-specific PII detection
        
    Returns:
        Dictionary containing PII detection results
    """
    # Get raw PII detection results
    pii_instances = _pii_detector.detect_pii(text)
    
    # Process results into standardized format
    has_pii = len(pii_instances) > 0
    
    return {
        "has_pii": has_pii,
        "instances": [
            {
                "start": item.get("start"),
                "end": item.get("end"),
                "score": item.get("risk_score"),
                "type": item.get("pii_type"),
                "matched_text": item.get("matched_text"),
                "explanation": item.get("explanation")
            } for item in pii_instances
        ],
        "overall_score": max([item.get('risk_score', 0) for item in pii_instances] or [0])
    }
