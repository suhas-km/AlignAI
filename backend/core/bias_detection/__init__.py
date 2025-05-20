# Bias detection module initialization
from typing import Dict, Any
from .bias_detector import BiasDetectionEngine

# Create a singleton instance of the bias detection engine
_bias_detector = BiasDetectionEngine(use_ml=True)

def detect_bias(text: str, threshold: float = 0.7) -> Dict[str, Any]:
    """Detect bias in the provided text.
    
    Args:
        text: The text to analyze for bias
        threshold: The confidence threshold for detection (0.0-1.0)
        
    Returns:
        Dictionary containing bias detection results
    """
    # Get raw bias detection results
    bias_instances = _bias_detector.detect_bias(text)
    
    # Process results into standardized format
    has_bias = any(item.get('risk_score', 0) >= threshold for item in bias_instances)
    
    return {
        "has_bias": has_bias,
        "instances": [
            {
                "start": item.get("start"),
                "end": item.get("end"),
                "score": item.get("risk_score"),
                "type": item.get("bias_type"),
                "explanation": item.get("explanation")
            } for item in bias_instances if item.get('risk_score', 0) >= threshold
        ],
        "overall_score": max([item.get('risk_score', 0) for item in bias_instances] or [0])
    }
