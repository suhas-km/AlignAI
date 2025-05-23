# Policy detection module initialization
from typing import Dict, Any
from .policy_detector import PolicyDetectionEngine

# Create a singleton instance of the policy detection engine
_policy_detector = PolicyDetectionEngine(use_ml=True)

def check_policy_violations(text: str) -> Dict[str, Any]:
    """Check for EU AI Act policy violations in the provided text.
    
    Args:
        text: The text to analyze for policy violations
        
    Returns:
        Dictionary containing policy violation detection results
    """
    # Get raw policy violation detection results
    policy_matches = _policy_detector.detect_policy_violations(text)
    
    # Process results into standardized format
    has_violation = len(policy_matches) > 0
    
    return {
        "has_violation": has_violation,
        "violations": [
            {
                "article": item.get("article", "").replace("Article Article", "Article"),
                "score": item.get("similarity_score", 0),
                "text": item.get("text_snippet", "")
            } for item in policy_matches
        ],
        "overall_score": max([item.get('similarity_score', 0) for item in policy_matches] or [0]),
        "relevant_policies": [
            {
                "article": item.get("article", "").replace("Article Article", "Article"),
                "text": item.get("text_snippet", "")
            } for item in policy_matches
        ]
    }
