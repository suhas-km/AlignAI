from typing import List, Dict, Any, Optional
import os
import json
import logging
from pathlib import Path

# Import ML-based implementation
from .ml_policy_detector import MLPolicyDetector

logger = logging.getLogger(__name__)

class PolicyDetectionEngine:
    """Engine for detecting EU AI Act policy violations in text."""
    
    def __init__(self, policy_file: Optional[str] = None, use_ml: bool = True, model_dir: Optional[str] = None):
        """Initialize policy detection engine.
        
        Args:
            policy_file: Path to JSON file containing policy information.
                       If None, uses default policies.
            use_ml: Whether to use ML-based detection when available.
            model_dir: Directory containing ML model files.
        """
        self.policy_info = self._load_policies(policy_file)
        self.ml_detector = None
        
        # Initialize ML-based detector if requested
        if use_ml:
            try:
                self.ml_detector = MLPolicyDetector(model_dir)
                logger.info("ML-based policy detector initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize ML-based policy detector: {str(e)}. Falling back to rule-based detection.")
                self.ml_detector = None
    
    def _load_policies(self, policy_file: Optional[str] = None) -> Dict[str, Any]:
        """Load policy information from file.
        
        Args:
            policy_file: Path to policy file
            
        Returns:
            Dict containing policy information
        """
        try:
            # If a specific file is provided, use it
            if policy_file and os.path.exists(policy_file):
                with open(policy_file, 'r') as f:
                    return json.load(f)
            
            # Otherwise, look for default policy file
            default_path = Path(__file__).parent / "policy_info.json"
            if default_path.exists():
                with open(default_path, 'r') as f:
                    return json.load(f)
            
            # Fallback to basic built-in policies
            return {
                "article_10": {
                    "title": "Data and data governance",
                    "summary": "Ensuring appropriate data governance and management practices.",
                    "risk_level": "high"
                },
                "article_15": {
                    "title": "Accuracy, robustness and cybersecurity",
                    "summary": "AI systems should be accurate, robust and secure.",
                    "risk_level": "high"
                },
                "article_17": {
                    "title": "Risk management system",
                    "summary": "Implementation of risk management systems for high-risk AI.",
                    "risk_level": "high"
                }
            }
        except Exception as e:
            logger.error(f"Error loading policy information: {str(e)}")
            # Return minimal fallback policies
            return {
                "article_10": {
                    "title": "Data and data governance",
                    "summary": "Data quality requirements",
                    "risk_level": "high"
                }
            }
    
    def detect_policy_violations(self, text: str) -> List[Dict[str, Any]]:
        """Detect potential policy violations in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of dictionaries containing policy match results
        """
        # Try ML-based detection first if available
        if self.ml_detector is not None:
            try:
                # Get comprehensive analysis from ML model
                analysis_result = self.ml_detector.analyze_text(text)
                
                # If violations are detected, use ML results
                if not analysis_result["is_compliant"]:
                    # Format results for API response
                    ml_results = self.ml_detector.format_for_api(analysis_result)
                    if ml_results:
                        logger.info(f"ML-based policy detector found {len(ml_results)} policy violations")
                        return ml_results
                
                # If compliant or empty results, return empty list
                logger.info("ML-based policy detector found no violations")
                return []
            except Exception as e:
                logger.error(f"Error in ML-based policy detection: {str(e)}.")
        
        # Basic rule-based policy detection as fallback
        logger.info("Using rule-based policy detection as fallback")
        policy_matches = []

        # Define some basic keywords for each policy
        policy_keywords = {
            "article_10": ["data quality", "training data", "bias", "discrimination", "dataset", "representative", "personal data"],
            "article_15": ["accuracy", "robust", "security", "resilient", "reliable", "cybersecurity", "reproducible"],
            "article_17": ["risk", "management", "mitigation", "assessment", "identify risks", "high-risk"],
        }
        
        # Simple keyword matching
        text_lower = text.lower()
        for policy_id, keywords in policy_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    info = self.get_policy_info(policy_id)
                    policy_matches.append({
                        "policy_id": int(policy_id.split("_")[1]) if "_" in policy_id else 0,
                        "article": f"Article {policy_id.split('_')[1]}" if "_" in policy_id else policy_id,
                        "similarity_score": 0.6,  # Medium confidence for rule-based detection
                        "text_snippet": info.get("summary", "Policy match detected")
                    })
                    # Only match once per policy
                    break
                    
        logger.info(f"Rule-based policy detection found {len(policy_matches)} potential matches")
        return policy_matches
    
    def get_policy_info(self, policy_id: str) -> Dict[str, Any]:
        """Get information about a specific policy.
        
        Args:
            policy_id: ID of the policy
            
        Returns:
            Dict containing policy information
        """
        return self.policy_info.get(policy_id, {})