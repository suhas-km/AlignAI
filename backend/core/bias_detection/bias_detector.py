from typing import List, Dict, Any
import re
import logging
import json
import os
from pathlib import Path

# Import ML-based implementation
from .ml_bias_detector import MLBiasDetector

logger = logging.getLogger(__name__)

class BiasDetectionEngine:
    """Engine for detecting various types of bias in text."""
    
    def __init__(self, rules_file: str = None, use_ml: bool = True, model_dir: str = None):
        """Initialize bias detection engine.
        
        Args:
            rules_file: Path to JSON file containing bias detection rules.
                       If None, uses default rules.
            use_ml: Whether to use ML-based detection when available.
            model_dir: Directory containing ML model files.
        """
        self.rules = self._load_rules(rules_file)
        self.ml_detector = None
        
        # Initialize ML-based detector if requested
        if use_ml:
            try:
                self.ml_detector = MLBiasDetector(model_dir)
                logger.info("ML-based bias detector initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize ML-based bias detector: {str(e)}. Falling back to rule-based detection.")
                self.ml_detector = None
    
    def _load_rules(self, rules_file: str = None) -> Dict[str, Any]:
        """Load bias detection rules from file.
        
        Args:
            rules_file: Path to rules file
            
        Returns:
            Dict containing rules configuration
        """
        try:
            # If a specific file is provided, use it
            if rules_file and os.path.exists(rules_file):
                with open(rules_file, 'r') as f:
                    return json.load(f)
            
            # Otherwise, look for default rules file
            default_path = Path(__file__).parent / "bias_rules.json"
            if default_path.exists():
                with open(default_path, 'r') as f:
                    return json.load(f)
            
            # Fallback to basic built-in rules
            return {
                "gender_bias": {
                    "patterns": [
                        r"\b(?:all|every|each|most)\s+(?:doctor|doctors)\s+(?:are|is|be)\s+(?:he|him|his)\b",
                        r"\b(?:all|every|each|most)\s+(?:nurse|nurses)\s+(?:are|is|be)\s+(?:she|her|hers)\b",
                        r"\b(?:all|every|each|most)\s+(?:engineer|engineers)\s+(?:are|is|be)\s+(?:he|him|his)\b",
                        r"\b(?:all|every|each|most)\s+(?:secretary|secretaries)\s+(?:are|is|be)\s+(?:she|her|hers)\b"
                    ],
                    "severity": "medium",
                    "explanation": "Text contains gender stereotypes related to professions."
                },
                "racial_bias": {
                    "patterns": [
                        r"\b(?:all|every|each|most)\s+(?:people|person|individuals|men|women)\s+from\s+(?:[A-Z][a-z]+)\s+(?:are|is)\s+(?:lazy|criminal|dishonest|violent|stupid)\b"
                    ],
                    "severity": "high",
                    "explanation": "Text contains racial stereotypes or generalizations."
                },
                "age_bias": {
                    "patterns": [
                        r"\b(?:all|every|each|most)\s+(?:old|young|elderly|senior)\s+(?:people|person|individuals|men|women)\s+(?:are|is)\s+(?:incompetent|incapable|unable|slow|confused)\b"
                    ],
                    "severity": "medium",
                    "explanation": "Text contains age-based stereotypes or generalizations."
                }
            }
        except Exception as e:
            logger.error(f"Error loading bias rules: {str(e)}")
            # Return minimal fallback rules
            return {
                "gender_bias": {
                    "patterns": [r"\b(?:all)\s+(?:men|women)\s+(?:are)\b"],
                    "severity": "low",
                    "explanation": "Potential gender bias detected."
                }
            }
    
    def detect_bias(self, text: str) -> List[Dict[str, Any]]:
        """Detect various types of bias in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of dictionaries containing bias detection results
        """
        # Try ML-based detection first if available
        if self.ml_detector is not None:
            try:
                # Get comprehensive analysis from ML model
                analysis_result = self.ml_detector.analyze_text(text)
                
                # If bias is detected, use ML results
                if analysis_result["contains_bias"]:
                    # Format results for API response
                    ml_results = self.ml_detector.format_for_api(analysis_result)
                    if ml_results:
                        logger.info(f"ML-based bias detector found {len(ml_results)} bias issues")
                        return ml_results
                    
                # If no bias detected with ML or empty results, fall back to rule-based
                logger.info("ML-based bias detector found no issues, falling back to rule-based detection")
            except Exception as e:
                logger.error(f"Error in ML-based bias detection: {str(e)}. Falling back to rule-based detection.")
        
        # Rule-based detection (fallback)
        results = []
        
        try:
            # Check each bias type
            for bias_type, config in self.rules.items():
                patterns = config.get("patterns", [])
                severity = config.get("severity", "low")
                explanation = config.get("explanation", f"{bias_type} detected")
                
                # Check each pattern for this bias type
                for pattern in patterns:
                    matches = list(re.finditer(pattern, text, re.IGNORECASE))
                    
                    # Add result for each match found
                    for match in matches:
                        start = match.start()
                        end = match.end()
                        matched_text = text[start:end]
                        
                        # Calculate risk score based on severity
                        risk_score = {
                            "low": 0.3,
                            "medium": 0.6,
                            "high": 0.9
                        }.get(severity, 0.5)
                        
                        results.append({
                            "start": start,
                            "end": end,
                            "risk_score": risk_score,
                            "risk_type": "bias",
                            "bias_type": bias_type,
                            "matched_text": matched_text,
                            "explanation": explanation
                        })
            
            return results
        except Exception as e:
            logger.error(f"Error in rule-based bias detection: {str(e)}")
            return []
    
    def get_bias_types(self) -> List[str]:
        """Get list of bias types supported by the detector.
        
        Returns:
            List of bias type strings
        """
        return list(self.rules.keys())
