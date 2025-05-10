from typing import List, Dict, Any
import re
import logging
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class BiasDetectionEngine:
    """Engine for detecting various types of bias in text."""
    
    def __init__(self, rules_file: str = None):
        """Initialize bias detection engine.
        
        Args:
            rules_file: Path to JSON file containing bias detection rules.
                       If None, uses default rules.
        """
        self.rules = self._load_rules(rules_file)
    
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
            logger.error(f"Error in bias detection: {str(e)}")
            return []
    
    def get_bias_types(self) -> List[str]:
        """Get list of bias types supported by the detector.
        
        Returns:
            List of bias type strings
        """
        return list(self.rules.keys())
