from typing import List, Dict, Any
import re
import logging
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class PIIDetectionEngine:
    """Engine for detecting personally identifiable information (PII) in text."""
    
    def __init__(self, rules_file: str = None):
        """Initialize PII detection engine.
        
        Args:
            rules_file: Path to JSON file containing PII detection rules.
                       If None, uses default rules.
        """
        self.rules = self._load_rules(rules_file)
    
    def _load_rules(self, rules_file: str = None) -> Dict[str, Any]:
        """Load PII detection rules from file.
        
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
            default_path = Path(__file__).parent / "pii_rules.json"
            if default_path.exists():
                with open(default_path, 'r') as f:
                    return json.load(f)
            
            # Fallback to basic built-in rules
            return {
                "EMAIL": {
                    "pattern": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
                    "severity": "high",
                    "explanation": "Email address detected"
                },
                "PHONE_NUMBER": {
                    "pattern": r"\b(?:\+\d{1,3}[- ]?)?\(?(?:\d{3})\)?[- ]?\d{3}[- ]?\d{4}\b",
                    "severity": "high",
                    "explanation": "Phone number detected"
                },
                "SSN": {
                    "pattern": r"\b\d{3}[-]?\d{2}[-]?\d{4}\b",
                    "severity": "high",
                    "explanation": "Social security number detected"
                },
                "CREDIT_CARD": {
                    "pattern": r"\b(?:\d{4}[- ]?){3}\d{4}\b",
                    "severity": "high",
                    "explanation": "Credit card number detected"
                },
                "IP_ADDRESS": {
                    "pattern": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
                    "severity": "medium",
                    "explanation": "IP address detected"
                },
                "DATE_OF_BIRTH": {
                    "pattern": r"\b(?:(?:19|20)\d{2}[/\-\.]\d{1,2}[/\-\.]\d{1,2})|(?:\d{1,2}[/\-\.]\d{1,2}[/\-\.](?:19|20)\d{2})\b",
                    "severity": "high",
                    "explanation": "Date of birth detected"
                },
                "NAME": {
                    "pattern": r"\b[A-Z][a-z]+\s[A-Z][a-z]+\b",
                    "severity": "medium",
                    "explanation": "Possible person name detected"
                },
                "ADDRESS": {
                    "pattern": r"\b\d+\s[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Court|Ct|Lane|Ln|Way)\b",
                    "severity": "high",
                    "explanation": "Physical address detected"
                }
            }
        except Exception as e:
            logger.error(f"Error loading PII rules: {str(e)}")
            # Return minimal fallback rules
            return {
                "EMAIL": {
                    "pattern": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
                    "severity": "high",
                    "explanation": "Email address detected"
                }
            }
    
    def detect_pii(self, text: str) -> List[Dict[str, Any]]:
        """Detect PII entities in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of dictionaries containing PII detection results
        """
        results = []
        
        try:
            # Check each PII type
            for pii_type, config in self.rules.items():
                pattern = config.get("pattern", "")
                severity = config.get("severity", "medium")
                explanation = config.get("explanation", f"{pii_type} detected")
                
                # Find matches
                matches = list(re.finditer(pattern, text))
                
                # Add result for each match found
                for match in matches:
                    start = match.start()
                    end = match.end()
                    matched_text = text[start:end]
                    
                    # Calculate risk score based on severity
                    risk_score = {
                        "low": 0.3,
                        "medium": 0.7,
                        "high": 0.9
                    }.get(severity, 0.7)
                    
                    results.append({
                        "start": start,
                        "end": end,
                        "risk_score": risk_score,
                        "risk_type": "pii",
                        "pii_type": pii_type,
                        "matched_text": matched_text,
                        "explanation": explanation
                    })
            
            return results
        except Exception as e:
            logger.error(f"Error in PII detection: {str(e)}")
            return []
    
    def get_pii_types(self) -> List[str]:
        """Get list of PII types supported by the detector.
        
        Returns:
            List of PII type strings
        """
        return list(self.rules.keys())
