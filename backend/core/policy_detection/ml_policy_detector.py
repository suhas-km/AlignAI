from typing import List, Dict, Any, Optional
import os
import json
import logging
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

class MLPolicyDetector:
    """Machine learning-based policy violation detector using the fine-tuned model."""
    
    def __init__(self, model_dir: Optional[str] = None):
        """Initialize ML-based policy detector.
        
        Args:
            model_dir: Directory containing the policy detection model.
                       If None, uses the default model directory.
        """
        self.model = None
        self.tokenizer = None
        self.id2label = None
        self.label2id = None
        self.policy_info = {}
        
        # Set model directory
        if model_dir is None:
            # Directly use the final_model path which we found contains the model files
            self.model_dir = Path(__file__).parent.parent.parent / "models" / "policy_detection" / "final_model"
            
            # If not found, use the original training path as fallback
            if not self.model_dir.exists():
                self.model_dir = Path("/Users/suhaskm/Desktop/EU AI Act/AlignAI/Model-Training/policy-detection/policy-model-weights")
        else:
            self.model_dir = Path(model_dir)
        
        # Load model
        self._load_model()
        # Load policy information
        self._load_policy_info()
    
    def _load_model(self) -> bool:
        """Load policy detection model from disk.
        
        Returns:
            True if model loaded successfully, False otherwise.
        """
        try:
            # Make absolutely sure we're using the final_model directory
            base_dir = Path(__file__).parent.parent.parent / "models" / "policy_detection" / "final_model"
            self.model_dir = base_dir
            
            logger.info(f"Loading policy detector model from {self.model_dir}")
            
            # Ensure the model directory exists
            if not self.model_dir.exists():
                logger.error(f"Policy detection model directory not found at {self.model_dir}")
                return False
                
            # Check that required files exist
            required_files = ["config.json", "model.safetensors", "tokenizer.json"]
            for file in required_files:
                if not (self.model_dir / file).exists():
                    logger.error(f"Required model file {file} not found in {self.model_dir}")
                    return False
            
            logger.info(f"Required files verified in {self.model_dir}")
                    
            # Load tokenizer and model with explicit path and local_files_only
            logger.info(f"Loading tokenizer from {self.model_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True)
            
            logger.info(f"Loading model from {self.model_dir}")
            self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir), local_files_only=True)
            
            # Try to load label mapping from label_mapping.json first
            label_mapping_path = self.model_dir / "label_mapping.json"
            if label_mapping_path.exists():
                try:
                    with open(label_mapping_path, "r") as f:
                        label_mapping = json.load(f)
                        # Convert the label mapping to the expected format
                        self.id2label = {str(i): category for i, category in enumerate(label_mapping.keys(), 1)}
                        self.id2label["0"] = "compliant"  # Add compliant class
                        self.label2id = {v: str(k) for k, v in self.id2label.items()}
                        logger.info(f"Loaded label mapping from {label_mapping_path}")
                except Exception as e:
                    logger.error(f"Error loading label mapping: {str(e)}")
                    self._setup_default_labels()
            else:
                logger.warning(f"Label mapping file not found at {label_mapping_path}")
                self._setup_default_labels()
                
            logger.info("Policy detector model loaded successfully")
            return True
                
        except Exception as e:
            logger.error(f"Error loading policy detector model: {str(e)}")
            return False
            
    def _setup_default_labels(self):
        """Set up default labels when no mapping file is found."""
        logger.warning("Using default label mapping")
        self.id2label = {"0": "compliant", "1": "non_compliant"}
        self.label2id = {"compliant": "0", "non_compliant": "1"}
    
    def _load_policy_info(self) -> None:
        """Load EU AI Act policy information."""
        try:
            policy_info_path = Path(__file__).parent / "policy_info.json"
            if policy_info_path.exists():
                with open(policy_info_path, "r") as f:
                    self.policy_info = json.load(f)
            else:
                # Create default policy info
                self.policy_info = {
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
                # Save default policy info for future use
                os.makedirs(os.path.dirname(policy_info_path), exist_ok=True)
                with open(policy_info_path, "w") as f:
                    json.dump(self.policy_info, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error loading policy information: {str(e)}")
            # Set default empty dict
            self.policy_info = {}
    
    def detect_policy_violations(self, text: str) -> Dict[str, Any]:
        """Detect potential policy violations in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dict with prediction results
        """
        if not self.model or not self.tokenizer:
            logger.error("Policy detector model not loaded")
            return {"is_compliant": True, "violations": [], "score": 0.0}
        
        try:
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1).numpy()[0]
            
            # Determine compliance (class 0 should be "compliant")
            non_compliant_prob = float(probabilities[1])
            is_compliant = non_compliant_prob <= 0.5
            
            # For simplicity, assume violations relate to common articles
            # In a real implementation, this would use a more sophisticated approach
            violations = []
            if not is_compliant:
                # Identify most likely violated articles
                # Here we're simplifying by using the top relevant articles from the policy_info
                for article_id, info in self.policy_info.items():
                    risk_level = info.get("risk_level", "medium")
                    risk_score = non_compliant_prob * {
                        "low": 0.7,
                        "medium": 0.85,
                        "high": 1.0
                    }.get(risk_level, 0.85)
                    
                    # Only include if risk score is high enough
                    if risk_score > 0.5:
                        violations.append({
                            "article_id": article_id,
                            "title": info.get("title", article_id),
                            "summary": info.get("summary", "Policy violation detected"),
                            "risk_score": risk_score
                        })
            
            # Sort violations by risk score (highest first)
            violations.sort(key=lambda x: x["risk_score"], reverse=True)
            
            return {
                "is_compliant": is_compliant,
                "violations": violations,
                "compliance_score": (1.0 - non_compliant_prob) * 100  # Convert to percentage
            }
            
        except Exception as e:
            logger.error(f"Error in policy violation detection: {str(e)}")
            return {"is_compliant": True, "violations": [], "compliance_score": 0.0}
    
    def get_recommendations(self, result: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on detected policy violations.
        
        Args:
            result: Results from detect_policy_violations
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if result["is_compliant"]:
            recommendations.append("The text appears to be compliant with EU AI Act policies.")
            return recommendations
        
        # General recommendation
        recommendations.append("Review the text for potential EU AI Act policy violations.")
        
        # Add specific recommendations based on violations
        for violation in result["violations"]:
            article_id = violation["article_id"]
            title = violation["title"]
            
            # Article-specific recommendations
            if "data" in title.lower():
                recommendations.append("Ensure proper data governance, quality, and privacy protections.")
            elif "accuracy" in title.lower() or "robustness" in title.lower():
                recommendations.append("Verify the accuracy, robustness, and security of AI systems.")
            elif "risk" in title.lower():
                recommendations.append("Implement appropriate risk management procedures for high-risk AI systems.")
            else:
                recommendations.append(f"Review compliance with {article_id.upper()}: {title}.")
        
        return recommendations
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Comprehensive analysis of text for policy compliance.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict with complete policy analysis results
        """
        # Detect policy violations
        result = self.detect_policy_violations(text)
        
        # Generate recommendations
        recommendations = self.get_recommendations(result)
        
        # Combine results
        analysis = {
            "is_compliant": result["is_compliant"],
            "compliance_score": result["compliance_score"],
            "violations": result["violations"],
            "recommendations": recommendations,
            "text": text
        }
        
        return analysis
    
    def format_for_api(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format policy analysis results for the API response format.
        
        Args:
            analysis_result: Results from analyze_text
            
        Returns:
            List of formatted policy match items for API response
        """
        policy_matches = []
        
        if analysis_result["is_compliant"]:
            return policy_matches
        
        # Format each violation as a policy match
        for violation in analysis_result["violations"]:
            policy_matches.append({
                "policy_id": violation["article_id"],
                "title": violation["title"],
                "description": violation["summary"],
                "match_score": violation["risk_score"],
                "risk_level": "high" if violation["risk_score"] > 0.7 else "medium"
            })
        
        return policy_matches
