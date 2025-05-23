from typing import List, Dict, Any, Optional
import os
import json
import logging
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

class MLBiasDetector:
    """Machine learning-based bias detector using the fine-tuned models."""
    
    def __init__(self, model_dir: Optional[str] = None):
        """Initialize ML-based bias detector.
        
        Args:
            model_dir: Directory containing the bias detection models.
                       If None, uses the default model directory.
        """
        self.binary_model = None
        self.binary_tokenizer = None
        self.multiclass_model = None
        self.multiclass_tokenizer = None
        self.id2label = None
        self.label2id = None
        
        # Set model directory
        if model_dir is None:
            # Try to find model in application path
            base_dir = Path(__file__).parent.parent.parent / "models" / "bias_detection"
            # Check if we have the nested structure (models/bias_detection/bias_detection/...)
            if (base_dir / "bias_detection").exists():
                self.model_dir = base_dir / "bias_detection"
            else:
                self.model_dir = base_dir
            # If not found, use the original training path
            if not self.model_dir.exists():
                self.model_dir = Path("/AlignAI/Model-Training/bias-detection/models/bias_detection")
        else:
            self.model_dir = Path(model_dir)
        
        # Load models
        self._load_models()
    
    def _load_models(self) -> bool:
        """Load bias detection models from disk.
        
        Returns:
            True if models loaded successfully, False otherwise.
        """
        try:
            # Load binary model (contains/doesn't contain bias)
            binary_model_path = self.model_dir / "binary_classifier" / "final_model"
            if binary_model_path.exists():
                logger.info(f"Loading binary bias classifier from {binary_model_path}")
                self.binary_tokenizer = AutoTokenizer.from_pretrained(str(binary_model_path))
                self.binary_model = AutoModelForSequenceClassification.from_pretrained(str(binary_model_path))
            else:
                logger.warning(f"Binary bias model not found at {binary_model_path}")
            
            # Load multiclass model (bias type classifier)
            multi_model_path = self.model_dir / "multiclass_classifier" / "final_model"
            if multi_model_path.exists():
                logger.info(f"Loading multiclass bias classifier from {multi_model_path}")
                self.multiclass_tokenizer = AutoTokenizer.from_pretrained(str(multi_model_path))
                self.multiclass_model = AutoModelForSequenceClassification.from_pretrained(str(multi_model_path))
                
                # Load labels from metadata file
                metadata_path = multi_model_path / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                        self.id2label = metadata.get("id2label", {})
                        self.label2id = metadata.get("label2id", {})
            else:
                logger.warning(f"Multiclass bias model not found at {multi_model_path}")
            
            # Check if at least one model loaded successfully
            return self.binary_model is not None or self.multiclass_model is not None
            
        except Exception as e:
            logger.error(f"Error loading bias models: {str(e)}")
            return False
    
    def detect_binary_bias(self, text: str) -> Dict[str, Any]:
        """Detect if text contains bias (binary classification).
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dict with prediction results
        """
        if not self.binary_model or not self.binary_tokenizer:
            logger.error("Binary bias model not loaded")
            return {"contains_bias": False, "score": 0.0}
        
        try:
            # Tokenize input
            inputs = self.binary_tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.binary_model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1).numpy()[0]
            
            # Class 1 corresponds to "contains bias"
            bias_prob = float(probabilities[1])
            contains_bias = bias_prob > 0.5
            
            return {
                "contains_bias": contains_bias,
                "score": bias_prob * 100  # Convert to percentage
            }
            
        except Exception as e:
            logger.error(f"Error in binary bias detection: {str(e)}")
            return {"contains_bias": False, "score": 0.0}
    
    def detect_bias_type(self, text: str) -> Dict[str, Any]:
        """Detect specific type of bias (multiclass classification).
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dict with prediction results for bias types
        """
        if not self.multiclass_model or not self.multiclass_tokenizer:
            logger.error("Multiclass bias model not loaded")
            return {"bias_type": "NONE", "bias_categories": []}
        
        try:
            # Tokenize input
            inputs = self.multiclass_tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.multiclass_model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1).numpy()[0]
            
            # Convert numeric labels to text categories
            categories = []
            for i, prob in enumerate(probabilities):
                category_id = str(i)  # Must convert to string to match metadata format
                if category_id in self.id2label:
                    category_name = self.id2label[category_id]
                    # Skip NONE category in results if probability is low
                    if category_name == "NONE" and prob < 0.5:
                        continue
                    
                    # Make category names more readable
                    readable_name = category_name.lower().replace("_", "-") + "-related bias"
                    
                    categories.append({
                        "category_id": category_id,
                        "category_name": readable_name,
                        "score": float(prob) * 100  # Convert to percentage
                    })
            
            # Sort by probability (highest first)
            categories.sort(key=lambda x: x["score"], reverse=True)
            
            # Primary bias type is the highest probability category
            primary_bias = categories[0] if categories else None
            
            return {
                "primary_bias_type": primary_bias,
                "bias_categories": categories
            }
            
        except Exception as e:
            logger.error(f"Error in multiclass bias detection: {str(e)}")
            return {"bias_type": "NONE", "bias_categories": []}
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Comprehensive bias analysis of text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dict with complete bias analysis results
        """
        # Get binary prediction (contains bias or not)
        binary_result = self.detect_binary_bias(text)
        
        # If binary model detects bias, get specific bias types
        if binary_result["contains_bias"]:
            bias_types = self.detect_bias_type(text)
        else:
            bias_types = {"primary_bias_type": None, "bias_categories": []}
        
        # Generate recommendations based on detected bias
        recommendations = self._generate_recommendations(binary_result, bias_types)
        
        # Combine results
        return {
            "contains_bias": binary_result["contains_bias"],
            "bias_score": binary_result["score"],
            "primary_bias_type": bias_types["primary_bias_type"],
            "bias_categories": bias_types["bias_categories"],
            "recommendations": recommendations,
            "text": text
        }
    
    def _generate_recommendations(self, binary_result: Dict[str, Any], 
                                bias_types: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on detected bias.
        
        Args:
            binary_result: Results from binary bias detection
            bias_types: Results from bias type detection
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if not binary_result["contains_bias"]:
            return recommendations
        
        primary_bias = bias_types.get("primary_bias_type", None)
        if primary_bias:
            bias_name = primary_bias["category_name"]
            recommendations.append(f"Review text for potential {bias_name}")
            
            # Add specific recommendations based on bias type
            if "gender" in bias_name.lower():
                recommendations.append("Consider using gender-neutral language and avoiding stereotypical gender associations")
            elif "age" in bias_name.lower():
                recommendations.append("Avoid age-related stereotypes and consider more neutral language")
            elif "ethnic" in bias_name.lower() or "racial" in bias_name.lower():
                recommendations.append("Review for ethnic stereotypes and consider more inclusive phrasing")
            elif "disab" in bias_name.lower():
                recommendations.append("Use person-first language and avoid ableist assumptions")
            elif "religion" in bias_name.lower():
                recommendations.append("Be mindful of religious stereotypes and ensure respect for diverse beliefs")
        
        # Always add reference to EU AI Act
        recommendations.append("Review EU AI Act Article 10 on data quality and non-discrimination requirements")
        
        return recommendations

    def format_for_api(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format bias analysis results for the API response format.
        
        Args:
            analysis_result: Results from analyze_text
            
        Returns:
            List of formatted risk items for API response
        """
        risks = []
        
        if not analysis_result["contains_bias"]:
            return risks
        
        # If bias detected, add a general bias risk item
        risks.append({
            "start": 0,
            "end": len(analysis_result["text"]),
            "risk_score": analysis_result["bias_score"] / 100,  # Convert percentage to 0-1 range
            "risk_type": "bias",
            "bias_type": analysis_result["primary_bias_type"]["category_name"] if analysis_result["primary_bias_type"] else "general",
            "matched_text": analysis_result["text"],
            "explanation": f"Potential bias detected: {analysis_result['primary_bias_type']['category_name'] if analysis_result['primary_bias_type'] else 'general bias'}"
        })
        
        return risks
