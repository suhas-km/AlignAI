#!/usr/bin/env python3
"""
Bias Detection Model Predictor
------------------------------
This module provides a simple API for using the trained bias detection models
to analyze text for various forms of bias. It's designed to be integrated
with the AlignAI frontend.
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class BiasPredictor:
    """Predict various forms of bias in text using fine-tuned models."""
    
    def __init__(self, model_dir="models/bias_detection"):
        """Initialize the bias predictor with trained models.
        
        Args:
            model_dir: Directory containing the trained models
        """
        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load binary classifier (biased vs. not biased)
        self.binary_model_path = self.model_dir / "binary_classifier" / "final_model"
        self.binary_tokenizer = None
        self.binary_model = None
        
        # Load multiclass classifier (specific bias types)
        self.multiclass_model_path = self.model_dir / "multiclass_classifier" / "final_model"
        self.multiclass_tokenizer = None
        self.multiclass_model = None
        
        # Load bias categories
        self.bias_categories = {
            0: 'AGE',
            1: 'DISABILITY',
            2: 'ETHNICITY',
            3: 'GENDER',
            4: 'RELIGION',
            5: 'SEXUAL_ORIENTATION',
            6: 'SOCIOECONOMIC',
            7: 'NONE'
        }
        
        # Human-readable category descriptions
        self.category_descriptions = {
            'AGE': "Age-related bias",
            'DISABILITY': "Disability-related bias",
            'ETHNICITY': "Ethnicity-related bias",
            'GENDER': "Gender-related bias",
            'RELIGION': "Religion-related bias",
            'SEXUAL_ORIENTATION': "Sexual orientation bias",
            'SOCIOECONOMIC': "Socioeconomic bias",
            'NONE': "No detected bias"
        }
        
        # Try to load metadata and overwrite defaults if available
        metadata_path = self.multiclass_model_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                if "bias_categories" in metadata:
                    self.bias_categories = {int(k): v for k, v in metadata["bias_categories"].items()}
    
    def load_models(self):
        """Load the models from disk."""
        # Load binary classifier
        if self.binary_model_path.exists():
            try:
                self.binary_tokenizer = AutoTokenizer.from_pretrained(str(self.binary_model_path))
                self.binary_model = AutoModelForSequenceClassification.from_pretrained(
                    str(self.binary_model_path)
                ).to(self.device)
                self.binary_model.eval()
                print(f"Binary classifier loaded from {self.binary_model_path}")
            except Exception as e:
                print(f"Error loading binary model: {e}")
        else:
            print(f"Binary model not found at {self.binary_model_path}")
        
        # Load multiclass classifier
        if self.multiclass_model_path.exists():
            try:
                self.multiclass_tokenizer = AutoTokenizer.from_pretrained(str(self.multiclass_model_path))
                self.multiclass_model = AutoModelForSequenceClassification.from_pretrained(
                    str(self.multiclass_model_path)
                ).to(self.device)
                self.multiclass_model.eval()
                print(f"Multiclass classifier loaded from {self.multiclass_model_path}")
            except Exception as e:
                print(f"Error loading multiclass model: {e}")
        else:
            print(f"Multiclass model not found at {self.multiclass_model_path}")
    
    def predict_binary(self, text):
        """Predict if text contains bias (binary classification).
        
        Args:
            text: The text to analyze
            
        Returns:
            A dictionary with prediction results
        """
        if self.binary_model is None or self.binary_tokenizer is None:
            self.load_models()
            if self.binary_model is None:
                return {"error": "Binary model not loaded"}
        
        # Tokenize input
        inputs = self.binary_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding="max_length"
        ).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.binary_model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        
        # Format results
        prediction = {
            "text": text,
            "contains_bias": bool(np.argmax(probabilities) == 1),
            "bias_probability": float(probabilities[1]),
            "no_bias_probability": float(probabilities[0])
        }
        
        return prediction
    
    def predict_bias_type(self, text):
        """Predict specific type of bias in text (multiclass classification).
        
        Args:
            text: The text to analyze
            
        Returns:
            A dictionary with prediction results
        """
        if self.multiclass_model is None or self.multiclass_tokenizer is None:
            self.load_models()
            if self.multiclass_model is None:
                return {"error": "Multiclass model not loaded"}
        
        # Tokenize input
        inputs = self.multiclass_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding="max_length"
        ).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.multiclass_model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        
        # Get top bias categories
        top_indices = np.argsort(probabilities)[::-1]
        top_categories = []
        
        for idx in top_indices:
            if idx == 7:  # NONE category
                continue
            category_code = self.bias_categories[idx]
            category_desc = self.category_descriptions.get(category_code, category_code)
            top_categories.append({
                "category_id": int(idx),
                "category_code": category_code,
                "category_name": category_desc,
                "probability": float(probabilities[idx]),
                "score": int(100 * probabilities[idx])
            })
        
        # Format results
        prediction = {
            "text": text,
            "primary_bias_type": top_categories[0] if top_categories else None,
            "contains_bias": top_indices[0] != 7,  # Not the NONE category
            "bias_probability": float(1.0 - probabilities[7]),  # Inverse of NONE probability
            "bias_categories": top_categories[:3]  # Return top 3 categories
        }
        
        return prediction
    
    def analyze_text(self, text):
        """Comprehensive bias analysis of text.
        
        Args:
            text: The text to analyze
            
        Returns:
            A dictionary with detailed bias analysis
        """
        # First check if the text contains bias (binary classifier)
        binary_result = self.predict_binary(text)
        
        # Only run multiclass analysis if bias is detected or probability is significant
        if binary_result.get("contains_bias", False) or binary_result.get("bias_probability", 0) > 0.3:
            # Run detailed bias type analysis
            type_result = self.predict_bias_type(text)
            
            # Generate formatted result for the UI
            result = {
                "text": text,
                "contains_bias": binary_result.get("contains_bias", False),
                "bias_probability": binary_result.get("bias_probability", 0),
                "bias_score": int(100 * binary_result.get("bias_probability", 0)),
                "bias_categories": type_result.get("bias_categories", []),
                "primary_bias_type": type_result.get("primary_bias_type"),
                "recommendations": self._generate_recommendations(type_result)
            }
        else:
            # No significant bias detected
            result = {
                "text": text,
                "contains_bias": False,
                "bias_probability": binary_result.get("bias_probability", 0),
                "bias_score": int(100 * binary_result.get("bias_probability", 0)),
                "bias_categories": [],
                "primary_bias_type": None,
                "recommendations": []
            }
        
        return result
    
    def _generate_recommendations(self, prediction_result):
        """Generate recommendations based on the detected bias.
        
        Args:
            prediction_result: Result from bias type prediction
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Get primary bias type if available
        primary_bias = prediction_result.get("primary_bias_type")
        if not primary_bias:
            return []
        
        category = primary_bias.get("category_code", "")
        
        # Add general recommendation
        recommendations.append(
            f"Review text for potential {self.category_descriptions.get(category, category).lower()}"
        )
        
        # Add specific recommendations based on bias type
        if category == "GENDER":
            recommendations.append(
                "Consider using gender-neutral language and avoiding stereotypical gender associations"
            )
        elif category == "ETHNICITY":
            recommendations.append(
                "Review for ethnic stereotypes and consider more inclusive phrasing"
            )
        elif category == "AGE":
            recommendations.append(
                "Avoid age-related stereotypes and consider more neutral language"
            )
        elif category == "RELIGION":
            recommendations.append(
                "Be mindful of religious stereotypes and ensure respect for diverse beliefs"
            )
        elif category == "DISABILITY":
            recommendations.append(
                "Use person-first language and avoid ableist assumptions"
            )
        elif category == "SEXUAL_ORIENTATION":
            recommendations.append(
                "Ensure language is inclusive and respectful of all sexual orientations"
            )
        elif category == "SOCIOECONOMIC":
            recommendations.append(
                "Be mindful of assumptions about people's socioeconomic status"
            )
        
        # Add general EU AI Act recommendation
        recommendations.append(
            "Review EU AI Act Article 10 on data quality and non-discrimination requirements"
        )
        
        return recommendations


def main():
    """Run a simple demo of the bias predictor."""
    # Create predictor
    predictor = BiasPredictor()
    predictor.load_models()
    
    # Example prompts to test
    examples = [
        "Women will be invited only if they are young",
        "We need someone who can work long hours without complaining",
        "All Muslim people believe in the same principles",
        "The programmer needs to be detail-oriented and logical",
        "We're looking for a native English speaker for this position"
    ]
    
    # Analyze each example
    for i, example in enumerate(examples):
        print(f"\nExample {i+1}: '{example}'")
        result = predictor.analyze_text(example)
        
        print(f"Bias detected: {result['contains_bias']}")
        print(f"Bias score: {result['bias_score']}%")
        
        if result["primary_bias_type"]:
            print(f"Primary bias type: {result['primary_bias_type']['category_name']}")
        
        if result["recommendations"]:
            print("Recommendations:")
            for rec in result["recommendations"]:
                print(f"- {rec}")


if __name__ == "__main__":
    main()
