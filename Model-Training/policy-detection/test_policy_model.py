#!/usr/bin/env python3
"""
EU AI Act Policy Model Evaluation Script

This script evaluates the trained policy detection model with various test cases
to ensure it correctly identifies policy violations according to the EU AI Act.

Usage:
  python test_policy_model.py [--model_path MODEL_PATH]

Example:
  python test_policy_model.py --model_path ./policy-model-weights/final_model
"""

import os
import json
import logging
import argparse
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('PolicyModelTester')

class PolicyModelTester:
    """Class to test the policy detection model with various inputs."""
    
    def __init__(self, model_path: str):
        """Initialize the policy model tester.
        
        Args:
            model_path: Path to the trained model directory
        """
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.label_mapping = {}
        self.label_keys = []
        
        # Load the model and tokenizer
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and tokenizer."""
        logger.info(f"Loading model from {self.model_path}")
        
        # Check if model files exist
        required_files = ["config.json", "model.safetensors", "tokenizer.json", "label_mapping.json"]
        for file in required_files:
            if not (self.model_path / file).exists():
                raise FileNotFoundError(f"Required file {file} not found in {self.model_path}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
        self.model.to(self.device)
        self.model.eval()
        
        # Load label mapping
        with open(self.model_path / "label_mapping.json", 'r') as f:
            self.label_mapping = json.load(f)
        self.label_keys = list(self.label_mapping.keys())
        
        logger.info(f"Model loaded with {len(self.label_keys)} categories")
    
    def predict(self, text: str, threshold: float = 0.5) -> Dict[str, Any]:
        """Make a prediction for the given text.
        
        Args:
            text: Input text to analyze
            threshold: Confidence threshold for considering a category as present
            
        Returns:
            Dictionary with prediction results
        """
        if not text.strip():
            return {"error": "Empty input text"}
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get probabilities using softmax
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            probs = probs.cpu().numpy()[0]
            
            # Get top predictions
            predictions = {}
            for i, prob in enumerate(probs):
                if i < len(self.label_keys):
                    label = self.label_keys[i]
                    predictions[label] = float(prob)
            
            # Get categories above threshold
            violations = {
                self.label_mapping[label]: {
                    "confidence": prob,
                    "explanation": f"Detected {self.label_mapping[label].lower()} with confidence {prob:.2f}"
                }
                for label, prob in predictions.items() 
                if prob >= threshold and label in self.label_mapping
            }
            
            return {
                "text": text,
                "predictions": predictions,
                "has_violation": len(violations) > 0,
                "violations": violations,
                "top_category": max(predictions.items(), key=lambda x: x[1])[0] if predictions else None,
                "top_confidence": max(predictions.values()) if predictions else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return {"error": str(e)}

def get_test_cases() -> List[Dict[str, Any]]:
    """Get test cases for evaluating the policy model."""
    return [
        {
            "text": "This AI system will rank job applicants based on their gender and age to ensure diversity.",
            "expected_categories": ["bias"],
            "description": "Direct discrimination based on protected attributes"
        },
        {
            "text": "The AI system makes decisions without any human oversight or possibility of appeal.",
            "expected_categories": ["human_oversight"],
            "description": "Lack of human oversight"
        },
        {
            "text": "We collect all available user data without consent and use it for training our models.",
            "expected_categories": ["data_governance"],
            "description": "Data collection without consent"
        },
        {
            "text": "The system's decision-making process is completely opaque with no explanation provided to users.",
            "expected_categories": ["transparency"],
            "description": "Lack of transparency"
        },
        {
            "text": "This AI system is designed to manipulate human behavior to increase engagement and time spent on the platform.",
            "expected_categories": ["prohibited_practices"],
            "description": "Manipulative AI practices"
        },
        {
            "text": "The system has not been tested for security vulnerabilities or adversarial attacks.",
            "expected_categories": ["technical_robustness"],
            "description": "Lack of security testing"
        },
        {
            "text": "We deploy this high-risk AI system without any risk assessment or mitigation measures.",
            "expected_categories": ["risk"],
            "description": "No risk assessment for high-risk AI"
        },
        {
            "text": "The system performs automated credit scoring without any human oversight or appeal process.",
            "expected_categories": ["human_oversight", "risk"],
            "description": "High-risk AI system without proper safeguards"
        },
        {
            "text": "This is a completely harmless text that should not trigger any policy violations.",
            "expected_categories": [],
            "description": "Benign text (should not trigger any violations)"
        }
    ]

def run_tests(model_path: str, threshold: float = 0.5):
    """Run tests on the policy model.
    
    Args:
        model_path: Path to the trained model
        threshold: Confidence threshold for predictions
    """
    tester = PolicyModelTester(model_path)
    test_cases = get_test_cases()
    
    print("\n" + "="*80)
    print(f"EU AI Act Policy Model Evaluation - Testing {len(test_cases)} cases")
    print("="*80)
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['description']}")
        print("-" * 60)
        print(f"Input: {test_case['text']}")
        
        # Get prediction
        result = tester.predict(test_case['text'], threshold)
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            results.append(False)
            continue
        
        # Print top prediction
        if result["top_category"]:
            print(f"Top category: {result['top_category']} (confidence: {result['top_confidence']:.2f})")
        
        # Print violations
        if result["violations"]:
            print("\nDetected violations:")
            for category, details in result["violations"].items():
                print(f"  - {category}: {details['confidence']:.2f}")
        else:
            print("✅ No policy violations detected")
        
        # Check if expected categories were detected
        expected_categories = test_case.get("expected_categories", [])
        detected_categories = [cat.lower() for cat in result["violations"].keys()]
        
        # For each expected category, check if it was detected
        correct = True
        for cat in expected_categories:
            cat_name = tester.label_mapping.get(cat, "").lower()
            if not any(cat_name in detected_cat.lower() for detected_cat in detected_categories):
                print(f"❌ Expected category '{cat}' not detected")
                correct = False
        
        # Check for false positives
        if not expected_categories and detected_categories:
            print(f"❌ False positive: Detected {len(detected_categories)} violations when none were expected")
            correct = False
        
        results.append(correct)
        print(f"Result: {'✅ PASS' if correct else '❌ FAIL'}")
    
    # Print summary
    print("\n" + "="*80)
    print(f"Test Results: {sum(results)}/{len(results)} passed ({sum(results)/len(results)*100:.1f}%)")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Test the EU AI Act policy detection model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./policy-model-weights/final_model",
        help="Path to the trained model directory"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for predictions (0.0-1.0)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model path '{args.model_path}' does not exist")
        return
    
    run_tests(args.model_path, args.threshold)

if __name__ == "__main__":
    main()