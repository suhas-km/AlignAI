#!/usr/bin/env python3
"""
PII Detection Predictor
--------------------
This script provides an API for detecting Personally Identifiable Information (PII) in text
using the fine-tuned token classification model.
"""

import os
import json
import torch
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PIIPredictor:
    """Class for detecting PII in text using a fine-tuned transformer model"""
    
    def __init__(self, model_dir: Optional[str] = None):
        """Initialize the PII predictor
        
        Args:
            model_dir: Path to the model directory. If None, use the default model path.
        """
        self.model = None
        self.tokenizer = None
        self.id2tag = None
        self.tag2id = None
        
        # Default model path
        if model_dir is None:
            self.model_dir = Path(__file__).parent / "models" / "pii_detection" / "final_model"
        else:
            self.model_dir = Path(model_dir)
    
    def load_model(self):
        """Load the PII detection model"""
        try:
            logger.info(f"Loading PII detection model from {self.model_dir}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
            self.model = AutoModelForTokenClassification.from_pretrained(str(self.model_dir))
            
            # Load metadata
            with open(self.model_dir / "metadata.json", "r") as f:
                metadata = json.load(f)
                self.id2tag = {int(k): v for k, v in metadata["id2tag"].items()}
                self.tag2id = metadata["tag2id"]
            
            logger.info("PII detection model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading PII detection model: {e}")
            return False
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Detect PII in the provided text
        
        Args:
            text: The text to analyze for PII
            
        Returns:
            A dictionary containing detected PII entities and their positions
        """
        if self.model is None or self.tokenizer is None:
            if not self.load_model():
                return {"error": "Model not loaded", "pii_detected": False, "entities": []}
        
        # Tokenize the input
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)
        
        # Process the predictions
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        token_predictions = [self.id2tag[prediction.item()] for prediction in predictions[0]]
        
        # Group tokens to get PII entities
        entities = []
        current_entity = None
        
        for i, (token, prediction) in enumerate(zip(tokens, token_predictions)):
            # Skip special tokens
            if token in [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token]:
                continue
            
            # If prediction is not "O" (not a PII entity), collect the entity
            if prediction != "O":
                # Extract the PII type from the prediction (e.g., "PII-PERSON" → "PERSON")
                pii_type = prediction.split("-")[1] if "-" in prediction else prediction
                
                # Get the word from subword token
                word = token
                if token.startswith("##"):
                    word = token[2:]
                    # Append to previous entity if it exists
                    if current_entity:
                        current_entity["text"] += word
                        continue
                
                # If we have a current entity of the same type, extend it
                if current_entity and current_entity["type"] == pii_type:
                    if not token.startswith("##"):
                        current_entity["text"] += " " + word
                    else:
                        current_entity["text"] += word
                else:
                    # Save previous entity if it exists
                    if current_entity:
                        entities.append(current_entity)
                    
                    # Start a new entity
                    # Get character offsets by finding the word in the original text
                    start_idx = text.lower().find(word.lower())
                    if start_idx != -1:
                        current_entity = {
                            "text": word,
                            "type": pii_type,
                            "start": start_idx,
                            "end": start_idx + len(word),
                            "score": 1.0  # Placeholder for confidence score
                        }
            else:
                # If the current token is not a PII entity, save any current entity
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # Add the last entity if it exists
        if current_entity:
            entities.append(current_entity)
        
        # Group entities by type
        pii_by_type = {}
        for entity in entities:
            pii_type = entity["type"]
            if pii_type not in pii_by_type:
                pii_by_type[pii_type] = []
            pii_by_type[pii_type].append(entity)
        
        # Create the result dictionary
        result = {
            "pii_detected": len(entities) > 0,
            "entities": entities,
            "pii_by_type": pii_by_type,
            "text": text
        }
        
        return result
    
    def mask_pii(self, text: str, mask_char: str = "*") -> Tuple[str, Dict[str, Any]]:
        """Mask PII entities in the text
        
        Args:
            text: The text to analyze and mask
            mask_char: The character to use for masking (default: '*')
            
        Returns:
            A tuple containing the masked text and the detection results
        """
        # Detect PII entities
        results = self.predict(text)
        
        if not results["pii_detected"]:
            return text, results
        
        # Sort entities by start position (reversed to avoid index changes)
        entities = sorted(results["entities"], key=lambda x: x["start"], reverse=True)
        
        # Mask each entity
        masked_text = text
        for entity in entities:
            start = entity["start"]
            end = entity["end"]
            entity_length = end - start
            masked_entity = f"[{entity['type']}: {mask_char * entity_length}]"
            masked_text = masked_text[:start] + masked_entity + masked_text[end:]
        
        # Update the results
        results["masked_text"] = masked_text
        
        return masked_text, results
    
    def get_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on detected PII
        
        Args:
            results: The PII detection results
            
        Returns:
            A list of recommendations
        """
        recommendations = []
        
        if not results["pii_detected"]:
            recommendations.append("No PII detected in the text.")
            return recommendations
        
        # General recommendation
        recommendations.append("Consider whether sharing this PII is necessary and compliant with data protection regulations.")
        
        # Type-specific recommendations
        for pii_type, entities in results["pii_by_type"].items():
            if pii_type == "PERSON":
                recommendations.append("Personal names are considered PII under GDPR and other privacy regulations.")
            elif pii_type == "LOCATION":
                recommendations.append("Location information may be sensitive and could be used to identify individuals.")
            elif pii_type == "ORGANIZATION":
                recommendations.append("Organization names might be commercially sensitive or help identify individuals.")
            elif pii_type in ["EMAIL", "PHONE"]:
                recommendations.append(f"Direct contact information like {pii_type.lower()} should be carefully protected.")
            elif pii_type in ["GOVERNMENT_ID", "FINANCIAL"]:
                recommendations.append(f"{pii_type.replace('_', ' ').title()} information is highly sensitive and should be encrypted or redacted.")
        
        # Reference EU AI Act
        recommendations.append("Review EU AI Act Article 10 on data quality and protection requirements for personal data.")
        
        return recommendations

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Comprehensive analysis of text for PII detection
        
        Args:
            text: The text to analyze
            
        Returns:
            A dictionary with analysis results including detected PII, masked text,
            and recommendations
        """
        # Detect PII
        results = self.predict(text)
        
        # Generate masked text
        masked_text, _ = self.mask_pii(text)
        results["masked_text"] = masked_text
        
        # Add recommendations
        results["recommendations"] = self.get_recommendations(results)
        
        # Calculate summary stats
        results["pii_count"] = len(results["entities"])
        results["pii_types"] = list(results["pii_by_type"].keys())
        
        return results

# Example usage
def main():
    """Simple example of using the PII Predictor"""
    predictor = PIIPredictor()
    predictor.load_model()
    
    test_texts = [
        "My name is John Smith and my email is john.smith@example.com",
        "Please contact our office at +1-555-123-4567 or visit us at 123 Main St, New York, NY",
        "The transaction was processed with card number 4111-1111-1111-1111 by Amazon Inc.",
        "A normal message without any personal information."
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\nExample {i+1}: {text}")
        results = predictor.analyze_text(text)
        
        print(f"PII detected: {results['pii_detected']}")
        if results['pii_detected']:
            print(f"Number of PII entities: {results['pii_count']}")
            print("Detected entities:")
            for entity in results['entities']:
                print(f"  - {entity['text']} ({entity['type']})")
            
            print(f"\nMasked text: {results['masked_text']}")
            
            print("\nRecommendations:")
            for rec in results['recommendations']:
                print(f"  - {rec}")

if __name__ == "__main__":
    main()
