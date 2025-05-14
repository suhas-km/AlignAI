from typing import List, Dict, Any, Optional, Tuple
import os
import json
import logging
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification

logger = logging.getLogger(__name__)

class MLPIIDetector:
    """Machine learning-based PII detector using the fine-tuned token classification model."""
    
    def __init__(self, model_dir: Optional[str] = None):
        """Initialize ML-based PII detector.
        
        Args:
            model_dir: Directory containing the PII detection model.
                      If None, uses the default model directory.
        """
        self.model = None
        self.tokenizer = None
        self.id2tag = None
        self.tag2id = None
        
        # Set model directory
        if model_dir is None:
            # Try to find model in application path
            self.model_dir = Path(__file__).parent.parent.parent / "models" / "pii_detection"
            # If not found, use the original training path
            if not self.model_dir.exists():
                self.model_dir = Path("/Users/suhaskm/Desktop/EU AI Act/AlignAI/Model-Training/pII-detection/models/pii_detection/final_model")
        else:
            self.model_dir = Path(model_dir)
        
        # Load model
        self._load_model()
    
    def _load_model(self) -> bool:
        """Load PII detection model from disk.
        
        Returns:
            True if model loaded successfully, False otherwise.
        """
        try:
            logger.info(f"Loading PII detection model from {self.model_dir}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
            self.model = AutoModelForTokenClassification.from_pretrained(str(self.model_dir))
            
            # Load metadata
            metadata_path = self.model_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    # Convert string keys to integers for id2tag
                    self.id2tag = {int(k): v for k, v in metadata.get("id2tag", {}).items()}
                    self.tag2id = metadata.get("tag2id", {})
            else:
                logger.warning(f"Metadata file not found at {metadata_path}")
                # Default to rule-based detection if metadata missing
                return False
            
            logger.info("PII detection model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading PII detection model: {str(e)}")
            return False
    
    def detect_pii(self, text: str) -> List[Dict[str, Any]]:
        """Detect PII entities in the provided text.
        
        Args:
            text: The text to analyze for PII
            
        Returns:
            List of dictionaries containing PII entities and their positions
        """
        if self.model is None or self.tokenizer is None:
            logger.error("PII detection model not loaded")
            return []
        
        try:
            # Tokenize the input
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=2)
            
            # Process the predictions
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            token_predictions = [self.id2tag.get(prediction.item(), "O") for prediction in predictions[0]]
            
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
                    if current_entity and current_entity["pii_type"] == pii_type:
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
                                "start": start_idx,
                                "end": start_idx + len(word),
                                "text": word,
                                "pii_type": pii_type,
                                "score": 0.9  # Placeholder for confidence score
                            }
                else:
                    # If the current token is not a PII entity, save any current entity
                    if current_entity:
                        entities.append(current_entity)
                        current_entity = None
            
            # Add the last entity if it exists
            if current_entity:
                entities.append(current_entity)
            
            # Format results for API response
            results = []
            for entity in entities:
                results.append({
                    "start": entity["start"],
                    "end": entity["end"],
                    "risk_score": entity["score"],
                    "risk_type": "pii",
                    "pii_type": entity["pii_type"],
                    "matched_text": entity["text"],
                    "explanation": f"{entity['pii_type']} detected"
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in PII detection: {str(e)}")
            return []
    
    def mask_pii(self, text: str, mask_char: str = "*") -> Tuple[str, List[Dict[str, Any]]]:
        """Mask PII entities in the text.
        
        Args:
            text: The text to analyze and mask
            mask_char: The character to use for masking (default: '*')
            
        Returns:
            A tuple containing the masked text and the detection results
        """
        # Detect PII entities
        entities = self.detect_pii(text)
        
        if not entities:
            return text, entities
        
        # Sort entities by start position (reversed to avoid index changes)
        sorted_entities = sorted(entities, key=lambda x: x["start"], reverse=True)
        
        # Mask each entity
        masked_text = text
        for entity in sorted_entities:
            start = entity["start"]
            end = entity["end"]
            entity_length = end - start
            masked_entity = f"[{entity['pii_type']}: {mask_char * entity_length}]"
            masked_text = masked_text[:start] + masked_entity + masked_text[end:]
        
        return masked_text, entities
    
    def get_recommendations(self, entities: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on detected PII.
        
        Args:
            entities: The PII detection results
            
        Returns:
            A list of recommendations
        """
        recommendations = []
        
        if not entities:
            return ["No PII detected in the text."]
        
        # General recommendation
        recommendations.append("Consider whether sharing this PII is necessary and compliant with data protection regulations.")
        
        # Collect unique PII types
        pii_types = set(entity["pii_type"] for entity in entities)
        
        # Type-specific recommendations
        for pii_type in pii_types:
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
        """Comprehensive analysis of text for PII detection.
        
        Args:
            text: The text to analyze
            
        Returns:
            A dictionary with analysis results including detected PII, masked text,
            and recommendations
        """
        # Generate masked text and get entities
        masked_text, entities = self.mask_pii(text)
        
        # Generate recommendations
        recommendations = self.get_recommendations(entities)
        
        # Group entities by type
        pii_by_type = {}
        for entity in entities:
            pii_type = entity["pii_type"]
            if pii_type not in pii_by_type:
                pii_by_type[pii_type] = []
            pii_by_type[pii_type].append(entity)
        
        # Create the result dictionary
        result = {
            "pii_detected": len(entities) > 0,
            "entities": entities,
            "pii_by_type": pii_by_type,
            "masked_text": masked_text,
            "recommendations": recommendations,
            "text": text
        }
        
        return result
