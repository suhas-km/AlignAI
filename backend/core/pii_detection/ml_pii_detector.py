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
            # Try to find model in application path - directly use final_model path
            self.model_dir = Path(__file__).parent.parent.parent / "models" / "pii_detection" / "final_model"
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
            # Make absolutely sure we're using the final_model directory
            base_dir = Path(__file__).parent.parent.parent / "models" / "pii_detection" / "final_model"
            self.model_dir = base_dir
            
            logger.info(f"Loading PII detection model from {self.model_dir}")
            
            # Ensure the model directory exists
            if not self.model_dir.exists():
                logger.error(f"PII detection model directory not found at {self.model_dir}")
                return False
                
            # Check that required files exist
            required_files = ["config.json", "model.safetensors", "tokenizer.json"]
            for file in required_files:
                if not (self.model_dir / file).exists():
                    logger.error(f"Required model file {file} not found in {self.model_dir}")
                    return False
            
            logger.info(f"Required files verified in {self.model_dir}")
            
            # Load tokenizer and model with explicit path to each file
            logger.info(f"Loading tokenizer from {self.model_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True)
            
            logger.info(f"Loading model from {self.model_dir}")
            self.model = AutoModelForTokenClassification.from_pretrained(str(self.model_dir), local_files_only=True)
            
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
            logger.debug(f"Detecting PII in text: {text[:100]}...")
            
            # Tokenize the input
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, return_offsets_mapping=True)
            
            # Get predictions with confidence scores
            with torch.no_grad():
                outputs = self.model(**{k: v for k, v in inputs.items() if k != 'offset_mapping'})
                predictions = torch.argmax(outputs.logits, dim=2)
                # Get confidence scores
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                max_probs = torch.max(probs, dim=-1).values[0].tolist()
            
            # Process the predictions
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            token_predictions = [self.id2tag.get(prediction.item(), "O") for prediction in predictions[0]]
            offset_mapping = inputs["offset_mapping"][0]
            
            # Log detailed token and prediction information
            logger.info(f"Input text: {text}")
            logger.info(f"Tokens and their predictions (confidence > 0.5):")
            any_pii_found = False
            
            for i, (token, pred, prob) in enumerate(zip(tokens, token_predictions, max_probs)):
                if pred != "O" and prob > 0.5:  # Only log predictions with confidence > 50%
                    any_pii_found = True
                    logger.info(f"  Token: {token:<15} | Prediction: {pred:<15} | Confidence: {prob:.2%}")
            
            if not any_pii_found:
                logger.info("  No PII detected with confidence > 50%")
            
            logger.debug(f"All tokens: {tokens}")
            logger.debug(f"All predictions: {token_predictions}")
            
            # Group tokens to get PII entities
            entities = []
            current_entity = None
            
            # First, collect all valid PII tokens with their positions and types
            pii_tokens = []
            for i, (token, pred, prob) in enumerate(zip(tokens, token_predictions, max_probs)):
                # Skip special tokens and padding
                if token in ['[CLS]', '[SEP]', '[PAD]']:
                    continue
                
                token_start, token_end = offset_mapping[i].tolist()
                clean_token = token.replace("##", "").strip()
                
                if pred != "O" and prob > 0.5:  # Only consider predictions with confidence > 50%
                    entity_type = pred[2:] if "-" in pred else pred
                    is_begin = pred.startswith("B-") or pred == "O"
                    
                    pii_tokens.append({
                        'token': clean_token,
                        'start': token_start,
                        'end': token_end,
                        'type': entity_type,
                        'is_begin': is_begin,
                        'score': prob
                    })
            
            # Now group consecutive tokens into entities
            current_entity = None
            for i, token_info in enumerate(pii_tokens):
                if current_entity is None or token_info['is_begin'] or token_info['type'] != current_entity['pii_type']:
                    # Save current entity if exists
                    if current_entity is not None:
                        entities.append(current_entity)
                    
                    # Start new entity
                    current_entity = {
                        'start': token_info['start'],
                        'end': token_info['end'],
                        'text': token_info['token'],
                        'pii_type': token_info['type'],
                        'score': token_info['score'],
                        'tokens': [token_info]
                    }
                else:
                    # Continue current entity
                    current_entity['end'] = token_info['end']
                    current_entity['text'] += ' ' + token_info['token'] if not token_info['token'].startswith('##') else token_info['token']
                    current_entity['tokens'].append(token_info)
                    # Update average score
                    current_entity['score'] = sum(t['score'] for t in current_entity['tokens']) / len(current_entity['tokens'])
            
            # Add the last entity if it exists
            if current_entity is not None:
                entities.append(current_entity)
                
            # Log the final entities
            logger.info(f"Grouped {len(entities)} PII entities:")
            for i, entity in enumerate(entities, 1):
                logger.info(f"  Entity {i}: '{entity['text']}' | Type: {entity['pii_type']} | "
                           f"Position: {entity['start']}-{entity['end']} | Score: {entity['score']:.2%}")
            
            # Add the last entity if it exists
            if current_entity:
                entities.append(current_entity)
            
            # Format results for API response
            results = []
            for entity in entities:
                result = {
                    "start": entity.get("start", 0),
                    "end": entity.get("end", 0),
                    "risk_score": entity.get("score", 0.9),  # Default to 0.9 if not provided
                    "risk_type": "pii",
                    "pii_type": entity.get("pii_type", "UNKNOWN"),
                    "matched_text": entity.get("text", ""),
                    "explanation": f"{entity.get('pii_type', 'PII')} detected"
                }
                results.append(result)
                
                # Log the formatted result for debugging
                logger.debug(f"Formatted PII result: {result}")
            
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
        
        logger.info(f"Detected {len(entities)} PII entities")
        for i, entity in enumerate(entities, 1):
            logger.info(f"  Entity {i}: {entity['text']} | Type: {entity['pii_type']} | Position: {entity['start']}-{entity['end']}")
        logger.debug(f"All entities: {entities}")
        
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
