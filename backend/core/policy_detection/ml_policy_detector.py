from typing import List, Dict, Any, Optional, Union
import os
import json
import logging
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DistilBertConfig, DistilBertForSequenceClassification

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
        self.model_config = {}
        self.categories = []
        self.severity_levels = []
        self.articles = []
        self.max_length = 512
        self.policy_info = {}
        
        # Set model directory
        if model_dir is None:
            # Use the new policy_detection directory directly
            self.model_dir = Path(__file__).parent.parent.parent / "models" / "policy_detection"
            
            # If not found, use the original training path as fallback
            if not self.model_dir.exists():
                self.model_dir = Path("/AlignAI/Model-Training/policy-detection/models/policy_model")
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
            logger.info(f"Loading policy detector model from {self.model_dir}")
            
            # Ensure the model directory exists
            if not self.model_dir.exists():
                logger.error(f"Policy detection model directory not found at {self.model_dir}")
                return False
                
            # Check that required files exist
            required_files = ["config.json", "tokenizer.json"]
            model_files = ["model.safetensors", "model_weights.bin", "pytorch_model.bin"]
            
            # Check for required configuration files
            for file in required_files:
                if not (self.model_dir / file).exists():
                    logger.error(f"Required model file {file} not found in {self.model_dir}")
                    return False
            
            # Check for at least one model weights file
            model_file_exists = False
            for file in model_files:
                if (self.model_dir / file).exists():
                    model_file_exists = True
                    self.weights_file = file
                    break
            
            if not model_file_exists:
                logger.error(f"No model weights file found in {self.model_dir}")
                return False
            
            logger.info(f"Required files verified in {self.model_dir}")
            
            # Load the model configuration
            try:
                with open(self.model_dir / "config.json", "r") as f:
                    self.model_config = json.load(f)
                logger.info(f"Loaded custom model configuration")
                
                # Define labels from config or use defaults
                self.categories = self.model_config.get("categories", [
                    "transparency", 
                    "data_governance", 
                    "technical_robustness", 
                    "risk_management", 
                    "human_oversight", 
                    "high_risk_systems",
                    "record_keeping",
                    "accuracy_robustness",
                    "prohibited_practices"
                ])
                
                self.severity_levels = self.model_config.get("severity_levels", [
                    "low", 
                    "medium", 
                    "high", 
                    "critical"
                ])
                
                self.articles = self.model_config.get("articles", [
                    "article_5", 
                    "article_9", 
                    "article_10", 
                    "article_11", 
                    "article_13", 
                    "article_14", 
                    "article_15", 
                    "article_16", 
                    "article_17", 
                    "article_18", 
                    "article_19", 
                    "article_29"
                ])
                
                self.max_length = self.model_config.get("max_length", 512)
                
            except Exception as e:
                logger.error(f"Error loading custom model configuration: {str(e)}")
                # Use default values if not found in config
                self.categories = [
                    "transparency", 
                    "data_governance", 
                    "technical_robustness", 
                    "risk_management", 
                    "human_oversight", 
                    "high_risk_systems",
                    "record_keeping",
                    "accuracy_robustness",
                    "prohibited_practices"
                ]
                self.severity_levels = ["low", "medium", "high", "critical"]
                self.articles = [
                    "article_5", 
                    "article_9", 
                    "article_10", 
                    "article_11", 
                    "article_13", 
                    "article_14", 
                    "article_15", 
                    "article_16", 
                    "article_17", 
                    "article_18", 
                    "article_19", 
                    "article_29"
                ]
                self.max_length = 512
            
            # Load tokenizer
            logger.info(f"Loading tokenizer from {self.model_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True)
            
            # Create appropriate model configuration
            logger.info(f"Configuring model architecture")
            num_labels = 1 + len(self.categories) + len(self.severity_levels) + len(self.articles)
            model_config = DistilBertConfig.from_pretrained(
                "distilbert-base-uncased",
                num_labels=num_labels,
                problem_type="multi_label_classification"
            )
            
            # Initialize model with configuration
            self.model = DistilBertForSequenceClassification(model_config)
            
            # Load model weights
            if (self.model_dir / "model_weights.bin").exists():
                logger.info(f"Loading weights from model_weights.bin")
                self.model.load_state_dict(torch.load(str(self.model_dir / "model_weights.bin"), map_location="cpu"))
            elif (self.model_dir / "pytorch_model.bin").exists():
                logger.info(f"Loading weights from pytorch_model.bin")
                self.model.load_state_dict(torch.load(str(self.model_dir / "pytorch_model.bin"), map_location="cpu"))
            elif (self.model_dir / "model.safetensors").exists():
                logger.info(f"Loading weights from model.safetensors")
                self.model = DistilBertForSequenceClassification.from_pretrained(
                    str(self.model_dir),
                    config=model_config,
                    local_files_only=True
                )
            else:
                logger.error("No model weights file found")
                return False
            
            # Set model to evaluation mode
            self.model.eval()
            
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading policy detection model: {str(e)}")
            return False

    def _load_policy_info(self) -> None:
        """Load EU AI Act policy information."""
        try:
            # Look for policy info file
            policy_info_path = Path(__file__).parent / "policy_info.json"
            
            if policy_info_path.exists():
                with open(policy_info_path, 'r') as f:
                    self.policy_info = json.load(f)
                logger.info(f"Loaded policy information from {policy_info_path}")
            else:
                # Default policy information if file not found
                self.policy_info = {
                    "article_5": {
                        "title": "Prohibited AI Practices",
                        "summary": "AI systems that deploy subliminal techniques, exploit vulnerabilities, or engage in social scoring are prohibited."
                    },
                    "article_10": {
                        "title": "Data and Data Governance",
                        "summary": "Training, validation and testing data shall be subject to appropriate data governance and management practices."
                    },
                    "article_13": {
                        "title": "Transparency and Information Provision",
                        "summary": "High-risk AI systems shall be designed and developed to enable transparency and clear communication to users."
                    },
                    "article_15": {
                        "title": "Accuracy, Robustness and Cybersecurity",
                        "summary": "High-risk AI systems should be accurate, robust and secure throughout their lifecycle."
                    },
                    "article_17": {
                        "title": "Risk Management System",
                        "summary": "A risk management system shall be established for high-risk AI systems."
                    }
                }
                logger.info("Using default policy information")
        except Exception as e:
            logger.error(f"Error loading policy information: {str(e)}")
            # Set minimal default policy info if loading fails
            self.policy_info = {
                "article_10": {
                    "title": "Data and Data Governance",
                    "summary": "Data quality requirements"
                }
            }
    
    def detect_policy_violations(self, text: str) -> Dict[str, Any]:
        """Detect potential policy violations in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dict with prediction results
        """
        try:
            # Check if model is loaded
            if self.model is None or self.tokenizer is None:
                logger.error("Model or tokenizer not loaded. Cannot detect policy violations.")
                return {"is_compliant": True, "violations": [], "compliance_score": 0.0}
            
            # Tokenize input text
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=self.max_length)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0].numpy()
                
                # Convert logits to probabilities using sigmoid for multi-label classification
                probs = 1 / (1 + np.exp(-logits))
            
            # Split predictions into separate tasks
            # Format: [violation, categories, severity, articles]
            
            # For binary violation detection
            violation_score = probs[0]
            is_violation = violation_score > 0.5
            
            # For category classification (multi-class)
            category_start = 1
            category_end = category_start + len(self.categories)
            category_scores = probs[category_start:category_end]
            category_idx = np.argmax(category_scores)
            category = self.categories[category_idx] if category_idx < len(self.categories) else "unknown"
            category_confidence = float(category_scores[category_idx]) if category_idx < len(category_scores) else 0.0
            
            # For severity classification (multi-class)
            severity_start = category_end
            severity_end = severity_start + len(self.severity_levels)
            severity_scores = probs[severity_start:severity_end]
            severity_idx = np.argmax(severity_scores)
            severity = self.severity_levels[severity_idx] if severity_idx < len(self.severity_levels) else "medium"
            severity_confidence = float(severity_scores[severity_idx]) if severity_idx < len(severity_scores) else 0.0
            
            # For article prediction (multi-label)
            article_start = severity_end
            article_end = article_start + len(self.articles)
            article_scores = probs[article_start:article_end]
            article_predictions = article_scores > 0.5
            
            # Get predicted articles with scores
            predicted_articles = []
            for i, is_predicted in enumerate(article_predictions):
                if is_predicted and i < len(self.articles):
                    predicted_articles.append({
                        "article_id": self.articles[i],
                        "score": float(article_scores[i])
                    })
            
            # Sort articles by score
            predicted_articles.sort(key=lambda x: x["score"], reverse=True)
            
            # Format the compliance score (inverse of violation score)
            compliance_score = (1.0 - violation_score) * 100
            
            # Format violations if detected
            violations = []
            if is_violation:
                # Calculate risk level based on severity
                risk_level = "high" if severity in ["critical", "high"] else "medium" if severity == "medium" else "low"
                
                # Format article references
                article_refs = [article["article_id"] for article in predicted_articles]
                article_ref_str = ", ".join(article_refs) if article_refs else "Unknown articles"
                
                # Get a more detailed category description
                category_desc = {
                    "transparency": "Transparency Requirements",
                    "data_governance": "Data and Data Governance",
                    "technical_robustness": "Technical Robustness and Security",
                    "risk_management": "Risk Management System",
                    "human_oversight": "Human Oversight",
                    "high_risk_systems": "High-Risk AI Systems",
                    "record_keeping": "Record-Keeping",
                    "accuracy_robustness": "Accuracy and Robustness",
                    "prohibited_practices": "Prohibited Practices"
                }.get(category, category)
                
                # Create violation entry
                violations.append({
                    "article_id": article_refs[0] if article_refs else "Unknown",  # Primary article
                    "title": category_desc,
                    "severity": severity,
                    "category": category,
                    "summary": f"{severity.capitalize()} severity violation in {category_desc}. Related to {article_ref_str}.",
                    "risk_score": float(violation_score),
                    "risk_level": risk_level,
                    "all_articles": article_refs
                })
            
            return {
                "is_compliant": not is_violation,
                "violations": violations,
                "compliance_score": float(compliance_score),
                "category": category,
                "category_confidence": category_confidence,
                "severity": severity,
                "severity_confidence": severity_confidence,
                "predicted_articles": predicted_articles
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
            elif "transparency" in title.lower():
                recommendations.append("Improve transparency and information provision to users.")
            elif "oversight" in title.lower():
                recommendations.append("Ensure appropriate human oversight for AI systems.")
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
            "text": text,
            "category": result.get("category", ""),
            "severity": result.get("severity", ""),
            "predicted_articles": result.get("predicted_articles", [])
        }
        
        return analysis
    
    def format_for_api(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format policy detection results for API response.
        
        Args:
            analysis_result: Results from policy detection
            
        Returns:
            List of formatted violations for API response
        """
        formatted_results = []
        
        try:
            # Extract violations from analysis result
            violations = analysis_result.get("violations", [])
            
            # Format each violation for API
            for violation in violations:
                # Get article ID
                article_id_str = violation.get("article_id", "unknown")
                
                # Extract article number if possible
                if "article_" in article_id_str.lower():
                    article_num = article_id_str.lower().replace("article_", "")
                else:
                    article_num = article_id_str
                    
                # Try to convert to int
                try:
                    article_num_int = int(article_num)
                except ValueError:
                    article_num_int = 0
                
                # Get all relevant articles
                all_articles = violation.get("all_articles", [])
                related_articles = ", ".join([art.replace("article_", "Article ") for art in all_articles]) if all_articles else f"Article {article_num}"
                
                # Get severity and risk level
                severity = violation.get("severity", "medium")
                risk_level = violation.get("risk_level", "medium")
                
                # Enhanced snippet with more details
                category = violation.get("category", "")
                title = violation.get("title", category)
                
                # Build enhanced text snippet
                text_snippet = f"{severity.capitalize()} severity {risk_level} risk violation: {title}. Related to {related_articles}."
                
                formatted_results.append({
                    "policy_id": article_num_int,
                    "article": f"Article {article_num}".replace("Article article_", "Article "),  # Fix duplication
                    "similarity_score": violation.get("risk_score", 0.6),  # Use risk score as similarity
                    "text_snippet": text_snippet,
                    "severity": severity,
                    "risk_level": risk_level,
                    "category": category,
                    "related_articles": all_articles
                })
                
        except Exception as e:
            logger.error(f"Error formatting policy results for API: {str(e)}")
            
        return formatted_results
