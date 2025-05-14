#!/usr/bin/env python3
"""
EU AI Act Policy Detection Model Trainer

This script trains a model to detect EU AI Act policy violations, bias, and provide
risk assessments and recommendations. The model is designed to integrate with the AlignAI
UI for policy compliance monitoring.

Usage:
  python policy_trainer.py [--data_dir DATA_DIR] [--output_dir OUTPUT_DIR] [--model MODEL_NAME]

Example:
  python policy_trainer.py --model distilbert-base-uncased --data_dir ./data --output_dir ./models
"""

import os
import json
import logging
import argparse
import random
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback,
    set_seed
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('policy_detection.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('PolicyModelTrainer')

# Constants
DEFAULT_MODEL = "distilbert-base-uncased"  # Lightweight but effective transformer
DEFAULT_DATA_DIR = "./data"
DEFAULT_OUTPUT_DIR = "./models"
RANDOM_SEED = 42

# EU AI Act Articles and Categories mapping
EU_AI_ACT_CATEGORIES = {
    "bias": "Bias and Discrimination",
    "transparency": "Transparency and Information",
    "risk": "Risk Management",
    "human_oversight": "Human Oversight",
    "data_governance": "Data Governance",
    "technical_robustness": "Technical Robustness and Safety",
    "prohibited_practices": "Prohibited AI Practices",
    "other": "Other Requirements"
}

@dataclass
class PolicyExample:
    """Data class to hold a single policy example with metadata."""
    text: str
    labels: Dict[str, float] = field(default_factory=dict)
    article_ids: List[str] = field(default_factory=list)
    policy_matches: Dict[str, float] = field(default_factory=dict)
    source: str = "EU AI Act"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "text": self.text,
            "labels": self.labels,
            "article_ids": self.article_ids,
            "policy_matches": self.policy_matches,
            "source": self.source
        }

class PolicyDataset(Dataset):
    """Dataset for training the policy detection model"""
    
    def __init__(self, examples: List[PolicyExample], tokenizer, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Determine which labels we're training for
        self.label_keys = list(EU_AI_ACT_CATEGORIES.keys())
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            example.text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Extract values from the encoding
        item = {
            key: val.squeeze(0) for key, val in encoding.items()
        }
        
        # Add labels
        labels = [example.labels.get(key, 0.0) for key in self.label_keys]
        item["labels"] = torch.tensor(labels, dtype=torch.float)
        
        return item

class PolicyModelTrainer:
    """Trainer for the EU AI Act policy detection model"""
    
    def __init__(
        self, 
        model_name: str = DEFAULT_MODEL,
        data_dir: str = DEFAULT_DATA_DIR,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        batch_size: int = 16,
        learning_rate: float = 5e-5,
        epochs: int = 5,
        max_length: int = 512,
        seed: int = RANDOM_SEED
    ):
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.max_length = max_length
        self.seed = seed
        
        # Set seed for reproducibility
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing PolicyModelTrainer with model: {model_name}")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        
    def _load_data(self) -> List[PolicyExample]:
        """Load and process training data from JSONL files"""
        examples = []
        
        # Load training data
        train_path = self.data_dir / "train.jsonl"
        if train_path.exists():
            logger.info(f"Loading training data from {train_path}")
            with open(train_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line.strip())
                    # Convert qa.jsonl data to policy examples
                    text = item.get("question", "")
                    
                    # Map categories to labels
                    category = item.get("category", "other")
                    labels = {category.lower(): 1.0}
                    
                    # Extract article IDs (format might be "Article XX: ...")
                    context = item.get("context", "")
                    article_ids = []
                    if "Article" in context:
                        article_match = re.search(r"Article\s+(\d+)", context)
                        if article_match:
                            article_ids.append(f"Article {article_match.group(1)}")
                    
                    # Create policy example
                    example = PolicyExample(
                        text=text,
                        labels=labels,
                        article_ids=article_ids,
                        policy_matches={}
                    )
                    examples.append(example)
        
        # Also try to load examples from dev.jsonl for evaluation
        dev_path = self.data_dir / "dev.jsonl"
        if dev_path.exists():
            logger.info(f"Adding evaluation data from {dev_path}")
            with open(dev_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line.strip())
                    text = item.get("question", "")
                    category = item.get("category", "other")
                    labels = {category.lower(): 1.0}
                    
                    example = PolicyExample(
                        text=text,
                        labels=labels,
                        article_ids=[],
                        policy_matches={}
                    )
                    examples.append(example)
        
        logger.info(f"Loaded {len(examples)} examples in total")
        return examples
    
    def _compute_metrics(self, pred_and_labels):
        """Compute evaluation metrics for the model - compatibility version"""
        # Handle different input formats based on transformers version
        if hasattr(pred_and_labels, 'predictions'):
            # Newer transformers version
            predictions = (pred_and_labels.predictions > 0.5).astype(np.int32)
            labels = pred_and_labels.label_ids.astype(np.int32)
        else:
            # Older transformers version passes predictions, labels as tuple
            predictions, labels = pred_and_labels
            predictions = (predictions > 0.5).astype(np.int32)
            labels = labels.astype(np.int32)
        
        # Calculate metrics for each label
        results = {}
        for i, category in enumerate(EU_AI_ACT_CATEGORIES.keys()):
            try:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    labels[:, i], predictions[:, i], average='binary'
                )
                acc = accuracy_score(labels[:, i], predictions[:, i])
                results[f"{category}_precision"] = precision
                results[f"{category}_recall"] = recall
                results[f"{category}_f1"] = f1
                results[f"{category}_accuracy"] = acc
            except Exception as e:
                logger.warning(f"Error computing metrics for {category}: {str(e)}")
                results[f"{category}_precision"] = 0.0
                results[f"{category}_recall"] = 0.0
                results[f"{category}_f1"] = 0.0
                results[f"{category}_accuracy"] = 0.0
        
        # Calculate overall metrics
        try:
            overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
                labels.flatten(), predictions.flatten(), average='macro'
            )
            results["precision"] = overall_precision
            results["recall"] = overall_recall
            results["f1"] = overall_f1
        except Exception as e:
            logger.warning(f"Error computing overall metrics: {str(e)}")
            results["precision"] = 0.0
            results["recall"] = 0.0
            results["f1"] = 0.0
        
        return results
    
    def train(self):
        """Train the policy detection model"""
        # Load data
        examples = self._load_data()
        if not examples:
            logger.error("No training data found. Exiting.")
            return
        
        # Split data into train and validation sets
        train_examples, val_examples = train_test_split(
            examples, test_size=0.1, random_state=self.seed
        )
        
        logger.info(f"Training on {len(train_examples)} examples")
        logger.info(f"Validating on {len(val_examples)} examples")
        
        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Get the number of labels
        num_labels = len(EU_AI_ACT_CATEGORIES)
        
        # Load the model with a multi-label classification head
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification"
        )
        
        # Create datasets
        train_dataset = PolicyDataset(train_examples, tokenizer, self.max_length)
        val_dataset = PolicyDataset(val_examples, tokenizer, self.max_length)
        
        # Set up training arguments with backward compatibility
        # Make everything simpler to avoid version compatibility issues
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "checkpoints"),
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            weight_decay=0.01,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=10,
            # Disable load_best_model_at_end to avoid compatibility issues
            load_best_model_at_end=False,
            save_total_limit=2,  # Only keep the 2 best checkpoints
            # Disable wandb and other integrations
            report_to="none"
        )
        
        # Initialize trainer with backward compatibility - no callbacks for simplicity
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics
        )
        
        # Train the model
        logger.info("Starting training...")
        trainer.train()
        
        # Evaluate the model
        logger.info("Evaluating model...")
        eval_results = trainer.evaluate()
        logger.info(f"Evaluation results: {eval_results}")
        
        # Save the final model
        logger.info(f"Saving model to {self.output_dir}")
        trainer.save_model(str(self.output_dir / "final_model"))
        tokenizer.save_pretrained(str(self.output_dir / "final_model"))
        
        # Save label mapping
        with open(self.output_dir / "final_model" / "label_mapping.json", "w") as f:
            json.dump(EU_AI_ACT_CATEGORIES, f)
        
        logger.info("Training complete!")

class PolicyPredictor:
    """Class for making predictions with the trained policy model"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer and model
        logger.info(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load label mapping
        with open(os.path.join(model_path, "label_mapping.json"), "r") as f:
            self.label_mapping = json.load(f)
        
        # Label keys in order
        self.label_keys = list(self.label_mapping.keys())
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Make a prediction for the given text"""
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            scores = torch.sigmoid(logits).squeeze().cpu().numpy()
        
        # Process results
        results = self._format_results(text, scores)
        return results
    
    def _format_results(self, text: str, scores: np.ndarray) -> Dict[str, Any]:
        """Format the prediction results in the format expected by the UI"""
        # Map scores to categories
        category_scores = {}
        for i, key in enumerate(self.label_keys):
            category_scores[key] = float(scores[i])
        
        # Calculate overall risk score (weighted average of all category scores)
        risk_weights = {
            "bias": 1.0,
            "transparency": 0.8,
            "risk": 1.0,
            "human_oversight": 0.7,
            "data_governance": 0.6,
            "technical_robustness": 0.9,
            "prohibited_practices": 1.2,
            "other": 0.5
        }
        
        weighted_scores = []
        weights = []
        for category, score in category_scores.items():
            weight = risk_weights.get(category, 0.5)
            weighted_scores.append(score * weight)
            weights.append(weight)
        
        overall_risk = sum(weighted_scores) / sum(weights) if weights else 0
        overall_risk = min(max(overall_risk, 0.0), 1.0)  # Clip to [0, 1]
        
        # Format policy categories for UI display
        risk_categories = []
        for category, score in category_scores.items():
            if score > 0.3:  # Only include categories with significant scores
                risk_categories.append({
                    "name": self.label_mapping[category],
                    "score": int(score * 100)  # Convert to percentage
                })
        
        # Sort by score in descending order
        risk_categories.sort(key=lambda x: x["score"], reverse=True)
        
        # Generate findings based on detected issues
        findings = []
        for category, score in category_scores.items():
            if score > 0.5:  # Only include significant findings
                if category == "bias":
                    findings.append({
                        "category": "Bias",
                        "description": "Potential bias detected in language or approach",
                        "score": int(score * 100)
                    })
                elif category == "prohibited_practices":
                    findings.append({
                        "category": "Policy Violation",
                        "description": "Potential violation of prohibited AI practices",
                        "score": int(score * 100)
                    })
        
        # Format relevant policies (articles) based on prediction
        relevant_policies = []
        if category_scores.get("bias", 0) > 0.5:
            relevant_policies.append({
                "id": "Article 10",
                "description": "Data quality and transparency requirements...",
                "match": int(category_scores["bias"] * 100)
            })
        
        if category_scores.get("prohibited_practices", 0) > 0.5:
            relevant_policies.append({
                "id": "Article 5",
                "description": "Prohibited artificial intelligence practices...",
                "match": int(category_scores["prohibited_practices"] * 100)
            })
        
        # Generate recommendations based on findings
        recommendations = []
        if category_scores.get("bias", 0) > 0.5:
            recommendations.append("Review text for potential bias and consider using more neutral language.")
        
        if any(category_scores.get(cat, 0) > 0.5 for cat in ["transparency", "risk", "data_governance"]):
            recommendations.append("Review relevant EU AI Act articles to ensure regulatory compliance.")
        
        # Construct the final result structure for the UI
        result = {
            "text": text,
            "overall_risk": {
                "score": int(overall_risk * 100)  # Convert to percentage
            },
            "risk_categories": risk_categories,
            "detailed_findings": findings,
            "relevant_policies": relevant_policies,
            "recommendations": recommendations,
            "raw_scores": category_scores
        }
        
        return result

# Command-line interface
def main():
    parser = argparse.ArgumentParser(description="EU AI Act Policy Detection Model Trainer")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new policy detection model")
    train_parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Base model to use for training")
    train_parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Directory containing training data")
    train_parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save trained model")
    train_parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    train_parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    train_parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make predictions with a trained model")
    predict_parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    predict_parser.add_argument("--text", type=str, help="Text to analyze")
    predict_parser.add_argument("--input_file", type=str, help="File containing text to analyze (one text per line)")
    predict_parser.add_argument("--output_file", type=str, help="File to save prediction results")
    
    args = parser.parse_args()
    
    if args.command == "train":
        # Train a new model
        trainer = PolicyModelTrainer(
            model_name=args.model,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            epochs=args.epochs
        )
        trainer.train()
    
    elif args.command == "predict":
        # Make predictions
        predictor = PolicyPredictor(args.model_path)
        
        if args.text:
            # Predict for a single text
            result = predictor.predict(args.text)
            print(json.dumps(result, indent=2))
        
        elif args.input_file:
            # Predict for texts in a file
            results = []
            with open(args.input_file, "r", encoding="utf-8") as f:
                for line in f:
                    text = line.strip()
                    if text:  # Skip empty lines
                        result = predictor.predict(text)
                        results.append(result)
            
            # Save results
            if args.output_file:
                with open(args.output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Saved {len(results)} predictions to {args.output_file}")
            else:
                print(json.dumps(results, indent=2))
        
        else:
            logger.error("Either --text or --input_file must be provided")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()