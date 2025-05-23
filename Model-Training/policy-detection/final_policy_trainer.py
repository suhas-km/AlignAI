#!/usr/bin/env python3
"""
EU AI Act Policy Detection Model Trainer

This script fine-tunes a DistilBERT model to detect EU AI Act policy violations, categorize
them, determine severity, and identify relevant articles. The model is trained on the EU AI Act
dataset and supports multi-label classification.

Usage:
  python final_policy_trainer.py [--data_dir DATA_DIR] [--output_dir OUTPUT_DIR] [--epochs EPOCHS]

Example:
  python final_policy_trainer.py --data_dir ./data/data2 --output_dir ./models/policy --epochs 5
"""

import os
import json
import logging
import argparse
import random
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
from tqdm import tqdm
from datetime import datetime

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback,
    set_seed
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

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
DEFAULT_DATA_DIR = "./data/data2"  # EU AI Act dataset directory
DEFAULT_OUTPUT_DIR = "./models/policy_model"
RANDOM_SEED = 42

# EU AI Act Categories mapping
EU_AI_ACT_CATEGORIES = {
    "bias": "Bias and Discrimination",
    "transparency": "Transparency and Information",
    "risk_management": "Risk Management",
    "human_oversight": "Human Oversight",
    "data_governance": "Data Governance",
    "technical_robustness": "Technical Robustness and Safety",
    "prohibited_practices": "Prohibited AI Practices",
    "high_risk_systems": "High-Risk AI Systems",
    "record_keeping": "Record-Keeping",
    "accuracy_robustness": "Accuracy, Robustness, and Cybersecurity"
}

# EU AI Act Articles
EU_AI_ACT_ARTICLES = [
    "Article 5", "Article 6", "Article 7", "Article 9", "Article 10", 
    "Article 12", "Article 13", "Article 14", "Article 15", "Article 29", "Article 52"
]

# Severity levels
SEVERITY_LEVELS = ["none", "low", "medium", "high", "critical", "borderline"]

@dataclass
class PolicyExample:
    """Data class to hold a single policy example with metadata."""
    text: str
    violation: Union[bool, str]  # Can be True, False, or "borderline"
    category: str
    severity: str
    articles: List[str] = field(default_factory=list)
    explanation: str = ""
    context: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "text": self.text,
            "violation": self.violation,
            "category": self.category,
            "severity": self.severity,
            "articles": self.articles,
            "explanation": self.explanation,
            "context": self.context
        }

class PolicyDataset(Dataset):
    """Dataset for training the policy detection model"""
    
    def __init__(self, examples: List[PolicyExample], tokenizer, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Initialize label encoders for multi-task learning
        self.categories = list(EU_AI_ACT_CATEGORIES.keys()) + ["none"]
        self.severity_levels = SEVERITY_LEVELS
        self.article_mlb = MultiLabelBinarizer()
        self.article_mlb.fit([EU_AI_ACT_ARTICLES])
        
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
        
        # Convert violation to binary label (0 or 1)
        if example.violation == "borderline":
            # For borderline cases, we use 0.5 to indicate uncertainty
            violation_label = 0.5
        else:
            violation_label = 1.0 if example.violation else 0.0
        
        # Convert category to one-hot encoding
        category_idx = self.categories.index(example.category) if example.category in self.categories else self.categories.index("none")
        category_label = torch.zeros(len(self.categories))
        category_label[category_idx] = 1.0
        
        # Convert severity to one-hot encoding
        severity_idx = self.severity_levels.index(example.severity) if example.severity in self.severity_levels else 0
        severity_label = torch.zeros(len(self.severity_levels))
        severity_label[severity_idx] = 1.0
        
        # Convert articles to multi-hot encoding
        article_label = torch.tensor(self.article_mlb.transform([example.articles])[0], dtype=torch.float)
        
        # Combine all labels into a single tensor for easier handling
        # Format: [violation, categories, severity, articles]
        # Concatenate all label tensors into a single tensor as required by Transformers
        all_labels = torch.cat([
            torch.tensor([violation_label], dtype=torch.float),
            category_label,
            severity_label,
            article_label
        ])
        
        item["labels"] = all_labels
        
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
        seed: int = RANDOM_SEED,
        save_steps: int = 500,
        eval_steps: int = 500,
        warmup_steps: int = 500,
        weight_decay: float = 0.01
    ):
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.max_length = max_length
        self.seed = seed
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        
        # Set random seeds for reproducibility
        set_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Setup datasets
        self.train_dataset = None
        self.eval_dataset = None
        self.test_dataset = None
        
        # Make sure data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Load data
        self._load_data()
        
        # Configure model
        num_labels = 1 + len(EU_AI_ACT_CATEGORIES) + 1 + len(SEVERITY_LEVELS) + len(EU_AI_ACT_ARTICLES)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification"
        )
        
        logger.info(f"Initialized trainer with model: {model_name}")
        logger.info(f"Training data: {len(self.train_dataset) if self.train_dataset else 0} examples")
        logger.info(f"Validation data: {len(self.eval_dataset) if self.eval_dataset else 0} examples")
        logger.info(f"Test data: {len(self.test_dataset) if self.test_dataset else 0} examples")
    
    def _load_data(self):
        """Load and process training data from JSONL files"""
        logger.info(f"Loading data from {self.data_dir}")
        
        train_file = self.data_dir / "train.jsonl"
        dev_file = self.data_dir / "dev.jsonl"
        test_file = self.data_dir / "test.jsonl"
        
        if not train_file.exists():
            raise FileNotFoundError(f"Training file not found: {train_file}")
        
        # Load training examples
        train_examples = []
        with open(train_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                train_examples.append(PolicyExample(
                    text=data["text"],
                    violation=data["violation"],
                    category=data["category"],
                    severity=data["severity"],
                    articles=data.get("articles", []),
                    explanation=data.get("explanation", ""),
                    context=data.get("context", "")
                ))
        
        # Load validation examples if available
        eval_examples = []
        if dev_file.exists():
            with open(dev_file, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    eval_examples.append(PolicyExample(
                        text=data["text"],
                        violation=data["violation"],
                        category=data["category"],
                        severity=data["severity"],
                        articles=data.get("articles", []),
                        explanation=data.get("explanation", ""),
                        context=data.get("context", "")
                    ))
        
        # Load test examples if available
        test_examples = []
        if test_file.exists():
            with open(test_file, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    test_examples.append(PolicyExample(
                        text=data["text"],
                        violation=data["violation"],
                        category=data["category"],
                        severity=data["severity"],
                        articles=data.get("articles", []),
                        explanation=data.get("explanation", ""),
                        context=data.get("context", "")
                    ))
        
        # Create datasets
        if train_examples:
            self.train_dataset = PolicyDataset(train_examples, self.tokenizer, self.max_length)
            logger.info(f"Loaded {len(train_examples)} training examples")
        
        if eval_examples:
            self.eval_dataset = PolicyDataset(eval_examples, self.tokenizer, self.max_length)
            logger.info(f"Loaded {len(eval_examples)} validation examples")
        elif train_examples:
            # If no validation set, use 10% of training data
            train_size = int(0.9 * len(train_examples))
            eval_size = len(train_examples) - train_size
            train_examples, eval_examples = train_examples[:train_size], train_examples[train_size:]
            self.train_dataset = PolicyDataset(train_examples, self.tokenizer, self.max_length)
            self.eval_dataset = PolicyDataset(eval_examples, self.tokenizer, self.max_length)
            logger.info(f"Split training data: {len(train_examples)} training, {len(eval_examples)} validation")
        
        if test_examples:
            self.test_dataset = PolicyDataset(test_examples, self.tokenizer, self.max_length)
            logger.info(f"Loaded {len(test_examples)} test examples")
    
    def _compute_metrics(self, pred_and_labels):
        """Compute evaluation metrics for the model"""
        predictions, labels = pred_and_labels
        
        # Convert predictions to probabilities using sigmoid for multi-label classification
        predictions = 1 / (1 + np.exp(-predictions))
        
        # Split predictions and labels into separate tasks
        # The output dimension is the sum of all label dimensions
        # Format: [violation, categories, severity, articles]
        
        # For binary violation detection (first element)
        violation_pred = predictions[:, 0] > 0.5
        violation_true = labels[:, 0] > 0.5
        
        # For category classification (multi-class)
        category_start = 1
        category_end = category_start + len(self.train_dataset.categories)
        category_pred = np.argmax(predictions[:, category_start:category_end], axis=1)
        category_true = np.argmax(labels[:, category_start:category_end], axis=1)
        
        # For severity classification (multi-class)
        severity_start = category_end
        severity_end = severity_start + len(self.train_dataset.severity_levels)
        severity_pred = np.argmax(predictions[:, severity_start:severity_end], axis=1)
        severity_true = np.argmax(labels[:, severity_start:severity_end], axis=1)
        
        # For article prediction (multi-label)
        article_start = severity_end
        article_pred = predictions[:, article_start:] > 0.5
        article_true = labels[:, article_start:] > 0.5
        
        # Calculate metrics
        violation_acc = accuracy_score(violation_true, violation_pred)
        violation_prec, violation_rec, violation_f1, _ = precision_recall_fscore_support(
            violation_true, violation_pred, average='binary', zero_division=0
        )
        
        category_acc = accuracy_score(category_true, category_pred)
        category_f1 = f1_score(category_true, category_pred, average='weighted', zero_division=0)
        
        severity_acc = accuracy_score(severity_true, severity_pred)
        severity_f1 = f1_score(severity_true, severity_pred, average='weighted', zero_division=0)
        
        # For articles (multi-label)
        # Handle case where all predictions or all labels are 0
        if np.sum(article_true) == 0 or np.sum(article_pred) == 0:
            article_f1 = 0.0
        else:
            article_f1 = f1_score(article_true, article_pred, average='weighted', zero_division=0)
        
        # Calculate overall F1 as weighted average of all tasks
        overall_f1 = (violation_f1 + category_f1 + severity_f1 + article_f1) / 4
        
        return {
            "accuracy_violation": float(violation_acc),
            "precision_violation": float(violation_prec),
            "recall_violation": float(violation_rec),
            "f1_violation": float(violation_f1),
            "accuracy_category": float(category_acc),
            "f1_category": float(category_f1),
            "accuracy_severity": float(severity_acc),
            "f1_severity": float(severity_f1),
            "f1_articles": float(article_f1),
            "overall_f1": float(overall_f1)
        }
    
    def train(self):
        """Train the policy detection model"""
        logger.info("Starting model training...")
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size * 2,
            warmup_steps=self.warmup_steps,
            weight_decay=self.weight_decay,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=100,
            # Modified parameters for newer transformers version
            eval_strategy="steps",  # Changed from evaluation_strategy
            eval_steps=self.eval_steps,
            save_strategy="steps",
            save_steps=self.save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="overall_f1",
            greater_is_better=True,
            fp16=torch.cuda.is_available(),  # Use mixed precision if available
            report_to="none",  # Disable wandb, tensorboard, etc.
            save_total_limit=2  # Only keep the 2 best checkpoints
        )
        
        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Start training
        logger.info(f"Training model with {len(self.train_dataset)} examples for {self.epochs} epochs")
        train_result = trainer.train()
        
        # Log and save training metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        # Evaluate the model
        if self.eval_dataset:
            logger.info("Evaluating model on validation set...")
            eval_metrics = trainer.evaluate()
            trainer.log_metrics("eval", eval_metrics)
            trainer.save_metrics("eval", eval_metrics)
            
            logger.info("Validation results:")
            for key, value in eval_metrics.items():
                logger.info(f"  {key}: {value:.4f}")
        
        # Test the model if test data is available
        if self.test_dataset:
            logger.info("Evaluating model on test set...")
            test_results = trainer.predict(self.test_dataset)
            test_metrics = test_results.metrics
            
            logger.info("Test results:")
            for key, value in test_metrics.items():
                logger.info(f"  {key}: {value:.4f}")
        
        # Save the final model
        logger.info(f"Saving final model to {self.output_dir}")
        trainer.save_model(str(self.output_dir))
        self.tokenizer.save_pretrained(str(self.output_dir))
        
        # Save the model weights separately for easier loading
        os.makedirs(self.output_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, "model_weights.bin"))
        
        # Save model configuration
        with open(self.output_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump({
                "model_name": self.model_name,
                "max_length": self.max_length,
                "categories": list(EU_AI_ACT_CATEGORIES.keys()) + ["none"],
                "severity_levels": SEVERITY_LEVELS,
                "articles": EU_AI_ACT_ARTICLES,
                "training_params": {
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                    "epochs": self.epochs,
                    "seed": self.seed
                },
                "trained_on": datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info("Training complete!")
        return self.output_dir

class PolicyPredictor:
    """Class for making predictions with the trained policy model"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        
        # Load configuration from our custom config file
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            # Default config if file doesn't exist
            self.config = {
                "categories": list(EU_AI_ACT_CATEGORIES.keys()) + ["none"],
                "severity_levels": SEVERITY_LEVELS,
                "articles": EU_AI_ACT_ARTICLES,
                "max_length": 512
            }
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Try to load model from our custom weights file or pytorch_model.bin
        # This avoids the model_type recognition issue
        model_weights_file = os.path.join(model_path, "model_weights.bin")
        pytorch_model_file = os.path.join(model_path, "pytorch_model.bin")
        
        # Create a new DistilBERT model with the correct number of labels
        from transformers import DistilBertConfig, DistilBertForSequenceClassification
        
        num_labels = 1 + len(self.config["categories"]) + len(self.config["severity_levels"]) + len(self.config["articles"])
        model_config = DistilBertConfig.from_pretrained(
            "distilbert-base-uncased",
            num_labels=num_labels,
            problem_type="multi_label_classification"
        )
        
        self.model = DistilBertForSequenceClassification(model_config)
        
        # Try loading from our custom weights file first, then fallback to pytorch_model.bin
        if os.path.exists(model_weights_file):
            logger.info(f"Loading model weights from {model_weights_file}")
            self.model.load_state_dict(torch.load(model_weights_file, map_location="cpu"))
        elif os.path.exists(pytorch_model_file):
            logger.info(f"Loading model weights from {pytorch_model_file}")
            self.model.load_state_dict(torch.load(pytorch_model_file, map_location="cpu"))
        else:
            raise FileNotFoundError(f"Model weights file not found at {model_weights_file} or {pytorch_model_file}")
            
        self.model.eval()  # Set to evaluation mode
        
        # Initialize device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        # Get categories and other config elements
        self.categories = self.config.get("categories", list(EU_AI_ACT_CATEGORIES.keys()) + ["none"])
        self.severity_levels = self.config.get("severity_levels", SEVERITY_LEVELS)
        self.articles = self.config.get("articles", EU_AI_ACT_ARTICLES)
        self.max_length = self.config.get("max_length", 512)
        
        logger.info(f"Loaded policy model from {model_path}")
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Make a prediction for the given text"""
        # Tokenize input
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.squeeze().cpu().numpy()
        
        # Split logits into separate predictions
        # Format: [violation, categories, severity, articles]
        
        # For binary violation detection
        violation_score = logits[0]
        violation = violation_score > 0.5
        
        # For category classification (multi-class)
        category_start = 1
        category_end = category_start + len(self.categories)
        category_scores = logits[category_start:category_end]
        category_idx = np.argmax(category_scores)
        category = self.categories[category_idx]
        
        # For severity classification (multi-class)
        severity_start = category_end
        severity_end = severity_start + len(self.severity_levels)
        severity_scores = logits[severity_start:severity_end]
        severity_idx = np.argmax(severity_scores)
        severity = self.severity_levels[severity_idx]
        
        # For article prediction (multi-label)
        article_start = severity_end
        article_scores = logits[article_start:]
        article_predictions = article_scores > 0.5
        predicted_articles = [self.articles[i] for i, pred in enumerate(article_predictions) if pred]
        
        # Format results
        result = {
            "text": text,
            "violation": bool(violation),
            "violation_score": float(violation_score),
            "category": category,
            "category_score": float(category_scores[category_idx]),
            "severity": severity,
            "severity_score": float(severity_scores[severity_idx]),
            "articles": predicted_articles,
            "article_scores": {
                self.articles[i]: float(score) 
                for i, score in enumerate(article_scores)
            }
        }
        
        # Add detailed analysis
        result["analysis"] = self._generate_analysis(result)
        
        return result
    
    def _generate_analysis(self, prediction: Dict[str, Any]) -> str:
        """Generate a detailed analysis based on the prediction"""
        if not prediction["violation"]:
            return "No EU AI Act policy violations detected."
        
        category_name = EU_AI_ACT_CATEGORIES.get(prediction["category"], prediction["category"])
        articles_str = ", ".join(prediction["articles"]) if prediction["articles"] else "No specific articles"
        
        analysis = f"Potential {prediction['severity']} severity violation detected in the '{category_name}' category. "
        analysis += f"Relevant articles: {articles_str}."
        
        return analysis

def main():
    """Main function to train and evaluate the policy model"""
    parser = argparse.ArgumentParser(description="EU AI Act Policy Detection Model Trainer")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Directory containing training data")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save trained model")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Base model to fine-tune")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save steps")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    
    args = parser.parse_args()
    
    # Train the model
    trainer = PolicyModelTrainer(
        model_name=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        max_length=args.max_length,
        seed=args.seed,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay
    )
    
    model_dir = trainer.train()
    
    logger.info(f"Model training complete. Model saved to {model_dir}")
    
    # Test the trained model on a sample text
    sample_text = "An AI system that uses facial recognition in public spaces without user consent."
    predictor = PolicyPredictor(str(model_dir))
    prediction = predictor.predict(sample_text)
    
    logger.info("Sample prediction:")
    logger.info(f"Text: {prediction['text']}")
    logger.info(f"Violation: {prediction['violation']}")
    logger.info(f"Category: {prediction['category']}")
    logger.info(f"Severity: {prediction['severity']}")
    logger.info(f"Articles: {prediction['articles']}")
    logger.info(f"Analysis: {prediction['analysis']}")

if __name__ == "__main__":
    main()