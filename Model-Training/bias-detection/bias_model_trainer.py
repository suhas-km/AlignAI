#!/usr/bin/env python3
"""
Bias Detection Model Trainer
----------------------------
This script fine-tunes a pre-trained language model on the CLEAR-Bias dataset
to detect various forms of bias in text, including:
- Age bias
- Disability bias
- Ethnicity bias
- Gender bias
- Religion bias
- Sexual orientation bias
- Socioeconomic bias

The model is designed to integrate with the AlignAI frontend for bias analysis.
"""

import os
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed
)
from datasets import Dataset, load_dataset

# Configuration
MODEL_NAME = "distilbert-base-uncased"  # Lightweight but effective
OUTPUT_DIR = Path("models/bias_detection")
BATCH_SIZE = 16
NUM_EPOCHS = 5  # Increased epochs for better learning
LEARNING_RATE = 2e-5
DATA_PATH = Path("data")
TEST_SIZE = 0.2
RANDOM_SEED = 42
MAX_SEQ_LENGTH = 256

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_and_prepare_data():
    """Load and prepare the CLEAR-Bias dataset for fine-tuning."""
    print("Loading and preparing data...")
    
    # Load all configurations
    configs = ["base_prompts", "jailbreak_prompts", "control_set"]
    dfs = []
    
    for config in configs:
        config_dir = DATA_PATH / config
        if not config_dir.exists():
            print(f"Warning: {config_dir} not found. Skipping...")
            continue
            
        # Load all CSV files in the config directory
        for csv_file in config_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                df['config'] = config
                
                # Ensure 'BIAS CATEGORY' column exists
                if 'BIAS CATEGORY' not in df.columns and 'BIAS_CATEGORY' in df.columns:
                    df['BIAS CATEGORY'] = df['BIAS_CATEGORY']
                
                # Ensure 'PROMPT' column exists
                if 'PROMPT' not in df.columns and 'prompt' in df.columns:
                    df['PROMPT'] = df['prompt']
                
                dfs.append(df)
                print(f"Loaded {len(df)} samples from {csv_file}")
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
    
    if not dfs:
        raise ValueError("No data found. Please run download_clear_bias.py first.")
    
    # Combine all data
    df = pd.concat(dfs, ignore_index=True)
    
    # Create binary labels: 1 for biased, 0 for neutral/control
    # For now, we'll use the presence of a bias category as a label
    df['biased'] = df['BIAS CATEGORY'].notna().astype(int)
    
    # Create multi-class labels for bias type
    bias_categories = {
        'AGE': 0,
        'DISABILITY': 1,
        'ETHNICITY': 2,
        'GENDER': 3,
        'RELIGION': 4,
        'SEXUAL_ORIENTATION': 5,
        'SOCIOECONOMIC': 6,
        'NONE': 7  # For non-biased examples
    }
    
    # Normalize bias categories
    df['bias_type'] = df['BIAS CATEGORY'].str.upper() if 'BIAS CATEGORY' in df.columns else 'NONE'
    df['bias_type'] = df['bias_type'].fillna('NONE')
    df['bias_type_id'] = df['bias_type'].map(lambda x: next((v for k, v in bias_categories.items() if k in x), 7))
    
    # Make sure we have PROMPT column
    if 'PROMPT' not in df.columns:
        raise ValueError("PROMPT column missing from dataset")
    
    # Split into train and validation sets
    train_df, val_df = train_test_split(
        df, 
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df['biased'] if 'biased' in df.columns else None
    )
    
    print(f"Training set: {len(train_df)} samples, Validation set: {len(val_df)} samples")
    
    # Save some example prompts for later reference
    with open(OUTPUT_DIR / "example_prompts.json", "w") as f:
        examples = []
        for bias_type, group in train_df.groupby('bias_type', dropna=True):
            if len(group) > 0:
                examples.append({
                    "bias_type": bias_type,
                    "prompt": group.iloc[0]["PROMPT"]
                })
        json.dump(examples, f, indent=2)
    
    # Convert to Hugging Face datasets
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)
    
    return train_ds, val_ds

def tokenize_function(examples, tokenizer):
    """Tokenize the input examples."""
    return tokenizer(
        examples["PROMPT"],
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH
    )

def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    acc = accuracy_score(labels, predictions)
    
    result = {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
    return result

def train_binary_classifier():
    """Train a binary classifier to detect biased vs. non-biased text."""
    # Load tokenizer and model
    print(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2  # Binary classification: biased or not
    )
    
    # Load and prepare data
    train_ds, val_ds = load_and_prepare_data()
    
    # Tokenize the datasets
    print("Tokenizing datasets...")
    
    def process_examples(examples):
        tokenized = tokenize_function(examples, tokenizer)
        tokenized["labels"] = examples["biased"]
        return tokenized
    
    tokenized_train = train_ds.map(
        process_examples,
        batched=True,
        remove_columns=train_ds.column_names
    )
    
    tokenized_val = val_ds.map(
        process_examples,
        batched=True,
        remove_columns=val_ds.column_names
    )
    
    # Set up training arguments with backward compatibility
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "binary_classifier"),
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        logging_dir=str(OUTPUT_DIR / "logs"),
        logging_steps=10,
        # Simplified args compatible with older versions
        save_total_limit=2,  # Only keep the 2 best checkpoints
        do_eval=True,        # Enable evaluation
        load_best_model_at_end=False,
        # Disable wandb and other tracking
        report_to="none"
    )
    
    # Initialize trainer
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    print("Starting binary classifier training...")
    trainer.train()
    
    # Evaluate the model
    print("Evaluating model...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    
    # Save the model and tokenizer
    model_path = OUTPUT_DIR / "binary_classifier" / "final_model"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    # Save model metadata
    with open(model_path / "metadata.json", "w") as f:
        metadata = {
            "model_name": MODEL_NAME,
            "task": "binary_bias_classification",
            "evaluation_metrics": eval_results,
            "max_sequence_length": MAX_SEQ_LENGTH,
            "training_parameters": {
                "epochs": NUM_EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE
            }
        }
        json.dump(metadata, f, indent=2)
    
    print(f"Binary classifier training complete! Model saved to {model_path}")
    
    return model_path

def train_multiclass_classifier():
    """Train a multiclass classifier to identify specific types of bias."""
    # Load tokenizer and model
    print(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=8  # 7 bias types + 1 for no bias
    )
    
    # Load and prepare data
    train_ds, val_ds = load_and_prepare_data()
    
    # Tokenize the datasets
    print("Tokenizing datasets...")
    
    def process_examples(examples):
        tokenized = tokenize_function(examples, tokenizer)
        tokenized["labels"] = examples["bias_type_id"]
        return tokenized
    
    tokenized_train = train_ds.map(
        process_examples,
        batched=True,
        remove_columns=train_ds.column_names
    )
    
    tokenized_val = val_ds.map(
        process_examples,
        batched=True,
        remove_columns=val_ds.column_names
    )
    
    # Set up training arguments with backward compatibility
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "multiclass_classifier"),
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        logging_dir=str(OUTPUT_DIR / "logs"),
        logging_steps=10,
        # Simplified args compatible with older versions
        save_total_limit=2,  # Only keep the 2 best checkpoints
        do_eval=True,        # Enable evaluation
        load_best_model_at_end=False,
        # Disable wandb and other tracking
        report_to="none"
    )
    
    # Initialize trainer
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    print("Starting multiclass classifier training...")
    trainer.train()
    
    # Evaluate the model
    print("Evaluating model...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    
    # Save the model and tokenizer
    model_path = OUTPUT_DIR / "multiclass_classifier" / "final_model"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    # Save model metadata and bias category mapping
    with open(model_path / "metadata.json", "w") as f:
        bias_categories = {
            0: 'AGE',
            1: 'DISABILITY',
            2: 'ETHNICITY',
            3: 'GENDER',
            4: 'RELIGION',
            5: 'SEXUAL_ORIENTATION',
            6: 'SOCIOECONOMIC',
            7: 'NONE'
        }
        
        metadata = {
            "model_name": MODEL_NAME,
            "task": "multiclass_bias_classification",
            "evaluation_metrics": eval_results,
            "max_sequence_length": MAX_SEQ_LENGTH,
            "bias_categories": bias_categories,
            "training_parameters": {
                "epochs": NUM_EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE
            }
        }
        json.dump(metadata, f, indent=2)
    
    print(f"Multiclass classifier training complete! Model saved to {model_path}")
    
    return model_path

def main():
    """Main function to train the bias detection models."""
    # Set seeds for reproducibility
    set_seed(RANDOM_SEED)
    
    print("Starting bias detection model training...")
    
    # Train a binary classifier (biased vs. not biased)
    binary_model_path = train_binary_classifier()
    
    # Train a multiclass classifier (specific bias types)
    multiclass_model_path = train_multiclass_classifier()
    
    print("\nBias detection model training complete!")
    print(f"Binary classifier saved to: {binary_model_path}")
    print(f"Multiclass classifier saved to: {multiclass_model_path}")
    
    print("\nNext steps:")
    print("1. Create a bias_predictor.py script to use these models")
    print("2. Integrate with the AlignAI frontend")
    print("3. Consider creating a quantized version for web deployment")

if __name__ == "__main__":
    main()
