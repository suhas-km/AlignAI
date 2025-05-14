# using the data from the CLEAR-Bias dataset to fine tune a model for bias detection


#!/usr/bin/env python3
"""
Bias Detection Model Fine-tuning
-------------------------------
This script fine-tunes a pre-trained language model on the CLEAR-Bias dataset
to detect various forms of bias in text.
"""

import os
import json
import torch
import pandas as pd
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset, load_metric
import numpy as np
from sklearn.model_selection import train_test_split

# Configuration
MODEL_NAME = "distilbert-base-uncased"  # Lightweight but effective
OUTPUT_DIR = Path("models/bias_detection")
BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
DATA_PATH = Path("data")
TEST_SIZE = 0.2
RANDOM_SEED = 42

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_and_prepare_data():
    """Load and prepare the CLEAR-Bias dataset for fine-tuning."""
    print("Loading and preparing data...")
    
    # Load all configurations
    configs = ["base_prompts", "jailbreak_prompts"]  # Skip control_set for now
    dfs = []
    
    for config in configs:
        config_dir = DATA_PATH / config
        if not config_dir.exists():
            print(f"Warning: {config_dir} not found. Skipping...")
            continue
            
        # Load all CSV files in the config directory
        for csv_file in config_dir.glob("*.csv"):
            df = pd.read_csv(csv_file)
            df['config'] = config
            dfs.append(df)
    
    if not dfs:
        raise ValueError("No data found. Please run download_clear_bias.py first.")
    
    # Combine all data
    df = pd.concat(dfs, ignore_index=True)
    
    # Create binary labels: 1 for biased, 0 for neutral/control
    # For now, we'll use the presence of a bias category as a label
    df['label'] = df['BIAS CATEGORY'].notna().astype(int)
    
    # Split into train and validation sets
    train_df, val_df = train_test_split(
        df, 
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df['label'] if 'label' in df.columns else None
    )
    
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
        max_length=256
    )

def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metric = load_metric("accuracy")
    return metric.compute(predictions=predictions, references=labels)

def train():
    """Main training function."""
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
    tokenized_train = train_ds.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=train_ds.column_names
    )
    tokenized_val = val_ds.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=val_ds.column_names
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        evaluation_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir=str(OUTPUT_DIR / "logs"),
        logging_steps=10,
        report_to="tensorboard"
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
    print("Starting training...")
    trainer.train()
    
    # Save the model and tokenizer
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"Model training complete! Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    train()