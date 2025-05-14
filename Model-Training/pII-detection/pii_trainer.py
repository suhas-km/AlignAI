#!/usr/bin/env python3
"""
PII Detection Model Trainer
------------------------
This script fine-tunes a transformer model for PII (Personally Identifiable Information) detection
using a token classification approach. The model identifies various types of PII in text.
"""

import os
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import logging
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, f1_score

# HuggingFace imports
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    set_seed
)
from datasets import Dataset, load_from_disk

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define constants
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = Path(__file__).parent / "models"
DATA_DIR = Path(__file__).parent / "data"
BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 5e-5
MAX_SEQ_LENGTH = 128
SEED = 42

# Set random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
set_seed(SEED)

def load_and_prepare_data():
    """Load and prepare the PII detection dataset"""
    logger.info("Loading PII detection dataset...")
    
    # Check if dataset exists, if not, download it
    if not (DATA_DIR / "train.json").exists():
        logger.info("Dataset not found. Running the download script...")
        from download_pii_dataset import download_and_prepare_pii_dataset
        download_and_prepare_pii_dataset()
    
    # Load the dataset
    with open(DATA_DIR / "train.json", "r") as f:
        train_data = json.load(f)
    
    with open(DATA_DIR / "val.json", "r") as f:
        val_data = json.load(f)
    
    # Load dataset info to get PII categories
    with open(DATA_DIR / "dataset_info.json", "r") as f:
        dataset_info = json.load(f)
    
    # Create ID to label mapping
    unique_tags = set()
    for example in train_data + val_data:
        if "pii_tags" in example:
            unique_tags.update(example["pii_tags"])
        if "labels" in example:
            unique_tags.update(example["labels"])
    
    unique_tags = sorted(list(unique_tags))
    tag2id = {tag: i for i, tag in enumerate(unique_tags)}
    id2tag = {i: tag for i, tag in enumerate(unique_tags)}
    
    # Convert token tags to IDs
    for example in train_data + val_data:
        if "pii_tags" in example:
            example["tag_ids"] = [tag2id[tag] for tag in example["pii_tags"]]
    
    # Create HuggingFace datasets
    train_dataset = Dataset.from_dict({
        "tokens": [ex["tokens"] if "tokens" in ex else ex["text"].split() for ex in train_data],
        "tags": [ex["tag_ids"] if "tag_ids" in ex else [tag2id["O"]] * len(ex["text"].split()) for ex in train_data],
        "text": [ex["text"] for ex in train_data]
    })
    
    val_dataset = Dataset.from_dict({
        "tokens": [ex["tokens"] if "tokens" in ex else ex["text"].split() for ex in val_data],
        "tags": [ex["tag_ids"] if "tag_ids" in ex else [tag2id["O"]] * len(ex["text"].split()) for ex in val_data],
        "text": [ex["text"] for ex in val_data]
    })
    
    logger.info(f"Loaded {len(train_dataset)} training examples and {len(val_dataset)} validation examples")
    
    return train_dataset, val_dataset, tag2id, id2tag

def tokenize_and_align_labels(examples, tokenizer, tag2id):
    """Tokenize inputs and align labels for token classification"""
    tokenized_inputs = tokenizer(
        examples["tokens"], 
        truncation=True, 
        is_split_into_words=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length"
    )
    
    labels = []
    for i, label in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        
        for word_id in word_ids:
            if word_id is None:
                # Special tokens get -100 (ignored in loss calculation)
                label_ids.append(-100)
            elif word_id >= len(label):
                # If the word_id is out of bounds, use the "O" tag
                label_ids.append(tag2id["O"] if "O" in tag2id else 0)
            else:
                # Use the actual label
                label_ids.append(label[word_id])
        
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(pred):
    """Compute evaluation metrics for token classification"""
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=2)
    
    # Remove ignored index (special tokens)
    true_predictions = [
        [id2tag[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2tag[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    # Flatten for overall metrics
    flat_predictions = [p for preds in true_predictions for p in preds]
    flat_labels = [l for labels in true_labels for l in labels]
    
    # Return overall metrics
    return {
        "accuracy": accuracy_score(flat_labels, flat_predictions),
        "f1": f1_score(flat_labels, flat_predictions, average="weighted")
    }

def train_pii_detection_model():
    """Train a token classification model for PII detection"""
    logger.info("Starting PII detection model training...")
    
    # Load tokenizer and prepare datasets
    logger.info(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load and prepare data
    logger.info("Loading and preparing data...")
    global id2tag  # Make id2tag available for compute_metrics
    train_dataset, val_dataset, tag2id, id2tag = load_and_prepare_data()
    
    # Initialize the model
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(tag2id),
        id2label=id2tag,
        label2id=tag2id
    )
    
    # Tokenize and align labels
    logger.info("Tokenizing datasets...")
    tokenize_function = lambda examples: tokenize_and_align_labels(examples, tokenizer, tag2id)
    
    # Apply tokenization to datasets
    tokenized_train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    tokenized_val_dataset = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=val_dataset.column_names
    )
    
    # Data collator for padding
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    # Set up training arguments for compatibility with older transformers versions
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "pii_detection"),
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        logging_dir=str(OUTPUT_DIR / "logs"),
        logging_steps=10,
        # Simplified args compatible with older versions
        save_total_limit=2,   # Only keep the 2 best checkpoints
        do_eval=True,         # Enable evaluation
        load_best_model_at_end=False,
        # Disable wandb and other tracking
        report_to="none"
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # Train the model
    logger.info("Starting PII detection model training...")
    trainer.train()
    
    # Evaluate the model
    logger.info("Evaluating model...")
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")
    
    # Save the final model
    model_path = OUTPUT_DIR / "pii_detection" / "final_model"
    model.save_pretrained(str(model_path))
    tokenizer.save_pretrained(str(model_path))
    
    # Save metadata
    with open(model_path / "metadata.json", "w") as f:
        metadata = {
            "model_name": MODEL_NAME,
            "task": "pii_detection",
            "evaluation_metrics": eval_results,
            "max_sequence_length": MAX_SEQ_LENGTH,
            "id2tag": id2tag,
            "tag2id": tag2id,
            "training_parameters": {
                "epochs": NUM_EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE
            }
        }
        json.dump(metadata, f, indent=2)
    
    logger.info(f"PII detection model training complete! Model saved to {model_path}")
    
    return model_path

if __name__ == "__main__":
    train_pii_detection_model()
