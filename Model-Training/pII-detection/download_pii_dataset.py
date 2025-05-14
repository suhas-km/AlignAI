#!/usr/bin/env python3
"""
PII Dataset Downloader
---------------------
This script downloads and prepares datasets for PII (Personally Identifiable Information) detection.
It uses standard NER datasets and augments them with synthetic PII data for training.
"""

import os
import json
import random
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset
from pathlib import Path
import logging
from tqdm import tqdm
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define constants
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# PII categories mapping
PII_CATEGORIES = {
    "PERSON": "PII-PERSON",       # Names of individuals
    "ORG": "PII-ORGANIZATION",    # Company/organization names
    "LOC": "PII-LOCATION",        # Locations (cities, countries)
    "EMAIL": "PII-EMAIL",         # Email addresses
    "PHONE": "PII-PHONE",         # Phone numbers
    "SSN": "PII-GOVERNMENT_ID",   # Social Security Numbers
    "CREDIT_CARD": "PII-FINANCIAL", # Credit card numbers
    "DOB": "PII-DATE",            # Date of birth
    "ADDRESS": "PII-ADDRESS",     # Physical addresses
    "IP": "PII-IP_ADDRESS",       # IP addresses
    "OTHER": "O"                  # Not PII
}

def generate_synthetic_pii():
    """Generate synthetic PII examples to supplement the dataset"""
    synthetic_data = []
    
    # Generate synthetic person names
    first_names = ["John", "Mary", "Michael", "Sarah", "David", "Linda", "James", "Patricia"]
    last_names = ["Smith", "Jones", "Brown", "Johnson", "Williams", "Davis", "Miller", "Wilson"]
    
    # Generate 100 synthetic PII examples
    for _ in range(100):
        name = f"{random.choice(first_names)} {random.choice(last_names)}"
        email = f"{name.lower().replace(' ', '.')}@example.com"
        phone = f"+1-{random.randint(200, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
        ssn = f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}"
        cc = f"{random.randint(1000, 9999)}-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}"
        dob = f"{random.randint(1, 12)}/{random.randint(1, 28)}/{random.randint(1940, 2000)}"
        ip = f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
        
        synthetic_data.append({
            "text": f"Name: {name}, Email: {email}, Phone: {phone}",
            "labels": ["PII-PERSON", "PII-EMAIL", "PII-PHONE"]
        })
        synthetic_data.append({
            "text": f"SSN: {ssn}, Credit Card: {cc}, DOB: {dob}",
            "labels": ["PII-GOVERNMENT_ID", "PII-FINANCIAL", "PII-DATE"]
        })
        synthetic_data.append({
            "text": f"The customer with IP {ip} made a purchase using card ending in {cc[-4:]}.",
            "labels": ["PII-IP_ADDRESS", "PII-FINANCIAL"]
        })
    
    return synthetic_data

def convert_wnut_to_pii_format(wnut_example):
    """Convert WNUT NER examples to PII format"""
    tokens = wnut_example["tokens"]
    ner_tags = wnut_example["ner_tags"]
    
    # WNUT tag mapping based on numeric IDs
    # 0 = 'O', 1 = 'B-corporation', 2 = 'I-corporation', 3 = 'B-creative-work',
    # 4 = 'I-creative-work', 5 = 'B-group', 6 = 'I-group', 7 = 'B-location',
    # 8 = 'I-location', 9 = 'B-person', 10 = 'I-person', 11 = 'B-product', 12 = 'I-product'
    wnut_tag_map = {
        0: "O",               # O (Outside any entity)
        1: "PII-ORGANIZATION", # B-corporation
        2: "PII-ORGANIZATION", # I-corporation
        3: "O",               # B-creative-work
        4: "O",               # I-creative-work
        5: "O",               # B-group
        6: "O",               # I-group
        7: "PII-LOCATION",    # B-location
        8: "PII-LOCATION",    # I-location
        9: "PII-PERSON",      # B-person
        10: "PII-PERSON",     # I-person
        11: "O",              # B-product
        12: "O"               # I-product
    }
    
    # Map the original WNUT tags to PII tags
    pii_tags = [wnut_tag_map.get(tag, "O") for tag in ner_tags]
    
    return {
        "tokens": tokens,
        "pii_tags": pii_tags,
        "text": " ".join(tokens)
    }

def download_and_prepare_pii_dataset():
    """Download and prepare the PII detection dataset"""
    logger.info("Downloading and preparing PII detection dataset...")
    
    # Download WNUT-17 dataset (a standard NER dataset)
    logger.info("Loading WNUT-17 dataset for Named Entity Recognition")
    try:
        wnut_dataset = load_dataset("wnut_17", trust_remote_code=True)
        logger.info(f"WNUT-17 dataset loaded successfully with {len(wnut_dataset['train'])} training examples")
    except Exception as e:
        logger.error(f"Error loading WNUT-17 dataset: {e}")
        logger.info("Creating synthetic dataset only...")
        wnut_dataset = None
    
    # Generate synthetic PII examples
    logger.info("Generating synthetic PII examples")
    synthetic_pii = generate_synthetic_pii()
    
    # Prepare training, validation, and test datasets
    train_examples = []
    val_examples = []
    test_examples = []
    
    # Add converted WNUT examples if available
    if wnut_dataset is not None:
        # Convert training examples
        for example in tqdm(wnut_dataset["train"], desc="Processing training examples"):
            train_examples.append(convert_wnut_to_pii_format(example))
        
        # Convert validation examples
        for example in tqdm(wnut_dataset["validation"], desc="Processing validation examples"):
            val_examples.append(convert_wnut_to_pii_format(example))
        
        # Convert test examples
        for example in tqdm(wnut_dataset["test"], desc="Processing test examples"):
            test_examples.append(convert_wnut_to_pii_format(example))
    
    # Add synthetic examples
    random.shuffle(synthetic_pii)
    train_split = int(len(synthetic_pii) * 0.7)
    val_split = int(len(synthetic_pii) * 0.85)
    
    # Add synthetic examples to appropriate splits
    for i, example in enumerate(synthetic_pii):
        if i < train_split:
            train_examples.append(example)
        elif i < val_split:
            val_examples.append(example)
        else:
            test_examples.append(example)
    
    # Save datasets to disk
    logger.info(f"Saving datasets with {len(train_examples)} training, {len(val_examples)} validation, and {len(test_examples)} test examples")
    
    # Save as JSON files
    with open(DATA_DIR / "train.json", "w") as f:
        json.dump(train_examples, f)
    
    with open(DATA_DIR / "val.json", "w") as f:
        json.dump(val_examples, f)
    
    with open(DATA_DIR / "test.json", "w") as f:
        json.dump(test_examples, f)
    
    # Save dataset info
    dataset_info = {
        "description": "PII detection dataset combining WNUT-17 NER data and synthetic PII examples",
        "num_train_examples": len(train_examples),
        "num_val_examples": len(val_examples),
        "num_test_examples": len(test_examples),
        "pii_categories": PII_CATEGORIES
    }
    
    with open(DATA_DIR / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    logger.info("PII dataset preparation complete!")
    return dataset_info

if __name__ == "__main__":
    download_and_prepare_pii_dataset()
