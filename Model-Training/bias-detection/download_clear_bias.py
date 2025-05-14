#!/usr/bin/env python3
"""
CLEAR-Bias Dataset Downloader
-----------------------------
This script downloads the CLEAR-Bias dataset from Hugging Face and 
saves it in the appropriate format for our bias detection model training.

Before running, make sure to authenticate with Hugging Face by running:
    huggingface-cli login
"""

import os
import json
from pathlib import Path
from datasets import load_dataset
import pandas as pd
from huggingface_hub import login

# Configuration
OUTPUT_DIR = Path("data")
DATASET_NAME = "RCantini/CLEAR-Bias"
DATASET_CONFIGS = ["base_prompts", "control_set", "jailbreak_prompts"]
SPLIT_NAMES = ["age", "disability", "ethnicity", "gender", "religion", "sexual_orientation", "socioeconomic"]

def check_hf_auth():
    """Check if user is authenticated with Hugging Face."""
    try:
        from huggingface_hub.hf_api import HfFolder
        return HfFolder.get_token() is not None
    except Exception as e:
        print(f"Error checking Hugging Face authentication: {e}")
        return False

def download_dataset():
    """Download and process the CLEAR-Bias dataset."""
    try:
        print(f"Downloading CLEAR-Bias dataset from {DATASET_NAME}...")
        
        # Load all configurations of the dataset
        datasets = {}
        for config in DATASET_CONFIGS:
            print(f"\nDownloading configuration: {config}")
            datasets[config] = load_dataset(
                DATASET_NAME,
                config,
                download_mode="force_redownload"
            )
        return datasets
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nMake sure you're authenticated with Hugging Face by running:")
        print("huggingface-cli login")
        print("Then enter your access token when prompted.")
        return None

def save_dataset_info(datasets):
    """Save dataset metadata to a JSON file."""
    try:
        info = {
            "name": "CLEAR-Bias",
            "description": "Corpus for Linguistic Evaluation of Adversarial Robustness against Bias",
            "configurations": {}
        }
        
        for config_name, dataset in datasets.items():
            info["configurations"][config_name] = {
                "total_samples": sum(len(dataset[split]) for split in dataset),
                "splits": {split: len(dataset[split]) for split in dataset}
            }
        
        with open(OUTPUT_DIR / "dataset_info.json", "w") as f:
            json.dump(info, f, indent=2)
        print(f"\nDataset info saved to {OUTPUT_DIR / 'dataset_info.json'}")
    except Exception as e:
        print(f"Error saving dataset info: {e}")

def save_splits_to_csv(datasets):
    """Save each dataset split as a separate CSV file."""
    for config_name, dataset in datasets.items():
        config_dir = OUTPUT_DIR / config_name
        config_dir.mkdir(exist_ok=True, parents=True)
        
        for split in dataset:
            try:
                df = pd.DataFrame(dataset[split])
                output_file = config_dir / f"{split}.csv"
                df.to_csv(output_file, index=False)
                print(f"Saved {len(df)} examples to {output_file}")
            except Exception as e:
                print(f"Error saving {config_name}/{split} split: {e}")

def create_sample_file(datasets):
    """Create a sample file with examples from each category."""
    try:
        all_samples = []
        for config_name, dataset in datasets.items():
            for split in dataset:
                if split in SPLIT_NAMES:  # Only process bias categories
                    # Get up to 2 samples from each category
                    for i in range(min(2, len(dataset[split]))):
                        sample = dataset[split][i]
                        if isinstance(sample, dict):
                            sample["config"] = config_name
                            sample["category"] = split
                            all_samples.append(sample)
        
        if all_samples:
            samples_df = pd.DataFrame(all_samples)
            samples_df.to_csv(OUTPUT_DIR / "samples.csv", index=False)
            print(f"Saved {len(samples_df)} sample examples to {OUTPUT_DIR / 'samples.csv'}")
    except Exception as e:
        print(f"Error creating sample file: {e}")

def main():
    """Main function to download and process the CLEAR-Bias dataset."""
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    # Check authentication
    if not check_hf_auth():
        print("Please authenticate with Hugging Face by running:")
        print("huggingface-cli login")
        print("Then run this script again.")
        return
    
    # Download the dataset
    datasets = download_dataset()
    if datasets is None:
        return
    
    # Process and save the dataset
    save_dataset_info(datasets)
    save_splits_to_csv(datasets)
    create_sample_file(datasets)
    
    print("\nCLEAR-Bias dataset has been successfully downloaded and processed!")
    print(f"Data is located in: {OUTPUT_DIR.absolute()}")
    print("\nNext steps:")
    print("1. Explore the samples.csv file to understand the data structure")
    print("2. Start building your bias detection model using data_pre_script.py")
    print("3. Fine-tune a classifier on this dataset to detect various forms of bias")

if __name__ == "__main__":
    main()