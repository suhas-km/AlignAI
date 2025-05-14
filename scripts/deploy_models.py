#!/usr/bin/env python3
"""
Script to deploy trained ML models to the backend for use in the AlignAI application.
This script copies the trained models from their training directories to the
appropriate locations in the backend directory structure.
"""

import os
import shutil
import argparse
from pathlib import Path

def create_directories(base_dir):
    """Create necessary directories if they don't exist."""
    model_dirs = [
        "bias_detection",
        "pii_detection",
        "policy_detection"
    ]
    
    for model_dir in model_dirs:
        path = os.path.join(base_dir, "backend", "models", model_dir)
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")

def copy_models(base_dir, models_to_copy=None):
    """Copy trained models to the backend model directories."""
    # Define source and destination paths
    model_paths = {
        "bias_detection": {
            "source": os.path.join(base_dir, "Model-Training", "bias-detection", "bias_model_weights"),
            "dest": os.path.join(base_dir, "backend", "models", "bias_detection")
        },
        "pii_detection": {
            # Updated path for PII detection model based on project structure
            "source": os.path.join(base_dir, "Model-Training", "pII-detection", "models", "pii_detection"),
            "dest": os.path.join(base_dir, "backend", "models", "pii_detection")
        },
        "policy_detection": {
            # Updated path for policy detection model based on project structure
            "source": os.path.join(base_dir, "Model-Training", "policy-detection", "models", "policy_detection"),
            "dest": os.path.join(base_dir, "backend", "models", "policy_detection")
        }
    }
    
    # If specific models are specified, filter the dictionary
    if models_to_copy:
        model_paths = {k: v for k, v in model_paths.items() if k in models_to_copy}
    
    for model_name, paths in model_paths.items():
        source_dir = paths['source']
        dest_dir = paths['dest']
        
        if not os.path.exists(source_dir):
            print(f"Warning: Source directory not found: {source_dir}")
            continue
        
        # Clear destination directory
        if os.path.exists(dest_dir):
            for item in os.listdir(dest_dir):
                item_path = os.path.join(dest_dir, item)
                if os.path.isfile(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
        
        # Copy all files from source to destination
        for item in os.listdir(source_dir):
            s = os.path.join(source_dir, item)
            d = os.path.join(dest_dir, item)
            if os.path.isfile(s):
                shutil.copy2(s, d)
                print(f"Copied: {s} -> {d}")
            elif os.path.isdir(s):
                shutil.copytree(s, d)
                print(f"Copied directory: {s} -> {d}")
        
        print(f"Deployed {model_name} model")

def main():
    parser = argparse.ArgumentParser(description='Deploy trained ML models to the backend.')
    parser.add_argument('--models', nargs='+', choices=['bias_detection', 'pii_detection', 'policy_detection'],
                        help='Specific models to deploy. If not specified, all models will be deployed.')
    args = parser.parse_args()
    
    # Get the base directory path
    base_dir = Path(__file__).resolve().parent.parent
    
    print(f"Base directory: {base_dir}")
    
    # Create necessary directories
    create_directories(base_dir)
    
    # Copy models
    copy_models(base_dir, args.models)
    
    print("Model deployment complete!")

if __name__ == "__main__":
    main()
