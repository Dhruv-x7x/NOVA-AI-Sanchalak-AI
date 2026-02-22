"""
a6on-i - Fine-Tuning Training Script
================================================
Train a custom model on your clinical trial data.

Usage:
    # Generate training data first
    python -m src.finetuning.train --prepare
    
    # Then train (requires GPU)
    python -m src.finetuning.train --train
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.finetuning.config import get_finetuning_config
from src.finetuning.data_preparation import get_data_generator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_training_data():
    """Generate training data from database."""
    print("=" * 60)
    print("PREPARING TRAINING DATA")
    print("=" * 60)
    
    generator = get_data_generator()
    
    # Generate from database
    count = generator.generate_from_database()
    print(f"\n Generated {count} training examples")
    
    # Save in chat format
    output_file = generator.save_training_data(format="chat")
    print(f" Saved to: {output_file}")
    
    # Show stats
    stats = generator.get_stats()
    print("\n Training Data Stats:")
    print(f"   Total examples: {stats['total_examples']}")
    print(f"   Categories: {stats['categories']}")
    
    return output_file


def generate_colab_notebook(data_file: Path):
    """Generate a Google Colab notebook for free GPU training."""
    config = get_finetuning_config()
    
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"}
        },
        "cells": [
            {
                "cell_type": "markdown",
                "source": ["# a6on-i - Fine-Tuning\n", 
                          "Train your custom clinical trial AI model.\n",
                          "**Runtime > Change runtime type > GPU (T4)**"]
            },
            {
                "cell_type": "code",
                "source": ["# Install dependencies\n",
                          "!pip install unsloth\n",
                          "!pip install --no-deps trl peft accelerate bitsandbytes"]
            },
            {
                "cell_type": "code", 
                "source": ["from unsloth import FastLanguageModel\n",
                          "import torch\n\n",
                          f"model, tokenizer = FastLanguageModel.from_pretrained(\n",
                          f"    model_name='{config.base_model}',\n",
                          f"    max_seq_length={config.training.max_seq_length},\n",
                          "    load_in_4bit=True,\n",
                          ")"]
            },
            {
                "cell_type": "code",
                "source": ["# Add LoRA adapters\n",
                          "model = FastLanguageModel.get_peft_model(\n",
                          "    model,\n",
                          f"    r={config.lora.r},\n",
                          f"    lora_alpha={config.lora.lora_alpha},\n",
                          "    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],\n",
                          ")"]
            },
            {
                "cell_type": "code",
                "source": ["# Upload your training data here\n",
                          "# Use Files panel to upload training_data_chat.jsonl\n",
                          "from datasets import load_dataset\n",
                          "dataset = load_dataset('json', data_files='training_data_chat.jsonl', split='train')"]
            },
            {
                "cell_type": "code",
                "source": ["from trl import SFTTrainer\n",
                          "from transformers import TrainingArguments\n\n",
                          "trainer = SFTTrainer(\n",
                          "    model=model,\n",
                          "    tokenizer=tokenizer,\n",
                          "    train_dataset=dataset,\n",
                          f"    max_seq_length={config.training.max_seq_length},\n",
                          "    args=TrainingArguments(\n",
                          f"        per_device_train_batch_size={config.training.batch_size},\n",
                          f"        num_train_epochs={config.training.num_epochs},\n",
                          f"        learning_rate={config.training.learning_rate},\n",
                          "        output_dir='outputs',\n",
                          "        fp16=True,\n",
                          "    ),\n",
                          ")\n",
                          "trainer.train()"]
            },
            {
                "cell_type": "code",
                "source": ["# Save and download the model\n",
                          f"model.save_pretrained('{config.output_model_name}')\n",
                          f"tokenizer.save_pretrained('{config.output_model_name}')\n",
                          "!zip -r model.zip a6on-i-ai-v1/"]
            }
        ]
    }
    
    notebook_path = config.base_dir / "TrialPulse_FineTuning.ipynb"
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"\n Created Colab notebook: {notebook_path}")
    print("   1. Upload to Google Colab")
    print("   2. Upload your training_data_chat.jsonl")
    print("   3. Run all cells")
    print("   4. Download the model.zip")
    
    return notebook_path


def main():
    parser = argparse.ArgumentParser(description="Fine-tune clinical trial AI model")
    parser.add_argument("--prepare", action="store_true", help="Prepare training data")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--colab", action="store_true", help="Generate Colab notebook")
    
    args = parser.parse_args()
    
    if args.prepare:
        data_file = prepare_training_data()
        print("\n Next step: Use --colab or --train")
    
    elif args.colab:
        config = get_finetuning_config()
        data_file = config.data_dir / "training_data_chat.jsonl"
        generate_colab_notebook(data_file)
    
    elif args.train:
        config = get_finetuning_config()
        data_file = config.data_dir / "training_data_chat.jsonl"
        
        if not data_file.exists():
            print(" Training data not found. Run --prepare first.")
            return
        
        # Check GPU
        try:
            import torch
            if not torch.cuda.is_available():
                print(" No GPU detected. Generating Colab notebook instead...")
                generate_colab_notebook(data_file)
                return
        except:
            print(" PyTorch not found. Generating Colab notebook...")
            generate_colab_notebook(data_file)
            return
        
        print("GPU training would start here...")
        print("For now, use Google Colab for free GPU access.")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
