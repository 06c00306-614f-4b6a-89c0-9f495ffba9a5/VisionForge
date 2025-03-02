#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vision-Language Model Fine-tuning Script
========================================

This script enables fine-tuning of large vision-language models on custom image datasets.
It utilizes Parameter-Efficient Fine-Tuning (PEFT) to adapt pre-trained models to specific
domains (e.g., medical imaging) without requiring full model retraining.

Features:
- Supports quantized fine-tuning (4-bit) for memory efficiency
- Fine-tunes vision, language, and cross-attention components
- Configurable LoRA parameters for adaptation control
- Includes before/after evaluation to demonstrate improvements
- Provides HuggingFace Hub integration for model sharing

Requirements:
- PyTorch
- Transformers
- Unsloth
- Datasets
- TRL

Author: Derek Lofaro (Software Engineer)
Date: March 3, 2025
"""

# Standard library imports
import os
import sys
import argparse
import logging
import subprocess
from datetime import datetime

# Third-party imports
import torch
from datasets import load_dataset
from transformers import TextStreamer, set_seed

# Fine-tuning related imports
from trl import SFTTrainer, SFTConfig
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "model_name": "unsloth/Llama-3.2-11B-Vision-Instruct",
    "dataset_name": "unsloth/Radiology_mini",
    "lora_rank": 16,
    "batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "max_steps": 30,
    "warmup_steps": 5,
    "output_dir": "output",
    "max_seq_length": 2048,
    "max_new_tokens": 128,
    "seed": 3407
}


def setup_argument_parser():
    """
    Configure command-line argument parser for customizing the fine-tuning process.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune vision-language models on custom datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--model_name", type=str, default=DEFAULT_CONFIG["model_name"],
                        help="Pre-trained model name or path")
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_CONFIG["dataset_name"],
                        help="Dataset name on HuggingFace Hub or local path")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_CONFIG["output_dir"],
                        help="Directory to save model checkpoints")
    parser.add_argument("--lora_rank", type=int, default=DEFAULT_CONFIG["lora_rank"],
                        help="Rank for LoRA adaptation")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"],
                        help="Per-device training batch size")
    parser.add_argument("--max_steps", type=int, default=DEFAULT_CONFIG["max_steps"],
                        help="Maximum number of training steps")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_CONFIG["learning_rate"],
                        help="Learning rate for training")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"],
                        help="Random seed for reproducibility")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace API token for model pushing")
    parser.add_argument("--hub_model_id", type=str, default=None,
                        help="Model ID for HuggingFace Hub (username/model_name)")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip before/after evaluation")
    parser.add_argument("--skip_push", action="store_true",
                        help="Skip pushing model to HuggingFace Hub")
    
    return parser.parse_args()


def install_requirements():
    """
    Install required packages for fine-tuning vision language models.
    
    This function ensures all dependencies are properly installed before
    beginning the fine-tuning process.
    
    Returns:
        bool: True if installation successful, False otherwise
    """
    try:
        # List of packages to install
        packages = ["unsloth", "transformers>=4.37.0", "accelerate>=0.27.1", "trl>=0.7.6"]
        
        # Ensure pip is installed and up-to-date
        logger.info("Updating pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install each package
        for package in packages:
            logger.info(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            logger.info(f"{package} installed successfully.")
            
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing requirements: {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during installation: {e}")
        return False


def setup_vision_language_model(model_name, lora_rank):
    """
    Initialize and configure a vision-language model for fine-tuning.
    
    This function loads a pre-trained model and applies PEFT configuration
    for efficient adaptation to new domains.
    
    Args:
        model_name (str): Name or path of the pre-trained model
        lora_rank (int): Rank parameter for LoRA adaptation
        
    Returns:
        tuple: (model, tokenizer) - The configured model and its tokenizer
    """
    logger.info(f"Loading model: {model_name}")
    
    try:
        # Load the pre-trained vision-language model and tokenizer
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name,
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
        )
        
        # Apply PEFT configuration
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=True,
            finetune_language_layers=True,
            finetune_attention_layers=True,
            finetune_mlp_modules=True,
            r=lora_rank,
            lora_alpha=lora_rank,
            lora_dropout=0,
            bias="none",
            random_state=DEFAULT_CONFIG["seed"],
            use_rslora=False,
            loftq_config=None,
        )
        
        logger.info(f"Model successfully loaded and configured with LoRA rank {lora_rank}")
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"Error setting up model: {e}")
        raise


def load_and_prepare_dataset(dataset_name, instruction):
    """
    Load and format dataset for vision-language model fine-tuning.
    
    Args:
        dataset_name (str): Dataset name on HuggingFace or local path
        instruction (str): Instruction prepended to each training example
        
    Returns:
        list: Formatted dataset ready for training
    """
    logger.info(f"Loading dataset: {dataset_name}")
    
    try:
        # Load the dataset
        dataset = load_dataset(dataset_name, split="train")
        logger.info(f"Loaded {len(dataset)} training examples")
        
        # Convert dataset to conversation format
        def convert_to_conversation(sample):
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image", "image": sample["image"]}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": sample["caption"]}
                    ]
                },
            ]
            return {"messages": conversation}
        
        converted_dataset = [convert_to_conversation(sample) for sample in dataset]
        logger.info("Dataset successfully converted to conversation format")
        
        return dataset, converted_dataset
    
    except Exception as e:
        logger.error(f"Error preparing dataset: {e}")
        raise


def evaluate_model(model, tokenizer, image, instruction, label="Evaluation"):
    """
    Generate a prediction from the model on a sample image.
    
    Args:
        model (FastVisionModel): The model to evaluate
        tokenizer: The tokenizer for the model
        image: Sample image for evaluation
        instruction (str): Text instruction for the model
        label (str): Label for the evaluation (e.g., "Before Training")
        
    Returns:
        None, prints the model output
    """
    logger.info(f"Running {label} evaluation...")
    
    # Set model to inference mode
    FastVisionModel.for_inference(model)
    
    # Prepare input
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": instruction}
        ]}
    ]
    
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate output
    print(f"\n{label}:\n")
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(
        **inputs, 
        streamer=text_streamer, 
        max_new_tokens=DEFAULT_CONFIG["max_new_tokens"],
        use_cache=True,
        temperature=1.5, 
        min_p=0.1
    )
    print("\n")


def train_model(model, tokenizer, dataset, args):
    """
    Fine-tune the model on the prepared dataset.
    
    Args:
        model (FastVisionModel): Model to be fine-tuned
        tokenizer: Tokenizer for the model
        dataset (list): Prepared dataset for training
        args (argparse.Namespace): Training configuration arguments
        
    Returns:
        dict: Training statistics
    """
    logger.info("Starting model fine-tuning...")
    
    # Set model to training mode
    FastVisionModel.for_training(model)
    
    # Configure SFT trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=dataset,
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=DEFAULT_CONFIG["gradient_accumulation_steps"],
            warmup_steps=DEFAULT_CONFIG["warmup_steps"],
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=args.seed,
            output_dir=args.output_dir,
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            dataset_num_proc=4,
            max_seq_length=DEFAULT_CONFIG["max_seq_length"],
        ),
    )
    
    # Log GPU information
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        logger.info(f"{start_gpu_memory} GB of memory reserved at start.")
    
    # Run training
    trainer_stats = trainer.train()
    
    # Log training statistics
    if torch.cuda.is_available():
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
        
        logger.info(f"Training completed in {trainer_stats.metrics['train_runtime']:.2f} seconds "
                    f"({trainer_stats.metrics['train_runtime']/60:.2f} minutes).")
        logger.info(f"Peak reserved memory = {used_memory} GB.")
        logger.info(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        logger.info(f"Peak reserved memory % of max memory = {used_percentage}%.")
        logger.info(f"Peak reserved memory % of max memory for LoRA = {lora_percentage}%.")
    
    return trainer_stats


def save_model(model, tokenizer, args):
    """
    Save the fine-tuned model locally and optionally to HuggingFace Hub.
    
    Args:
        model (FastVisionModel): Fine-tuned model to save
        tokenizer: Tokenizer for the model
        args (argparse.Namespace): Configuration arguments
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create local output directory if it doesn't exist
        os.makedirs("lora_model", exist_ok=True)
        
        # Save LoRA adapters locally
        logger.info("Saving LoRA adapters to 'lora_model' directory...")
        model.save_pretrained("lora_model")
        tokenizer.save_pretrained("lora_model")
        
        # Push to HuggingFace Hub if configured
        if not args.skip_push and args.hub_model_id:
            if not args.hf_token:
                logger.warning("No HuggingFace token provided. Using environment variable HF_TOKEN if available.")
            
            hf_token = args.hf_token or os.environ.get("HF_TOKEN")
            
            if hf_token:
                logger.info(f"Saving merged model to HuggingFace Hub as {args.hub_model_id}...")
                model.push_to_hub_merged(
                    args.hub_model_id,
                    tokenizer,
                    save_method="merged_16bit",
                    token=hf_token
                )
                logger.info("Model successfully pushed to HuggingFace Hub.")
            else:
                logger.error("HuggingFace token not found. Model not pushed to Hub.")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return False


def main():
    """
    Main execution function to orchestrate the fine-tuning process.
    """
    # Parse command-line arguments
    args = setup_argument_parser()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Install requirements
    if not install_requirements():
        logger.error("Failed to install requirements. Exiting.")
        sys.exit(1)
    
    # Initialize model and tokenizer
    model, tokenizer = setup_vision_language_model(args.model_name, args.lora_rank)
    
    # Define the instruction for the dataset
    instruction = """
    You are an expert radiographer. Describe accurately what you see
    in the provided images as detailed, concise, and professional as 
    possible.
    """
    
    # Load and prepare dataset
    raw_dataset, converted_dataset = load_and_prepare_dataset(args.dataset_name, instruction)
    
    # Evaluate model before training (if not skipped)
    if not args.skip_eval:
        evaluate_model(
            model, tokenizer, raw_dataset[0]["image"], instruction, "Before Training"
        )
    
    # Train the model
    train_stats = train_model(model, tokenizer, converted_dataset, args)
    
    # Evaluate model after training (if not skipped)
    if not args.skip_eval:
        evaluate_model(
            model, tokenizer, raw_dataset[0]["image"], instruction, "After Training"
        )
    
    # Save the model
    save_model(model, tokenizer, args)
    
    logger.info("Fine-tuning process completed successfully.")


if __name__ == "__main__":
    # Record start time
    start_time = datetime.now()
    
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)
    
    # Calculate and log total runtime
    elapsed_time = datetime.now() - start_time
    logger.info(f"Total runtime: {elapsed_time}")
