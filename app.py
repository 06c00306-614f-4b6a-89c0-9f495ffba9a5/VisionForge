#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Standard library imports for basic functionality
import os
import sys
import argparse
import logging
import subprocess
from datetime import datetime

#PyTorch is the backbone deep learning framework
import torch
#Datasets library handles loading and processing training data
from datasets import load_dataset
#TextStreamer for real-time generation output, set_seed for reproducibility
from transformers import TextStreamer, set_seed

#SFTTrainer and SFTConfig enable supervised fine-tuning
from trl import SFTTrainer, SFTConfig
#FastVisionModel provides optimized loading of vision-language models
from unsloth import FastVisionModel, is_bf16_supported
#Custom data collator for handling vision-language data during training
from unsloth.trainer import UnslothVisionDataCollator

#Configure logging to track the fine-tuning process in real-time
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
#Create logger instance for this script
logger = logging.getLogger(__name__)

#Default configuration parameters optimized for vision-language model fine-tuning
DEFAULT_CONFIG = {
    #Base model selection - Llama-3.2-11B-Vision has excellent multimodal capabilities
    "model_name": "unsloth/Llama-3.2-11B-Vision-Instruct",
    #Medical imaging dataset for domain adaptation
    "dataset_name": "unsloth/Radiology_mini",
    #LoRA rank controls parameter efficiency vs. adaptation capability
    "lora_rank": 16,
    #Small batch size to accommodate GPU memory constraints
    "batch_size": 2,
    #Gradient accumulation compensates for small batch sizes
    "gradient_accumulation_steps": 4,
    #Higher learning rate works well with LoRA fine-tuning
    "learning_rate": 2e-4,
    #Limited steps to demonstrate improvements quickly
    "max_steps": 30,
    #Gradual learning rate warmup prevents early training instability
    "warmup_steps": 5,
    #Directory for saving checkpoints and outputs
    "output_dir": "output",
    #Maximum sequence length balances context window and memory usage
    "max_seq_length": 2048,
    #Controls response generation length
    "max_new_tokens": 128,
    #Random seed for reproducible results
    "seed": 3407
}


def setup_argument_parser():
    #Create argument parser for command-line customization
    parser = argparse.ArgumentParser(
        description="Fine-tune vision-language models on custom datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    #Pre-trained model selection parameter
    parser.add_argument("--model_name", type=str, default=DEFAULT_CONFIG["model_name"],
                        help="Pre-trained model name or path")
    #Dataset selection parameter
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_CONFIG["dataset_name"],
                        help="Dataset name on HuggingFace Hub or local path")
    #Output directory parameter for saving fine-tuned model
    parser.add_argument("--output_dir", type=str, default=DEFAULT_CONFIG["output_dir"],
                        help="Directory to save model checkpoints")
    #LoRA rank controls parameter efficiency vs adaptation capability
    parser.add_argument("--lora_rank", type=int, default=DEFAULT_CONFIG["lora_rank"],
                        help="Rank for LoRA adaptation")
    #Batch size affects memory usage and training speed
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"],
                        help="Per-device training batch size")
    #Maximum steps controls training duration
    parser.add_argument("--max_steps", type=int, default=DEFAULT_CONFIG["max_steps"],
                        help="Maximum number of training steps")
    #Learning rate affects convergence speed and quality
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_CONFIG["learning_rate"],
                        help="Learning rate for training")
    #Random seed ensures reproducible experiments
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"],
                        help="Random seed for reproducibility")
    #HuggingFace token for model uploading
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace API token for model pushing")
    #Model ID for HuggingFace Hub uploading
    parser.add_argument("--hub_model_id", type=str, default=None,
                        help="Model ID for HuggingFace Hub (username/model_name)")
    #Option to skip evaluation to save time
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip before/after evaluation")
    #Option to skip model uploading
    parser.add_argument("--skip_push", action="store_true",
                        help="Skip pushing model to HuggingFace Hub")
    
    #Return parsed arguments
    return parser.parse_args()


def install_requirements():
    try:
        #Define critical packages needed for vision-language fine-tuning
        packages = ["unsloth", "transformers>=4.37.0", "accelerate>=0.27.1", "trl>=0.7.6"]
        
        #Update pip first to ensure compatibility with latest packages
        logger.info("Updating pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        #Install each package sequentially to handle dependencies properly
        for package in packages:
            logger.info(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            logger.info(f"{package} installed successfully.")
            
        #Return success status
        return True
    except subprocess.CalledProcessError as e:
        #Log specific subprocess errors during installation
        logger.error(f"Error installing requirements: {e}")
        return False
    except Exception as e:
        #Catch any other unexpected errors
        logger.error(f"An unexpected error occurred during installation: {e}")
        return False


def setup_vision_language_model(model_name, lora_rank):
    logger.info(f"Loading model: {model_name}")
    
    try:
        #Load pre-trained model with 4-bit quantization to reduce memory usage
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name,
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",  # Enable gradient checkpointing to save memory
        )
        
        #Configure Parameter-Efficient Fine-Tuning (PEFT) using LoRA
        model = FastVisionModel.get_peft_model(
            model,
            #Enable fine-tuning of vision encoder for domain adaptation
            finetune_vision_layers=True,
            #Enable fine-tuning of language layers for specialized vocabulary
            finetune_language_layers=True,
            #Enable fine-tuning of cross-attention for multimodal understanding
            finetune_attention_layers=True,
            #Enable MLP fine-tuning for richer representations
            finetune_mlp_modules=True,
            #LoRA rank controls parameter efficiency vs. adaptation capability
            r=lora_rank,
            #LoRA alpha scales the LoRA adaptation contribution
            lora_alpha=lora_rank,
            #Disabling dropout for more stable fine-tuning
            lora_dropout=0,
            #No bias terms to reduce parameter count
            bias="none",
            #Set random state for reproducibility
            random_state=DEFAULT_CONFIG["seed"],
            #Disable rank-stabilized LoRA which benefits longer training
            use_rslora=False,
            #Disable LoftQ quantization since we're using 4-bit quantization
            loftq_config=None,
        )
        
        #Log successful model configuration
        logger.info(f"Model successfully loaded and configured with LoRA rank {lora_rank}")
        return model, tokenizer
    
    except Exception as e:
        #Log detailed error information for troubleshooting
        logger.error(f"Error setting up model: {e}")
        raise


def load_and_prepare_dataset(dataset_name, instruction):
    logger.info(f"Loading dataset: {dataset_name}")
    
    try:
        #Load dataset from HuggingFace Hub or local path
        dataset = load_dataset(dataset_name, split="train")
        logger.info(f"Loaded {len(dataset)} training examples")
        
        #Define conversion function for multimodal conversation format
        def convert_to_conversation(sample):
            #Create conversation in format expected by vision-language models
            conversation = [
                {
                    #User message contains text instruction and image
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image", "image": sample["image"]}
                    ]
                },
                {
                    #Assistant response contains target text for training
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": sample["caption"]}
                    ]
                },
            ]
            #Return formatted conversation
            return {"messages": conversation}
        
        #Apply conversion function to all dataset samples
        converted_dataset = [convert_to_conversation(sample) for sample in dataset]
        logger.info("Dataset successfully converted to conversation format")
        
        #Return both original and converted datasets for evaluation and training
        return dataset, converted_dataset
    
    except Exception as e:
        #Log detailed error for dataset preparation issues
        logger.error(f"Error preparing dataset: {e}")
        raise


def evaluate_model(model, tokenizer, image, instruction, label="Evaluation"):
    logger.info(f"Running {label} evaluation...")
    
    #Set model to inference mode for efficient prediction
    FastVisionModel.for_inference(model)
    
    #Prepare input in multimodal conversation format
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": instruction}
        ]}
    ]
    
    #Apply chat template to format conversation for model input
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    
    #Tokenize input with image for multimodal processing
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    #Display evaluation label for comparison
    print(f"\n{label}:\n")
    #Set up text streamer for real-time output visualization
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    #Generate model prediction with temperature and minimum probability sampling
    _ = model.generate(
        **inputs, 
        streamer=text_streamer, 
        max_new_tokens=DEFAULT_CONFIG["max_new_tokens"],
        use_cache=True,
        temperature=1.5,  # Higher temperature encourages diversity
        min_p=0.1  # Minimum probability filtering for quality outputs
    )
    #Add visual separator after output
    print("\n")


def train_model(model, tokenizer, dataset, args):
    logger.info("Starting model fine-tuning...")
    
    #Switch model to training mode for gradient updates
    FastVisionModel.for_training(model)
    
    #Configure Supervised Fine-Tuning trainer with optimized parameters
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        #Use specialized collator for vision-language data
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=dataset,
        #Configure training parameters
        args=SFTConfig(
            #Batch size per device balances memory usage and training efficiency
            per_device_train_batch_size=args.batch_size,
            #Gradient accumulation compensates for small batch sizes
            gradient_accumulation_steps=DEFAULT_CONFIG["gradient_accumulation_steps"],
            #Warmup steps prevent unstable early updates
            warmup_steps=DEFAULT_CONFIG["warmup_steps"],
            #Maximum training steps limit training time
            max_steps=args.max_steps,
            #Learning rate controls update magnitude
            learning_rate=args.learning_rate,
            #FP16 precision reduces memory usage when BF16 not available
            fp16=not is_bf16_supported(),
            #BF16 precision offers better training stability when available
            bf16=is_bf16_supported(),
            #Logging interval for progress tracking
            logging_steps=1,
            #8-bit Adam optimizer reduces memory usage
            optim="adamw_8bit",
            #Weight decay prevents overfitting
            weight_decay=0.01,
            #Linear learning rate schedule for stable training
            lr_scheduler_type="linear",
            #Seed for reproducibility
            seed=args.seed,
            #Output directory for checkpoints
            output_dir=args.output_dir,
            #Disable reporting to save resources
            report_to="none",
            #Preserve all columns in dataset
            remove_unused_columns=False,
            #Empty text field since we're using custom formatting
            dataset_text_field="",
            #Skip automatic dataset preparation
            dataset_kwargs={"skip_prepare_dataset": True},
            #Parallel processing for dataset preparation
            dataset_num_proc=4,
            #Maximum sequence length constraints memory usage
            max_seq_length=DEFAULT_CONFIG["max_seq_length"],
        ),
    )
    
    #Log GPU information to monitor resource usage
    if torch.cuda.is_available():
        #Get GPU device properties
        gpu_stats = torch.cuda.get_device_properties(0)
        #Track initial GPU memory usage
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        #Calculate maximum available GPU memory
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        logger.info(f"{start_gpu_memory} GB of memory reserved at start.")
    
    #Execute training process
    trainer_stats = trainer.train()
    
    #Log comprehensive training statistics for analysis
    if torch.cuda.is_available():
        #Calculate peak memory usage during training
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        #Calculate memory usage specific to LoRA training
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        #Calculate percentage of total memory used
        used_percentage = round(used_memory / max_memory * 100, 3)
        #Calculate percentage of total memory used for LoRA
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
        
        #Log training duration
        logger.info(f"Training completed in {trainer_stats.metrics['train_runtime']:.2f} seconds "
                    f"({trainer_stats.metrics['train_runtime']/60:.2f} minutes).")
        #Log peak memory usage
        logger.info(f"Peak reserved memory = {used_memory} GB.")
        #Log memory usage specifically for LoRA training
        logger.info(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        #Log percentage of total memory used
        logger.info(f"Peak reserved memory % of max memory = {used_percentage}%.")
        #Log percentage of total memory used for LoRA
        logger.info(f"Peak reserved memory % of max memory for LoRA = {lora_percentage}%.")
    
    #Return training statistics for analysis
    return trainer_stats


def save_model(model, tokenizer, args):
    try:
        #Create directory for saving LoRA adapters
        os.makedirs("lora_model", exist_ok=True)
        
        #Save LoRA adapters locally for future use
        logger.info("Saving LoRA adapters to 'lora_model' directory...")
        model.save_pretrained("lora_model")
        tokenizer.save_pretrained("lora_model")
        
        #Optionally push to HuggingFace Hub if specified
        if not args.skip_push and args.hub_model_id:
            #Check for HuggingFace token
            if not args.hf_token:
                logger.warning("No HuggingFace token provided. Using environment variable HF_TOKEN if available.")
            
            #Use provided token or environment variable
            hf_token = args.hf_token or os.environ.get("HF_TOKEN")
            
            #Verify token is available
            if hf_token:
                #Upload merged model to HuggingFace Hub
                logger.info(f"Saving merged model to HuggingFace Hub as {args.hub_model_id}...")
                model.push_to_hub_merged(
                    args.hub_model_id,
                    tokenizer,
                    #Use 16-bit precision for merged model to balance size and quality
                    save_method="merged_16bit",
                    token=hf_token
                )
                logger.info("Model successfully pushed to HuggingFace Hub.")
            else:
                #Log error if token is missing
                logger.error("HuggingFace token not found. Model not pushed to Hub.")
                return False
        
        #Return success status
        return True
        
    except Exception as e:
        #Log detailed error for saving issues
        logger.error(f"Error saving model: {e}")
        return False


def main():
    #Parse command-line arguments for customization
    args = setup_argument_parser()
    
    #Set random seed for reproducible results
    set_seed(args.seed)
    
    #Install necessary dependencies
    if not install_requirements():
        logger.error("Failed to install requirements. Exiting.")
        sys.exit(1)
    
    #Initialize model and tokenizer with PEFT configuration
    model, tokenizer = setup_vision_language_model(args.model_name, args.lora_rank)
    
    #Define domain-specific instruction for dataset formatting
    instruction = """
    You are an expert radiographer. Describe accurately what you see
    in the provided images as detailed, concise, and professional as 
    possible.
    """
    
    #Load and prepare dataset for training
    raw_dataset, converted_dataset = load_and_prepare_dataset(args.dataset_name, instruction)
    
    #Run pre-training evaluation for baseline comparison
    if not args.skip_eval:
        evaluate_model(
            model, tokenizer, raw_dataset[0]["image"], instruction, "Before Training"
        )
    
    #Execute fine-tuning process
    train_stats = train_model(model, tokenizer, converted_dataset, args)
    
    #Run post-training evaluation to measure improvement
    if not args.skip_eval:
        evaluate_model(
            model, tokenizer, raw_dataset[0]["image"], instruction, "After Training"
        )
    
    #Save fine-tuned model locally and optionally to Hub
    save_model(model, tokenizer, args)
    
    #Log successful completion
    logger.info("Fine-tuning process completed successfully.")


if __name__ == "__main__":
    #Record starting time for runtime tracking
    start_time = datetime.now()
    
    try:
        #Run main function
        main()
    except KeyboardInterrupt:
        #Handle user interruption gracefully
        logger.info("Process interrupted by user.")
    except Exception as e:
        #Log unexpected errors
        logger.error(f"An error occurred: {e}")
        sys.exit(1)
    
    #Calculate and log total runtime for performance analysis
    elapsed_time = datetime.now() - start_time
    logger.info(f"Total runtime: {elapsed_time}")
