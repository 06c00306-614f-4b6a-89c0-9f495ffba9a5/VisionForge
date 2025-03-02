<p>
  <img src="https://github.com/user-attachments/assets/99ad5c43-fd55-429b-9e6e-dc27994a39f0" alt="VisionForge Logo" width="400">
  <br>
  <em>Forge powerful vision-language models for specialized domains</em>
</p>

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/docs/transformers/index)

## 🌟 Overview

VisionForge is a collection of scripts for fine-tuning vision-language models (VLMs) on domain-specific image datasets. Transform general-purpose models into specialized experts that understand your visual domain with impressive accuracy and minimal computational resources. **Perfect for:** healthcare professionals, scientific researchers, industrial inspectors, specialized content creators, and any domain expert seeking to harness AI vision capabilities for their specific field.

<br>

## ✨ Key Features

- **Memory-Efficient Fine-tuning**: Train 11B+ parameter models on a single consumer GPU using 4-bit quantization
- **Comprehensive Adaptation**: Fine-tune vision, language, and cross-attention components simultaneously
- **Domain Specialization**: Adapt pre-trained models to specialized domains (medical imaging, industrial inspection, etc.)
- **One-Command Deployment**: Push fine-tuned models directly to Hugging Face Hub
- **Built-in Evaluation**: Compare before/after performance with integrated evaluation tools
- **Flexible Configuration**: Customize all aspects of training via command line or config files
- **Hardware-Aware Optimizations**: Automatic detection of hardware capabilities (BF16 support, etc.)

<br>

## 📋 Requirements

- Python 3.9+
- CUDA-compatible GPU with 16GB+ VRAM (24GB+ recommended for larger models)
- CUDA 11.7+ and cuDNN 8.5+

<br>

## 🚀 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/visionforge.git
cd visionforge

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🏁 Quickstart

```bash
# Fine-tune a model on a sample medical dataset
python visionforge.py --model_name "unsloth/Llama-3.2-11B-Vision-Instruct" \
                     --dataset_name "unsloth/Radiology_mini" \
                     --output_dir "output/my_radiology_model" \
                     --max_steps 100
```

<br>

## 🎯 Use Cases

### Medical Imaging Analysis
```bash
python visionforge.py --model_name "unsloth/Llama-3.2-11B-Vision-Instruct" \
                     --dataset_name "your-medical-dataset" \
                     --hub_model_id "your-username/medical-llm-expert" \
                     --instruction "You are a medical imaging expert. Describe what you see in this image in professional medical terminology."
```

<br>

### Industrial Inspection
```bash
python visionforge.py --model_name "unsloth/Llama-3.2-11B-Vision-Instruct" \
                     --dataset_name "your-industrial-dataset" \
                     --lora_rank 32 \
                     --batch_size 4 \
                     --instruction "Identify any defects or anomalies in this industrial component."
```

<br>

## ⚙️ Advanced Configuration

VisionForge offers extensive customization options:

```bash
python visionforge.py --help
```

<br>

Key parameters:
- `--model_name`: Base model to fine-tune
- `--dataset_name`: Dataset name on HuggingFace Hub or local path
- `--lora_rank`: Rank for LoRA adaptation (higher = more capacity)
- `--batch_size`: Training batch size
- `--learning_rate`: Learning rate for training
- `--max_steps`: Maximum number of training steps
- `--hub_model_id`: Model ID for HuggingFace Hub (username/model_name)
