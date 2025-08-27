# GPT2_With_LoRA

## Description
This project demonstrates how to fine-tune the GPT-2 language model using two approaches: full model fine-tuning and parameter-efficient fine-tuning with LoRA (Low-Rank Adaptation). The notebook uses the Reddit-TIFU dataset for training, incorporates memory optimization strategies like gradient accumulation, mixed precision, and gradient checkpointing to handle resource constraints, and includes text generation examples to compare results between the base model, fully fine-tuned model, and LoRA-adapted model.

## What is LoRA?
- LoRA, or Low-Rank Adaptation, is a technique used to fine-tune large language models (LLMs) efficiently by adding small, trainable matrices to the model while keeping the original weights unchanged.
- This approach significantly reduces the number of parameters that need to be trained, making it more resource-efficient and faster compared to traditional fine-tuning methods.

# Key features:
- Loads and preprocesses the Reddit-TIFU dataset.
- Implements full fine-tuning of GPT-2.
- Applies LoRA for efficient adaptation with reduced trainable parameters.
- Monitors GPU memory usage during training.
- Generates sample text outputs for evaluation.
- Saves models and LoRA adapters for reuse.
**This is ideal for learning about efficient fine-tuning techniques in NLP, especially on limited hardware.**

# Requirements
- Python 3.8+
- PyTorch (with CUDA support for GPU acceleration)
- Transformers library
- PEFT (Parameter-Efficient Fine-Tuning)
- Datasets library
- Accelerate
- Matplotlib and TQDM for visualization and progress tracking

## Install dependencies via:
``bash
pip install transformers peft datasets accelerate matplotlib tqdm
``
