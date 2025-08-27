# GPT2_With_LoRA

## Description
This project demonstrates how to fine-tune the GPT-2 language model using two approaches: full model fine-tuning and parameter-efficient fine-tuning with LoRA (Low-Rank Adaptation). The notebook uses the Reddit-TIFU dataset for training, incorporates memory optimization strategies like gradient accumulation, mixed precision, and gradient checkpointing to handle resource constraints, and includes text generation examples to compare results between the base model, fully fine-tuned model, and LoRA-adapted model.

## What is LoRA?
- LoRA, or Low-Rank Adaptation, is a technique used to fine-tune large language models (LLMs) efficiently by adding small, trainable matrices to the model while keeping the original weights unchanged.
- This approach significantly reduces the number of parameters that need to be trained, making it more resource-efficient and faster compared to traditional fine-tuning methods.

## Key features:
- Loads and preprocesses the Reddit-TIFU dataset.
- Implements full fine-tuning of GPT-2.
- Applies LoRA for efficient adaptation with reduced trainable parameters.
- Monitors GPU memory usage during training.
- Generates sample text outputs for evaluation.
- Saves models and LoRA adapters for reuse.
**This is ideal for learning about efficient fine-tuning techniques in NLP, especially on limited hardware.**

## Requirements
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

## Usage
1. Open the gpt2_lora_pytorch.ipynb notebook in Jupyter or Google Colab.
2. Ensure you have a GPU-enabled environment for optimal performance (e.g., Colab with T4 GPU).
3. Run the cells sequentially:
   - Install packages.
   - Set up hyperparameters and environment.
   - Load and preprocess the dataset.
   - Train the full GPT-2 model.
   - Train the LoRA-adapted model.
   - Generate text samples.
4. Adjust hyperparameters like BATCH_SIZE, EPOCHS, or LORA_RANK as needed.
5. Saved models are stored in ./my-fine-tuned-gpt2 (full model) and ./gpt2-lora-reddit (LoRA adapters).

## Example text generation:
- Base GPT-2: Basic completions.
- Full Fine-Tuned: Dataset-specific responses.
- LoRA: Efficient, low-memory fine-tuning with similar quality.

## Examples:
- Input Prompt: "I like basketball"
  - Base GPT-2 Output: Short, generic continuation.
  - Fine-Tuned Output: More context-aware response influenced by Reddit-TIFU stories.
  - LoRA Output: Efficiently adapted response with reduced training overhead.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- Hugging Face Transformers and PEFT libraries.
- Reddit-TIFU dataset by Fredithefish.
