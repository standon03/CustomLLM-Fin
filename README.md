# CustomLLM-Fin

**CustomLLM-Fin** is a two-part project exploring the design, training, and fine-tuning of financial language models using instruction-based data. It focuses on the [`wealth-alpaca_lora`](https://huggingface.co/datasets/gbharti/wealth-alpaca_lora) dataset, which consists of prompt-response pairs related to financial literacy and investing.

This repository documents:

- A failed attempt to train a transformer-based language model from scratch using PyTorch (unsuccessful but educational)
- A successful fine-tuning pipeline using GPT-2 to generate accurate and coherent financial responses

This project reflects a full machine learning development cycle: data preprocessing, model experimentation, training, evaluation, and interpretation.

---

## Project Goals

- Understand the limitations of building a custom transformer model on small, domain-specific data
- Apply transfer learning to fine-tune a pre-trained GPT model for improved performance
- Evaluate performance through prompt-response outputs and standard metrics
- Learn from failure, iterate, and improve with a more effective model design

---

## Notebook Overviews

### `models/Custom_FinLLM.ipynb`

This notebook implements a transformer language model from scratch in PyTorch. It includes:

- Tokenization using SentencePiece
- Manual implementation of:
  - Positional encoding
  - Multi-head self-attention
  - Transformer encoder-decoder blocks
- Custom loss functions and sampling strategy

**Outcome**: This approach failed to generate coherent outputs, highlighting the difficulty of training from scratch on a small dataset. The notebook is preserved as a transparent record of experimentation, showcasing understanding of deep learning architecture and model limitations.

---

### `models/Fin_GPT_2_Fine_Tuning_CLEAN.ipynb`

This notebook builds on the lessons from the first attempt and successfully fine-tunes a pre-trained GPT-2 model using Hugging Face Transformers. It includes:

- Loading and preprocessing the financial dataset
- Tokenization aligned with GPT-2's vocabulary
- Fine-tuning setup using Hugging Face’s `Trainer` API
- Training and validation loss tracking
- Inference pipeline for generating text completions

**Outcome**: The fine-tuned model produces high-quality financial text completions. It outperforms the scratch-built model significantly in both quantitative metrics and human-like output quality.

---

## Dataset Details

- **Source**: [`gbharti/wealth-alpaca_lora`](https://huggingface.co/datasets/gbharti/wealth-alpaca_lora)
- **Format**: Prompt–response instruction pairs for financial Q&A
- **Preprocessing**: Encoded using SentencePiece tokenizer, stored in `Dataset.json.zip`
