title: CustomLLM-Fin

description: >
  CustomLLM-Fin is a two-part project exploring the design, training, and fine-tuning of financial language models using instruction-based data.
  It focuses on the wealth-alpaca_lora dataset, consisting of prompt-response pairs related to financial literacy and investing.

objectives:
  - Understand the limitations of building a custom transformer model on small, domain-specific data
  - Apply transfer learning to fine-tune a pre-trained GPT model
  - Evaluate output quality using perplexity, accuracy, and realistic prompt-response completions
  - Learn from failure, iterate, and improve with a more effective model design

repository_structure:
  root: CustomLLM-Fin/
  contents:
    - models/
    - models/Custom_FinLLM.ipynb: Attempt to build a transformer from scratch
    - models/Fin_GPT_2_Fine_Tuning_CLEAN.ipynb: GPT-2 fine-tuning and evaluation pipeline
    - Dataset.json.zip: Tokenized instruction dataset
    - README.md: Documentation

notebooks:
  - name: Custom_FinLLM.ipynb
    path: models/Custom_FinLLM.ipynb
    description: >
      Implements a transformer model from scratch using PyTorch with positional encoding,
      attention, feedforward layers, and custom sampling. It failed due to convergence issues
      on limited data, but remains as an educational experiment.
    outcome: Failed to generate coherent outputs. Included as a learning reference.

  - name: Fin_GPT_2_Fine_Tuning_CLEAN.ipynb
    path: models/Fin_GPT_2_Fine_Tuning_CLEAN.ipynb
    description: >
      Successfully fine-tunes a GPT-2 model using Hugging Face Transformers. Includes
      tokenization, Trainer API setup, and an inference pipeline for realistic financial text generation.
    outcome: Produced high-quality outputs with strong evaluation results.

dataset:
  name: wealth-alpaca_lora
  source: https://huggingface.co/datasets/gbharti/wealth-alpaca_lora
  format: Prompt-response instruction pairs
  domain: Finance, investing, financial literacy
  preprocessing: Tokenized using SentencePiece and saved in JSON format

results:
  Test Loss: 0.5625
  Test Accuracy: 93.57%
  Perplexity: 1.76

sample_outputs:
  - prompt: How should I start investing with $10,000?
    output: Start with low-cost index funds or ETFs, maintain diversification, and invest for the long term.
  - prompt: What are the tax implications of selling stocks?
    output: Capital gains tax applies. Short-term gains are taxed as ordinary income; long-term gains at lower rates.

setup_instructions:
  dependencies:
    - torch
    - sentencepiece
    - transformers
    - matplotlib
  steps:
    - Unzip Dataset.json.zip
    - Open notebook of choice
    - Run all cells

skills_demonstrated:
  - Manual implementation of deep learning architectures
  - Experimentation with transformer models
  - Application of Hugging Face’s fine-tuning API
  - Evaluation and text generation with domain-specific data
  - Interpreting model limitations and improving through iteration

future_improvements:
  - Apply parameter-efficient tuning methods such as LoRA or QLoRA
  - Expand dataset to include more financial domains
  - Use metrics like BLEU or ROUGE for deeper output comparison

license: MIT

author:
  name: Shubh Tandon
  github: https://github.com/standon03
  contact: Open to collaboration and feedback
