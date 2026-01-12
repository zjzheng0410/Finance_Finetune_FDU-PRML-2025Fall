# Financial Domain LLM Fine-tuning: Ablation Study

This repository contains supplementary materials for a fine-tuning experiment on financial domain language models, including data processing scripts, training code, evaluation results, and fine-tuned model checkpoints from various ablation studies.

## Project Overview

This project focuses on fine-tuning the Qwen2.5-7B-Instruct model for financial domain tasks using LoRA (Low-Rank Adaptation). The repository includes comprehensive ablation studies examining the effects of different hyperparameters and configurations, including data cleaning, sequence length cutoff, training epochs, learning rates, and LoRA target modules.

## Project Structure

```
.
├── data/                          # Data files and processing scripts
│   ├── finance_alpaca.json        # Original finance dataset in Alpaca format
│   ├── final_dataset_clean.json   # Cleaned dataset
│   ├── final_dataset_clean_no_test.json  # Cleaned dataset without test set
│   ├── Finance_test.json          # Finance domain test set
│   ├── general_test.json          # General domain test set
│   ├── val.json                   # Validation set
│   ├── cleantest.py               # Script for cleaning test data
│   ├── move_test_duplicate.py     # Script for removing test duplicates
│   └── splitdata.py               # Script for data splitting
│
├── core_code/                     # Core training and evaluation code
│   ├── run_finetune.py            # Main fine-tuning script
│   └── eval.py                    # Model evaluation script
│
├── eval_result/                   # Evaluation results from ablation studies
│   ├── summary.csv                # Summary of all experiments
│   ├── ablation_overview_v2.png   # Visualization of ablation results
│   ├── ablation_overview_v2.pdf   # PDF version of visualization
│   ├── baseline_qwen/             # Baseline Qwen2.5-7B-Instruct results
│   ├── clean_data/                # Results: clean vs unclean data
│   ├── cutoff 1024/               # Results: sequence length ablation
│   ├── epoch/                     # Results: training epochs ablation
│   ├── learning_rate/             # Results: learning rate ablation
│   └── lora/                      # Results: LoRA target modules ablation
│
└── Fine-tuned model/              # Fine-tuned model checkpoints
    ├── attention/                 # LoRA on attention layers only
    ├── cleandata/                 # Model trained on cleaned data
    ├── cut1024/                   # Model with cutoff_len=1024
    ├── e2baseline/                # Baseline: 2 epochs
    ├── ep3/                       # Model trained for 3 epochs
    ├── ep4/                       # Model trained for 4 epochs
    ├── lr1e/                      # Model with learning_rate=1e-4
    ├── lr3e/                      # Model with learning_rate=3e-4
    └── MLP/                       # LoRA on MLP layers only
```

## Directory Descriptions

### `data/`

Contains raw and processed datasets along with data preprocessing scripts.

- **Raw Data**: `finance_alpaca.json` - Original financial domain dataset in Alpaca format
- **Processed Data**:
  - `final_dataset_clean.json` - Cleaned training dataset
  - `final_dataset_clean_no_test.json` - Cleaned dataset excluding test samples
  - `Finance_test.json` - Test set for finance domain evaluation
  - `general_test.json` - Test set for general domain evaluation
  - `val.json` - Validation set
- **Processing Scripts**:
  - `cleantest.py` - Cleans test data
  - `move_test_duplicate.py` - Removes duplicate test samples
  - `splitdata.py` - Splits data into train/val/test sets

### `core_code/`

Contains the core implementation files for model training and evaluation.

- **`run_finetune.py`**: Main fine-tuning script using LlamaFactory framework

  - Supports LoRA fine-tuning with configurable hyperparameters
  - Uses 4-bit quantization for memory efficiency
  - Configurable training parameters (epochs, learning rate, batch size, etc.)
- **`eval.py`**: Comprehensive evaluation script

  - Evaluates models on finance and general domain test sets
  - Computes multiple metrics: ROUGE-L, BLEU, chrF, Exact Match
  - Supports evaluation of both baseline and fine-tuned models
  - Generates comparison plots and summary statistics

### `eval_result/`

Stores evaluation results from all ablation experiments.

Each subdirectory contains:

- **`models/`**: Per-model evaluation results (metrics.json, predictions in JSONL format)
- **`plots/`**: Comparison visualizations (compare_finance.png, compare_general.png)
- **`summary.csv`** and **`summary.json`**: Aggregated results
- **`delta_vs_baseline.json`**: Performance deltas compared to baseline
- **`run_config.json`**: Configuration used for the evaluation run

**Ablation Studies**:

- **`baseline_qwen/`**: Baseline Qwen2.5-7B-Instruct performance
- **`clean_data/`**: Effect of data cleaning
- **`cutoff 1024/`**: Effect of sequence length (cutoff_len=1024 vs 2048)
- **`epoch/`**: Effect of training epochs (2, 3, 4 epochs)
- **`learning_rate/`**: Effect of learning rate (1e-4, 2e-4, 3e-4)
- **`lora/`**: Effect of LoRA target modules (all, attention-only, MLP-only)

### `Fine-tuned model/`

Contains fine-tuned model checkpoints from different ablation experiments.

Each model directory includes:

- **Adapter files**: `adapter_config.json`, `adapter_model.safetensors` (LoRA adapters)
- **Tokenizer files**: `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, etc.
- **Training artifacts**:
  - `train_results.json`, `eval_results.json` - Training and evaluation metrics
  - `trainer_log.jsonl` - Training logs
  - `training_loss.png`, `training_eval_loss.png` - Loss curves
  - `run_args.json` - Training configuration
- **Checkpoints**: `checkpoint-*/` directories at different training steps

**Model Variants**:

- **`e2baseline/`**: Baseline configuration (2 epochs, lr=2e-4, LoRA=all)
- **`cleandata/`**: Trained on cleaned dataset
- **`cut1024/`**: Trained with cutoff_len=1024
- **`ep3/`**, **`ep4/`**: Trained for 3 and 4 epochs respectively
- **`lr1e/`**, **`lr3e/`**: Trained with learning rates 1e-4 and 3e-4
- **`attention/`**: LoRA applied only to attention layers
- **`MLP/`**: LoRA applied only to MLP layers

## Key Findings

Based on the evaluation results in `eval_result/summary.csv`:

1. **Data Cleaning**: Minimal impact on performance metrics
2. **Sequence Length**: Cutoff=1024 shows slight improvements in general domain
3. **Training Epochs**: 3-4 epochs show marginal improvements over 2 epochs
4. **Learning Rate**: 1e-4 shows better chrF scores on finance domain
5. **LoRA Targets**: Attention-only and MLP-only configurations show different trade-offs in latency and performance

## Usage

### Training a Model

```bash
cd core_code
python run_finetune.py
```

Modify the configuration in `run_finetune.py` to adjust hyperparameters.

### Evaluating Models

```bash
cd core_code
python eval.py --model_path <path_to_model> --test_files <test_file_paths>
```

See `eval.py` for detailed command-line arguments and options.

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- LlamaFactory
- PEFT (for LoRA)
- rouge-score
- sacrebleu
- Other dependencies as specified in the code

## Notes

- All models use 4-bit quantization for memory efficiency
- Models are fine-tuned using LoRA (Low-Rank Adaptation) technique
- Evaluation is performed on both finance-specific and general domain test sets
- Checkpoint directories contain intermediate training states for analysis
