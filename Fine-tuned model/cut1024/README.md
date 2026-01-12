---
library_name: peft
license: other
base_model: Qwen/Qwen2.5-7B-Instruct
tags:
- base_model:adapter:Qwen/Qwen2.5-7B-Instruct
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: qwen2.5-7b_finance_0111_1753_e2p0_lr2e04_cut1024_loraall
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# qwen2.5-7b_finance_0111_1753_e2p0_lr2e04_cut1024_loraall

This model is a fine-tuned version of [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) on the finance_alpaca dataset.
It achieves the following results on the evaluation set:
- Loss: 1.6872

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0002
- train_batch_size: 2
- eval_batch_size: 1
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 16
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 2.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 1.8844        | 0.3583 | 200  | 1.7044          |
| 1.8388        | 0.7165 | 400  | 1.6872          |
| 1.8316        | 1.0734 | 600  | 1.6878          |
| 1.774         | 1.4317 | 800  | 1.7014          |
| 1.7813        | 1.7900 | 1000 | 1.6904          |


### Framework versions

- PEFT 0.17.1
- Transformers 4.57.1
- Pytorch 2.5.1+cu124
- Datasets 4.0.0
- Tokenizers 0.22.2