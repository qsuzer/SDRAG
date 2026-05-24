# SDRAG Fine-tuning Scripts

This directory contains the fine-tuning experiment utilities used for SDRAG.
The scripts are intentionally parameterized so the released code does not
contain machine-specific paths, usernames, or private credentials.

## Files

- `prepare_sdrag_training_data.py`: builds augmented SDRAG training data by generating synthetic pruning negatives and balancing decomposition, dependency, and pruning examples.
- `split_single_task_data.py`: splits a mixed ChatML-style jsonl file into decomposition, dependency, and pruning jsonl files.
- `run_multitask_lora.sh`: runs a small set of multitask LoRA configurations with ms-swift.
- `run_single_task_lora_sweep.sh`: splits the mixed data and runs task-specific LoRA sweeps.
- `resolve_swift_best_checkpoint.py`: extracts validation statistics and the best checkpoint from a Swift run directory.
- `merge_lora_checkpoint.py`: merges a PEFT LoRA adapter into the base model.

## Example

```bash
export MODEL_PATH=meta-llama/Meta-Llama-3.1-8B-Instruct
export DATASET_PATH=data/aligned_train_data_balanced_1to1to1.jsonl
export SWIFT_BIN=swift

bash scripts/finetune/run_multitask_lora.sh runs/sdrag_multitask 0
```

For single-task sweeps:

```bash
export MODEL_PATH=meta-llama/Meta-Llama-3.1-8B-Instruct
bash scripts/finetune/run_single_task_lora_sweep.sh runs/sdrag_single_task 0 data/aligned_train_data_balanced_1to1to1.jsonl
```

Synthetic data generation uses `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL`.
