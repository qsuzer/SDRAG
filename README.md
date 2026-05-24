# SDRAG: Sub-question Dependency Reasoning for RAG

This repository contains the reproduction code for the SDRAG project and several baseline Retrieval-Augmented Generation (RAG) pipelines. SDRAG models multi-hop reasoning as a graph of sub-questions, analyzes dependencies among them, prunes unnecessary leaf nodes, and then executes the remaining reasoning path.

![Model Framework](framework.pdf)

## Features

* **Graph-based reasoning**: Models the reasoning process as a directed graph of sub-questions.
* **Dependency analysis**: Builds explicit dependency edges between sub-questions.
* **Dynamic pruning**: Removes irrelevant sub-questions to reduce noise.
* **Multiple baselines**: Includes standard RAG, IRCoT, GenGround, PERQA, and DualRAG.
* **Fine-tuning utilities**: Includes SDRAG data preparation and LoRA experiment scripts.

## Installation

1. Clone this repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

**Note**: You need to have `FlashRAG` installed.

- [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG)

## Project Structure

```text
SDRAG/
|-- src/
|   |-- pipeline/
|   |   |-- base_pipeline.py
|   |   |-- subgraph_pipeline.py
|   |   |-- ircot_pipeline.py
|   |   |-- genground_pipeline.py
|   |   |-- perqa_pipeline.py
|   |   |-- dualrag_pipeline.py
|   |   |-- rag_pipeline.py
|   |   |-- direct_pipeline.py
|   |   `-- prompts.py
|   `-- data/
|-- config/
|   `-- basic_config.yaml
|-- scripts/
|   |-- run_inference.py
|   `-- finetune/
|-- run_pipeline.py
`-- requirements.txt
```

## Supported Pipelines

This repository supports the following pipelines:

* **`subgraph`**: SDRAG graph-based reasoning.
* **`ircot`**: Interleaving Retrieval with Chain-of-Thought.
* **`genground`**: Generate-and-Ground.
* **`perqa`**: Planner-Executor-Reasoner architecture.
* **`dualrag`**: Dual-view RAG.
* **`rag`**: Standard retrieve-then-generate.
* **`direct`**: Direct LLM generation.

## Usage

Run any supported pipeline with `run_pipeline.py`.

```bash
python run_pipeline.py --pipeline <pipeline_name> --dataset <dataset_name>
```

Run SDRAG with the default config:

```bash
python run_pipeline.py --pipeline subgraph --config config/basic_config.yaml
```

Run IRCoT on HotpotQA:

```bash
python run_pipeline.py --pipeline ircot --dataset hotpotqa --split dev
```

Run standard RAG:

```bash
python run_pipeline.py --pipeline rag --dataset hotpotqa
```

## Configuration

Configuration is managed via YAML files such as `config/basic_config.yaml`. You can override common settings through command-line arguments:

* `--dataset`: Override the dataset name.
* `--split`: Override the dataset split.
* `--gpu_id`: Specify GPU IDs.
* `--test_sample_num`: Limit the number of samples.

## Fine-tuning

Fine-tuning scripts are under `scripts/finetune/`. They include:

* `prepare_sdrag_training_data.py`: Build augmented and balanced SDRAG training data.
* `split_single_task_data.py`: Split mixed data into decomposition, dependency, and pruning subsets.
* `run_multitask_lora.sh`: Run multitask LoRA experiments with ms-swift.
* `run_single_task_lora_sweep.sh`: Run task-specific LoRA sweeps.
* `resolve_swift_best_checkpoint.py`: Parse Swift logs and locate the best checkpoint.
* `merge_lora_checkpoint.py`: Merge a PEFT LoRA adapter into a base model.

The scripts use environment variables and command-line arguments for local paths and credentials, so no private machine paths are required in the released code.

The seed data for fine-tuning is derived from the **PER-PSE** dataset: [https://huggingface.co/datasets/GenIRAG/PER-PSE](https://huggingface.co/datasets/GenIRAG/PER-PSE)
