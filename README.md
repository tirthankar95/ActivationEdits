# ActivationEdits

ActivationEdits is an experimental project for safety alignment using activation-level methods and lightweight adaptation.

The repository has two complementary tracks:

1. Activation probing and steering direction discovery from secure vs insecure code examples.
2. LoRA-based fine-tuning and adapter-driven inference for safer code generation behavior.

## Motivation

Large language models often encode safety-relevant concepts in intermediate hidden representations. If secure and insecure coding patterns are separable in activation space, those directions can be used to analyze, and eventually steer, model behavior.

This project focuses on:

1. Measuring secure/insecure separability with linear probes.
2. Visualizing activation geometry across transformer layers.
3. Training low-rank adapters on curated secure/insecure code-style prompts.

## Repository Layout

- [activation_edit.py](activation_edit.py): activation-space probing experiment (probe accuracy, PCA plots, steering vector).
- [config/reduced_error.yaml](config/reduced_error.yaml): config for activation probing examples and output paths.
- [train_low_rank/train.py](train_low_rank/train.py): Hydra training/inference entrypoint for LoRA workflow.
- [train_low_rank/prepare_data.py](train_low_rank/prepare_data.py): dataset preparation and dataset-disk sharding/loading helpers.
- [config/train_low_rank.yaml](config/train_low_rank.yaml): LoRA training and adapter inference config.
- [data/CVEFixes](data/CVEFixes): CSV resources for secure/insecure code data.
- [act_output](act_output): generated artifacts (plots and analysis outputs).

## Environment Setup

This project uses Python and uv.

1. Install dependencies:

```bash
uv sync
```

2. If you only want to add a missing package:

```bash
uv add <package-name>
```

## Activation Probing Workflow

The activation experiment script reads [config/reduced_error.yaml](config/reduced_error.yaml), loads the configured model, extracts hidden states, then trains a layer-wise linear probe.

Run:

```bash
uv run python activation_edit.py
```

Outputs are written under the configured output directory (for example, in act_output/reduced_error):

1. Probe accuracy plot by layer.
2. Activation separation visualization.
3. Console diagnostics for steering vector geometry.

## LoRA Training and Adapter Inference

The LoRA pipeline is managed through Hydra config in [config/train_low_rank.yaml](config/train_low_rank.yaml).

### 1) Prepare/Train LoRA

Set training mode in config:

- train: true

Then run:

```bash
uv run python train_low_rank/train.py
```

Artifacts are written to the configured output directory, typically:

- lora_output/adapter

### 2) Inference with Adapter (no merged model required)

Set in [config/train_low_rank.yaml](config/train_low_rank.yaml):

1. train: false
2. inference.use_adapter: true
3. inference.adapter_dir: path to saved adapter (for example, ./lora_output/adapter)

Then run:

```bash
uv run python train_low_rank/train.py
```

In this mode, vLLM loads the base model and applies the LoRA adapter at generation time via LoRA request.

## Data Notes

- Input CSVs for LoRA data preparation are read from data/CVEFixes/sample.
- Prepared datasets are persisted under data/train by the dataset preparation utility.
- Language labels are normalized (for example, Other to any) during preprocessing.

## Configuration Guide

### Activation probe config

In [config/reduced_error.yaml](config/reduced_error.yaml):

1. model.name: Hugging Face model id.
2. data.secure_examples and data.insecure_examples: in-memory examples for probing.
3. output.dir and plot names: where visualizations are saved.

### LoRA config

In [config/train_low_rank.yaml](config/train_low_rank.yaml):

1. train: toggle between training and inference.
2. model.name: base model path/id.
3. output.dir: adapter artifact root.
4. training.num_train_epochs and training.per_device_train_batch_size.
5. inference.use_adapter and inference.adapter_dir.

## Current Status

This is an active experimental codebase. Interfaces and defaults may evolve while iterating on safety alignment experiments.

## License

No license file is currently included in this repository.
