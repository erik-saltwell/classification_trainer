# classification_trainer Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-03-05

## Active Technologies
- Python 3.11+ + wandb (sweep + agent API), unsloth `FastLanguageModel`, TRL `SFTTrainer`, Pydantic v2, Typer, HuggingFace `datasets` (001-wandb-sweep)
- YAML config files (`inference_info/`, `training_info/`); per-trial checkpoint directories under `outputs/<model_name>/<run_id>/` (001-wandb-sweep)
- Python 3.11+ + Pydantic v2, Typer, PyYAML, Unsloth, TRL, HuggingFace datasets (001-reorganize-config)
- YAML config files in `training_info/`, `dataset_info/`, `base_model_info/`, (001-reorganize-config)

- Python 3.11+ + Unsloth (GGUF/LoRA/merged save), `huggingface_hub` (upload + ModelCard), (001-model-save-publish)

## Project Structure

```text
src/
tests/
```

## Commands

cd src [ONLY COMMANDS FOR ACTIVE TECHNOLOGIES][ONLY COMMANDS FOR ACTIVE TECHNOLOGIES] pytest [ONLY COMMANDS FOR ACTIVE TECHNOLOGIES][ONLY COMMANDS FOR ACTIVE TECHNOLOGIES] ruff check .

## Code Style

Python 3.11+: Follow standard conventions

## Recent Changes
- 001-reorganize-config: Added Python 3.11+ + Pydantic v2, Typer, PyYAML, Unsloth, TRL, HuggingFace datasets
- 001-wandb-sweep: Added Python 3.11+ + wandb (sweep + agent API), unsloth `FastLanguageModel`, TRL `SFTTrainer`, Pydantic v2, Typer, HuggingFace `datasets`

- 001-model-save-publish: Added Python 3.11+ + Unsloth (GGUF/LoRA/merged save), `huggingface_hub` (upload + ModelCard),

<!-- MANUAL ADDITIONS START -->
## Config Architecture (001-reorganize-config)

`training_info` is now the central config bundle. It references reusable configs by filename stem:
- `base_model: <name>` → loads `base_model_info/<name>.yaml`
- `inference: <name>` → loads `inference_info/<name>.yaml`
- `publishing: <name>` → loads `publishing_info/<name>.yaml` (or `null` to skip)
- `model_card_description` → moved here from `publishing_info`

CLI commands require only `--dataset` and `--training-info`. All other configs resolved from training_info.
All config filenames use kebab-case. Each config directory has a fully-commented `example.yaml`.
<!-- MANUAL ADDITIONS END -->
