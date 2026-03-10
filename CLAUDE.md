# classification_trainer Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-03-05

## Active Technologies
- Python 3.11+ + wandb (sweep + agent API), unsloth `FastLanguageModel`, TRL `SFTTrainer`, Pydantic v2, Typer, HuggingFace `datasets` (001-wandb-sweep)
- YAML config files (`inference_info/`, `training_info/`); per-trial checkpoint directories under `outputs/<model_name>/<run_id>/` (001-wandb-sweep)
- Python 3.11+ + Pydantic v2, Typer, PyYAML, Unsloth, TRL, HuggingFace datasets (001-reorganize-config)
- YAML config files in `training_info/`, `dataset_info/`, `base_model_info/`, (001-reorganize-config)
- Python 3.11+ + Pydantic v2, wandb (sweep API), Typer, PyYAML, Unsloth, TRL (004-sweep-config)
- YAML config files (`training_info/`) (004-sweep-config)
- Python 3.11+ + Pydantic v2, PyYAML, Typer (005-configurable-sequence-lengths)
- YAML config files (`dataset_info/`) (005-configurable-sequence-lengths)
- YAML config files (`training_info/`, `dataset_info/`) (006-dataset-ref-training-info)
- Python 3.11+ + Pydantic v2 (not directly used), standard library only for this feature (007-file-logger)
- Plain text file output (007-file-logger)
- Python 3.11+ + Unsloth `FastLanguageModel`, TRL `SFTTrainer`, HuggingFace `datasets` + `transformers`, Pydantic v2 (009-pretokenize-runner)
- HuggingFace Dataset (in-memory, Arrow-backed) (009-pretokenize-runner)
- Python 3.11+ + Pydantic v2, wandb sweep/agent API, TRL SFTTrainer, Typer/Rich (010-sweep-params-observability)
- N/A (config files only — YAML in `training_info/`) (010-sweep-params-observability)

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
- 010-sweep-params-observability: Added Python 3.11+ + Pydantic v2, wandb sweep/agent API, TRL SFTTrainer, Typer/Rich
- 009-pretokenize-runner: Added Python 3.11+ + Unsloth `FastLanguageModel`, TRL `SFTTrainer`, HuggingFace `datasets` + `transformers`, Pydantic v2
- 009-pretokenize-runner: Added [if applicable, e.g., PostgreSQL, CoreData, files or N/A]


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
