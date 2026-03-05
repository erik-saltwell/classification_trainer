# classification_trainer Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-03-05

## Active Technologies
- Python 3.11+ + wandb (sweep + agent API), unsloth `FastLanguageModel`, TRL `SFTTrainer`, Pydantic v2, Typer, HuggingFace `datasets` (001-wandb-sweep)
- YAML config files (`inference_info/`, `training_info/`); per-trial checkpoint directories under `outputs/<model_name>/<run_id>/` (001-wandb-sweep)

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
- 001-wandb-sweep: Added Python 3.11+ + wandb (sweep + agent API), unsloth `FastLanguageModel`, TRL `SFTTrainer`, Pydantic v2, Typer, HuggingFace `datasets`

- 001-model-save-publish: Added Python 3.11+ + Unsloth (GGUF/LoRA/merged save), `huggingface_hub` (upload + ModelCard),

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
