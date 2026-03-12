# Implementation Plan: Model Save and Publish

**Branch**: `001-model-save-publish` | **Date**: 2026-03-05 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/001-model-save-publish/spec.md`

## Summary

After training completes, the best-performing model checkpoint is saved to disk in
configurable formats (GGUF, LoRA adapter, merged HF checkpoint) alongside a generated
model card. A new `publish` CLI command uploads saved artifacts to HuggingFace Hub,
one repository per format, named `<username>/<model-name>-<format>`. All behavior is
driven by a new `PublishingInfo` YAML config model. Saving uses Unsloth's first-class
save APIs for GGUF/LoRA/merged.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: Unsloth (GGUF/LoRA/merged save), `huggingface_hub` (upload + ModelCard),
Pydantic v2 (config model), Typer + Rich (CLI)
**Storage**: Local filesystem (`outputs/`), HuggingFace Hub (remote)
**Testing**: pytest (existing project convention)
**Target Platform**: Linux (GPU server) or any machine with sufficient VRAM
**Project Type**: CLI tool
**Performance Goals**: Save operations are batch/offline; no latency target.
**Constraints**: All saves occur after training completes (not during).
**Scale/Scope**: Single model per training run; one repository per format on HuggingFace.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Configuration-First | ✅ PASS | `PublishingInfo` is a Pydantic model loaded from `publishing_info/<name>.yaml`. No hardcoded formats or save paths. |
| II. Protocol-Based Interfaces | ✅ PASS | `PublishCommand` accepts `LoggingProtocol`. `publishing_helper` functions accept `LoggingProtocol`. No concrete logger types in helpers. |
| III. Separation of Concerns | ✅ PASS | `publishing_helper.py` owns all save/publish/card-gen logic. `PublishCommand` and `TrainCommand` orchestrate only. `PublishingInfo` in `configuration/`. |
| IV. Observability | ✅ PASS | Pre/post metrics flow into model card. All log messages via `LoggingProtocol`. Progress indicators for each format during upload. |
| V. Simplicity & Scope | ✅ PASS | Saving/publishing fine-tuned classification models is in direct scope. No premature abstractions. |

*Post-Phase-1 re-check*: All gates pass. No complexity violations.

## Project Structure

### Documentation (this feature)

```text
specs/001-model-save-publish/
├── plan.md              # This file
├── research.md          # Phase 0 — save APIs, model card, config schema
├── data-model.md        # Phase 1 — entities, file layout, state transitions
├── quickstart.md        # Phase 1 — end-to-end usage guide
├── contracts/
│   └── cli-contracts.md # Phase 1 — CLI schemas, YAML schema, naming contract
└── tasks.md             # Phase 2 output (/speckit.tasks — not yet created)
```

### Source Code (repository root)

```text
src/classification_trainer/
├── configuration/
│   └── publishing_info.py          # NEW: SaveFormat enum, PublishingInfo, load_publishing_info
├── helpers/
│   └── publishing_helper.py        # EXTEND: save_model(), generate_model_card(), publish_model()
├── commands/
│   └── publish.py                  # NEW: PublishCommand
├── console/
│   └── main.py                     # MODIFY: add `publish` command; add --publishing-info to train
└── utils/
    └── common_paths.py             # MODIFY: add publishing_info + outputs paths

publishing_info/                    # NEW directory (config YAMLs)
outputs/                      # Created at runtime by publishing_helper
```

**Structure Decision**: Single-project layout. All changes follow the existing three-layer
architecture (configuration / helpers / commands). No new top-level packages needed.

## Phase 0: Research

Complete. See [research.md](research.md).

Key decisions:
- GGUF: `model.save_pretrained_gguf(path, tokenizer, quantization_method=...)`
- LoRA: `model.save_pretrained(path)` + `tokenizer.save_pretrained(path)`
- Merged: `model.save_pretrained_merged(path, tokenizer, save_method="merged_16bit")`
- HF upload: `HfApi.create_repo(..., exist_ok=True, private=True)` + `HfApi.upload_folder`
- Model card: `huggingface_hub.ModelCard` with f-string content; no LLM generation

## Phase 1: Design

Complete. See [data-model.md](data-model.md), [contracts/cli-contracts.md](contracts/cli-contracts.md),
[quickstart.md](quickstart.md).

### Key Design Points

**PublishingInfo fields**: `description`, `gguf_quantizations` (list, default `["q8_0"]`),
`merged_save_method` (default `"merged_16bit"`), and eight boolean flags:
`save_gguf`, `save_lora`, `save_merged`, `publish_gguf`, `publish_lora`,
`publish_merged` (all default `false`).
All GGUF quant levels share one directory (`gguf/`) and one HF repo (`<model>-gguf`).
Files inside are named `<model-name>-gguf-<quant>.gguf` (e.g. `my-classifier-gguf-q8_0.gguf`).

**Model card content** (generated from configs + live metrics):
1. Title, Description, Model Details, Dataset, Training Config
2. Pre-training classification metrics (omitted if `run_comparison_before_training=False`)
3. Post-training classification metrics
4. Format-specific usage instructions

**Data flow into TrainCommand**: `publishing_helper.save_model()` receives:
- `model`, `tokenizer` — live Unsloth model objects
- `training_info: TrainingInfo`, `dataset_info: DatasetInfo`, `base_model_info: BaseModelInfo`
- `publishing_info: PublishingInfo`
- `pre_metrics: list[MetricResult]`, `post_metrics: list[MetricResult]`

**HF repository naming**: `<hf-username>/<model-name>-<format-slug>` (see contracts).

**CommonPaths changes**:
- Add `PUBLISHING_INFO_DIR = Path("publishing_info")` constant
- Add `OUTPUT_MODELS_DIR = Path("outputs")` constant
- Add `publishing_info` property
- Add `outputs` property (returns path, does NOT auto-create)
- Add `publishing_info` to `ensure_all_dirs_exist()`

## Complexity Tracking

No constitution violations. No entries required.
