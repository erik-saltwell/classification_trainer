# Implementation Plan: Dataset Reference in Training Config

**Branch**: `006-dataset-ref-training-info` | **Date**: 2026-03-07 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/006-dataset-ref-training-info/spec.md`

## Summary

Add a required `dataset` field to `TrainingInfo` (filename stem, no extension) following the existing pattern for `base_model`, `inference`, and `publishing`. Add a `dataset_info` property that loads and returns the `DatasetInfo`. Remove `--dataset` from all four CLI commands (`train`, `sweep`, `analyze-dataset`, `compute-batch-size`). Update all command classes and calling code to resolve the dataset from the training config. Migrate all existing training config YAMLs. Document in `example.yaml`.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: Pydantic v2, Typer, PyYAML
**Storage**: YAML config files (`training_info/`, `dataset_info/`)
**Testing**: pytest
**Target Platform**: Linux (GPU workstation)
**Project Type**: CLI tool
**Performance Goals**: N/A (config loading, no hot path)
**Constraints**: Clean CLI break — no deprecation period
**Scale/Scope**: Single-user CLI tool; 4 commands affected

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Configuration-First | PASS | Feature moves the dataset binding into the training config — directly aligned. Follows existing `base_model`/`inference`/`publishing` pattern. |
| II. Protocol-Based Interfaces | PASS | No new protocols needed. Command classes continue using existing protocols. |
| III. Separation of Concerns | PASS | Config field added to `configuration/training_info.py`. CLI changes in `console/main.py`. Command classes in `commands/`. No cross-layer violations. |
| IV. Observability | PASS | Error messages for missing/invalid dataset references use `LoggingProtocol`. |
| V. Simplicity & Scope | PASS | Reduces CLI surface area (fewer args). Follows existing pattern — no new abstractions. |

## Project Structure

### Documentation (this feature)

```text
specs/006-dataset-ref-training-info/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   └── cli-commands.md  # Updated CLI contract
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
src/classification_trainer/
├── configuration/
│   └── training_info.py         # MODIFIED: add dataset field + dataset_info property
├── commands/
│   ├── analyze_dataset.py       # MODIFIED: remove dataset_info param, use training_info.dataset_info
│   ├── sweep.py                 # MODIFIED: remove dataset_info param, use training_info.dataset_info
│   ├── train.py                 # MODIFIED: remove dataset_info param, use training_info.dataset_info
│   └── compute_batch_size.py    # MODIFIED: remove dataset_info param, use training_info.dataset_info
├── console/
│   └── main.py                  # MODIFIED: remove --dataset from all 4 commands, resolve from training_info

training_info/
├── example.yaml                 # MODIFIED: add documented dataset field
├── imdb.yaml                    # MODIFIED: add dataset field
├── reddit-rpg-questions-classifier.yaml  # MODIFIED: add dataset field
└── test-reddit-questions.yaml   # MODIFIED: add dataset field

tests/
└── unit/
    ├── test_training_info.py    # MODIFIED: add tests for dataset field
    └── test_cli.py              # MODIFIED: update CLI tests if they reference --dataset
```

**Structure Decision**: Single-project CLI structure. All changes modify existing files. No new files needed.

## Complexity Tracking

No constitution violations. No complexity justifications needed.
