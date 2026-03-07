# Research: Dataset Reference in Training Config

**Feature**: 006-dataset-ref-training-info | **Date**: 2026-03-07

## R1: Field Name and Pattern

**Decision**: Name the field `dataset` and add a `dataset_info` property that lazy-loads the `DatasetInfo`, following the exact pattern of `base_model`/`base_model_info`, `inference`/`inference_info`, and `publishing`/`publishing_info`.

**Rationale**: The existing pattern in `TrainingInfo` is: short name field (e.g., `base_model: str`) with a property (e.g., `base_model_info -> BaseModelInfo`) that loads from the corresponding directory. Replicating this pattern ensures consistency and discoverability.

**Alternatives considered**:
- `dataset_info` as the field name: Rejected — breaks the pattern where short names like `base_model`, `inference`, `publishing` are used, with `_info` reserved for the property that returns the loaded model.
- Optional field with default: Rejected — every training run requires a dataset, so making it optional would just delay the error.

## R2: CLI Migration Strategy

**Decision**: Remove `--dataset` from all four commands in a single change. No deprecation period.

**Rationale**: This is a small single-user project. A clean break is simpler and less error-prone than maintaining deprecated args. The CLAUDE.md already documents the CLI signature as `--dataset` + `--training-info`, and the 003-reorganize-config spec anticipated this change (FR-002a: "A training run MUST be launchable by specifying only a dataset config name and a training config name on the CLI").

## R3: Command Class Changes

**Decision**: Remove `dataset_info: DatasetInfo` parameter from all command `@dataclass` classes. Each command accesses `self.training_info.dataset_info` instead.

**Rationale**: The `dataset_info` is fully derivable from `training_info`. Removing it from command constructors eliminates redundant data and the possibility of mismatched dataset/training configs.

## R4: Existing Config Migration

**Decision**: Add `dataset` field to all existing training config YAMLs based on their known dataset pairings:
- `imdb.yaml` → `dataset: "imdb"`
- `reddit-rpg-questions-classifier.yaml` → `dataset: "reddit-rpg-questions"`
- `test-reddit-questions.yaml` → `dataset: "test-reddit-questions"`

**Rationale**: The pairings are deterministic from the project's usage. Migration ensures all configs work immediately after the change.
