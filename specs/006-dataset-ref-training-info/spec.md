# Feature Specification: Dataset Reference in Training Config

**Feature Branch**: `006-dataset-ref-training-info`
**Created**: 2026-03-07
**Status**: Draft
**Input**: User description: "Add a field to the training info YAML to specify the dataset info YAML to use (filename stem, no extension), following the existing pattern for base_model, inference, and publishing references. Update the training info example YAML with documentation. Remove the --dataset CLI argument from all commands and resolve the dataset from the training config instead."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Launch Any Command with Only --training-info (Priority: P1)

A practitioner wants to run a training, sweep, analyze-dataset, or compute-batch-size command. Instead of specifying both `--dataset` and `--training-info`, they specify only `--training-info`. The system resolves the dataset config from the `dataset` field inside the training config YAML — just as it already resolves the base model, inference, and publishing configs.

**Why this priority**: This is the core change — moving the dataset reference into the training config and simplifying all CLI commands from two required arguments to one.

**Independent Test**: Can be fully tested by adding `dataset: "imdb"` to a training config YAML, running any CLI command with only `--training-info` (no `--dataset`), and confirming it loads the correct dataset config.

**Acceptance Scenarios**:

1. **Given** a training config with `dataset: "imdb"`, **When** the user runs the train command with only `--training-info`, **Then** the system loads `dataset_info/imdb.yaml` and training proceeds as expected.
2. **Given** a training config with `dataset: "reddit-rpg-questions"`, **When** the user runs the analyze-dataset command with only `--training-info`, **Then** the system analyzes the correct dataset.
3. **Given** a training config with `dataset: "imdb"`, **When** the user runs the sweep command with only `--training-info`, **Then** the sweep uses the correct dataset.
4. **Given** a training config with `dataset: "imdb"`, **When** the user runs the compute-batch-size command with only `--training-info`, **Then** the batch size is computed against the correct dataset.

---

### User Story 2 - Error on Missing or Invalid Dataset Reference (Priority: P2)

A practitioner has a training config that either omits the `dataset` field or references a dataset config file that does not exist. The system reports a clear error before attempting any work.

**Why this priority**: Without clear error messages, users will be confused when the system fails to find the dataset.

**Independent Test**: Can be fully tested by running a command with a training config that has no `dataset` field, or one that references a nonexistent dataset name, and confirming a clear error is shown.

**Acceptance Scenarios**:

1. **Given** a training config with no `dataset` field, **When** the user runs any command, **Then** the system reports a clear error indicating the `dataset` field is required.
2. **Given** a training config with `dataset: "nonexistent-dataset"`, **When** the user runs any command, **Then** the system reports a clear error stating which dataset config file was not found and where it was expected.

---

### User Story 3 - Discover the Feature via Documentation (Priority: P3)

A practitioner browsing the training config example YAML discovers the `dataset` field. The example YAML explains the field's purpose, format, and where the referenced file must be located — following the same documentation pattern as `base_model`, `inference`, and `publishing`.

**Why this priority**: Without documentation, users won't know to add the `dataset` field or understand the naming convention.

**Independent Test**: Can be fully tested by reading the `training_info/example.yaml` file and confirming the `dataset` field is documented with inline comments explaining its purpose, format, and a concrete example.

**Acceptance Scenarios**:

1. **Given** the `training_info/example.yaml` file, **When** a user reads the references section, **Then** they find a `dataset` field with inline comments explaining: what it does, the naming convention (filename stem, no .yaml extension), and a concrete example value.

---

### Edge Cases

- What happens when a training config has a `dataset` field referencing a file that exists but contains invalid YAML? The system MUST report the validation error from the dataset config, not a generic failure.
- What happens when a user passes a `--dataset` argument that no longer exists on the CLI? The CLI MUST reject the unknown argument with a standard error (no silent ignoring).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The training config YAML MUST include a required `dataset` field containing the filename stem (no .yaml extension) of the dataset config to use. This field follows the same pattern as `base_model`, `inference`, and `publishing`.
- **FR-002**: The system MUST resolve the dataset config by loading the file at `dataset_info/<dataset>.yaml` when the training config is used.
- **FR-003**: The `--dataset` CLI argument MUST be removed from all commands that currently accept it: `train`, `sweep`, `analyze-dataset`, and `compute-batch-size`.
- **FR-004**: All four commands MUST resolve the dataset config from the training config's `dataset` field instead of from a CLI argument.
- **FR-005**: If the `dataset` field is missing from the training config, the system MUST reject the config at load time with a clear error.
- **FR-006**: If the referenced dataset config file does not exist, the system MUST report a clear error identifying the missing file path.
- **FR-007**: The `training_info/example.yaml` MUST include the `dataset` field in the references section, with inline comments following the same documentation pattern as `base_model`, `inference`, and `publishing`.
- **FR-008**: All existing training config YAML files MUST be updated to include the correct `dataset` field value, so they continue to work with the new CLI interface.

### Key Entities

- **Dataset Reference**: A filename stem in the training config that identifies which dataset config YAML to load, following the same naming convention as other config references in the training config.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A user can launch any command (train, sweep, analyze-dataset, compute-batch-size) with only `--training-info` — no `--dataset` argument needed or accepted.
- **SC-002**: The `dataset` field in the training config follows the exact same pattern as `base_model`, `inference`, and `publishing` — filename stem, no extension, loaded from the corresponding config directory.
- **SC-003**: All existing training config YAML files are updated with the correct `dataset` field and continue to work with the new single-argument CLI.
- **SC-004**: Missing or invalid dataset references produce clear, actionable error messages before any processing begins.
- **SC-005**: The `example.yaml` for training config documents the `dataset` field in the references section, matching the documentation pattern of adjacent fields.

## Assumptions

- The `dataset` field is required (not optional) on `TrainingInfo`. Every training run requires a dataset — there is no sensible default.
- The field is named `dataset` (not `dataset_info`) to match the pattern of `base_model`, `inference`, and `publishing` — short names, not full model names.
- Existing training config YAMLs (`imdb.yaml`, `reddit-rpg-questions-classifier.yaml`, `test-reddit-questions.yaml`) will be migrated to include the `dataset` field as part of this feature.
- The `--dataset` CLI argument is removed entirely — no deprecation period. This is a small project with a single user; breaking the CLI cleanly is preferred over a gradual migration.
