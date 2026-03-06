# Feature Specification: Configuration File Reorganization

**Feature Branch**: `001-reorganize-config`
**Created**: 2026-03-05
**Status**: Draft
**Input**: User description: "I want to re-organize our configuration files. My main goal is that if i am training a new model, I will likely need to implement a new DatasetInfo yaml, and will certainly need a new TrainingInfo.yaml, but for all other yaml files, i should be able to re-use existing configuration files if the model is trained in a similar manner."

## Clarifications

### Session 2026-03-05

- Q: How do other configs get resolved when only dataset + training configs are specified at the CLI? → A: The `training_info` YAML gains named reference fields for base model, inference, and publishing configs. CLI args for those configs (--base-model, --inference-info, --publishing-info) may be removed.
- Q: Where does the model card description field move from `publishing_info`? → A: Into `training_info`, alongside model_name and hugging_face_user_name, making `publishing_info` entirely free of model-specific content.
- Q: Are existing real config files migrated to the new structure as part of this feature? → A: Yes — all existing configs are migrated in this feature; no unmigrated files remain after delivery.
- Q: What naming convention should config files follow? → A: kebab-case for all config files (e.g., `my-model-name.yaml`).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Start Training a New Model with Minimal New Config (Priority: P1)

A user wants to train a new classifier on a different dataset using the same base model and training approach as a previous run. They should only need to create two new YAML files (one for the dataset and one for the training run), then reference existing base model, inference, and publishing configs by name.

**Why this priority**: This is the core use case described by the user. Reducing the number of files to create for each new model minimizes mistakes and speeds up the workflow.

**Independent Test**: Can be fully tested by preparing a new dataset and training scenario, creating only a dataset config and a training config, and successfully launching a training run using existing configs for all other parameters — verifying no additional config files were created.

**Acceptance Scenarios**:

1. **Given** an existing base model config and inference config, **When** a user trains a new model by creating only a new dataset config and a new training config, **Then** training launches successfully using the referenced existing configs without modification.
2. **Given** a training config that references a named base model config, **When** the referenced config does not exist, **Then** the system reports a clear error identifying which referenced config is missing and where to find or create it.

---

### User Story 2 - Discover and Browse Reusable Configs (Priority: P2)

A user wants to know which existing base model, inference, and publishing configs are available before starting a new training run, so they can choose appropriate ones to reference rather than creating duplicates.

**Why this priority**: Reuse only works if users can easily discover what exists. Without this, users create duplicate configs or miss suitable options.

**Independent Test**: Can be fully tested by browsing the config directories and identifying all available reusable configs for each category without needing to read code — the directory structure and file naming alone should make the categories and reusability obvious.

**Acceptance Scenarios**:

1. **Given** a set of existing reusable configs, **When** a user browses the config directories, **Then** the directory layout clearly separates configs that are always model-specific (dataset, training) from those that can be shared across models (base model, inference, publishing).
2. **Given** a previously used base model config, **When** a user trains a new model using the same base model, **Then** they can reference the existing config by name without copying or modifying it.

---

### User Story 3 - Author a New Config Using the Example as a Reference (Priority: P3)

A user needs to create a new config file for a category they haven't used before (e.g., a new inference config or a new base model config). They open the `example.yaml` in the relevant directory and use it as a self-contained reference — reading the inline comments to understand what each field means, whether it is required, and what values are allowed — without needing to consult external documentation or read source code.

**Why this priority**: Example files with thorough inline documentation eliminate the need to look at source code or ask for help when authoring any config. This lowers the barrier to creating correctly-structured configs and reduces misconfiguration errors.

**Independent Test**: Can be fully tested by giving a new user only the example YAML files (no other documentation) and asking them to author a valid config for each category — verifying they can produce a working config for every field without outside assistance.

**Acceptance Scenarios**:

1. **Given** only the `example.yaml` file for a config category, **When** a user wants to add a new field or change an existing one, **Then** the example comments tell them the field's purpose, whether it is required or optional, and what values are accepted.
2. **Given** an `example.yaml` that documents every field in the schema, **When** the schema for that config category changes (field added or removed), **Then** the example YAML is updated in the same change to stay accurate.
3. **Given** an `example.yaml` for any config category, **When** a user copies it and fills in their values, **Then** the resulting config is valid and accepted by the system without modification.

---

### User Story 4 - Create a New Base Model Config for Future Reuse (Priority: P4)

A user wants to add support for a new base model that isn't yet in the config library. They create a base model config once, and from that point forward can reference it in any number of training runs.

**Why this priority**: The value of this system grows as the library of reusable configs expands. Users should feel that adding a new base model config is a one-time investment, not per-model overhead.

**Independent Test**: Can be fully tested by creating a new base model config and using it in two separate training runs — verifying both runs succeed and the config was not duplicated or modified between runs.

**Acceptance Scenarios**:

1. **Given** a new base model config placed in the shared config location, **When** a user references it in a training config, **Then** training loads the correct base model without requiring any other config changes.

---

### Edge Cases

- What happens when a training config references a base model or inference config that has been deleted or renamed? The system must produce a clear, actionable error rather than a generic file-not-found message.
- What happens when the user creates a new training config with the same model name as an existing one? The system should warn or prevent overwriting existing model output artifacts.
- How should the user handle a case where no existing inference or publishing config suits their needs? Each directory's `example.yaml` serves as a fully-commented starting point — the user copies it, fills in their values guided by the inline comments, and places the new file alongside the examples.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The config directory structure MUST visually separate configs that are always created per model (dataset config, training config) from configs that are intended for reuse across models (base model config, inference config, publishing config).
- **FR-002**: The `training_info` YAML MUST contain named reference fields for the base model config, inference config, and (optionally) publishing config. These fields identify which reusable config to load by name, making the training config the single file that ties a run together.
- **FR-002a**: A training run MUST be launchable by specifying only a dataset config name and a training config name on the CLI. The system resolves all other configs from the references embedded in the training config.
- **FR-002b**: CLI commands (train, sweep, analyze-dataset, compute-batch-size) MAY remove the separate `--base-model`, `--inference-info`, and `--publishing-info` arguments, as these are now declared inside the training config. Existing invocations that still pass those args MUST either be supported via a deprecation warning or clearly documented as broken.
- **FR-003**: Reusable config files (base model, inference, publishing) MUST be identifiable by name alone — a user MUST be able to select an existing reusable config without reading its contents.
- **FR-004**: The system MUST validate that all referenced config names exist and report missing references with the name of the missing config and the expected location.
- **FR-005**: Existing training workflows (train, sweep, publish, analyze-dataset commands) MUST continue to produce correct results after the reorganization. CLI argument signatures for `--base-model`, `--inference-info`, and `--publishing-info` MAY be removed; if removed, the change MUST be documented clearly.
- **FR-009**: All existing real config files (across all config directories) MUST be migrated to the new structure as part of this feature. No config file in an unmigrated format MUST remain after delivery.
- **FR-010**: All config files MUST use kebab-case naming (e.g., `my-model-name.yaml`). Existing files with snake_case or mixed-case names MUST be renamed as part of the migration in FR-009.
- **FR-006**: Every config directory (dataset, training, base model, inference, publishing, chat template) MUST contain an `example.yaml` file that covers every field defined in that config's schema.
- **FR-007**: Each `example.yaml` MUST include inline comments for every field documenting: (a) what the field represents and how it is used at runtime, (b) whether the field is required or optional, (c) the allowed values or value range, and (d) a concrete example value where the meaning might otherwise be ambiguous.
- **FR-008**: The model card description field MUST be moved from `publishing_info` into `training_info`, alongside `model_name` and `hugging_face_user_name`. The `publishing_info` config MUST contain only reusable settings (save flags, quantization levels, merged save method) with no model-specific content.

### Key Entities

- **Dataset Config**: Describes the source dataset — always model-specific (dataset location, column names, label definitions). One new file required per model.
- **Training Config**: Describes the training run — always model-specific (model name, hyperparameters, wandb project). One new file required per model.
- **Base Model Config**: Describes the pretrained base model to fine-tune from (model ID, chat template). Reusable across all models trained from the same base.
- **Inference Config**: Describes evaluation and inference settings (sampling, metrics). Reusable across models evaluated in the same way.
- **Publishing Config**: Describes how to save and publish model artifacts (save flags, quantization levels, merged save method). Contains no model-specific content. Fully reusable across models with the same artifact preferences.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A user can start training a new classifier by creating exactly 2 new YAML files (dataset config and training config), with all other parameters resolved from existing reusable configs.
- **SC-002**: All existing training, sweep, publish, and analyze-dataset commands continue to work without any change to their invocation syntax after the reorganization.
- **SC-003**: A user new to the project can identify which config files are model-specific and which are reusable within 2 minutes of browsing the directory structure, without reading code.
- **SC-004**: Zero duplicate base model, inference, or publishing configs exist across two separate model training setups that use the same base model and training approach.
- **SC-005**: Every config category has an `example.yaml` where every field is covered by inline comments addressing purpose, required/optional status, and allowed values — verified by inspection without running any code.
- **SC-006**: A user with no prior knowledge of the project can author a valid, working config for any category using only the corresponding `example.yaml` as reference, without consulting source code or external documentation.
- **SC-007**: After delivery, every existing model training scenario (imdb, reddit-rpg classifier) can be launched using the new two-argument CLI invocation (`--dataset` + `--training-info` only), with no remaining config files in the old format.

## Assumptions

- The current six config categories (dataset, training, base model, inference, publishing, chat template) remain the correct set — no new categories are introduced in this feature.
- The `training_info` YAML will continue to contain both model identity (name) and hyperparameters; splitting these into separate files is out of scope unless the user explicitly requests it.
- The model card description field moves from `publishing_info` into `training_info`; after this move, `publishing_info` contains no model-specific content and is fully reusable.
- Chat template configs are already fully reusable and referenced from base model configs; they require no structural changes.
- All config files use kebab-case naming; the existing snake_case dataset config (`rpg_reddit_post_classification.yaml`) is renamed as part of migration.
- The CLI command signatures for `--dataset` and `--training-info` are unchanged. CLI arguments for `--base-model`, `--inference-info`, and `--publishing-info` may be removed since those configs are now referenced from within the training config.
