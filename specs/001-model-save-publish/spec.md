# Feature Specification: Model Save and Publish

**Feature Branch**: `001-model-save-publish`
**Created**: 2026-03-05
**Status**: Draft
**Input**: User description: "at the end of the train command, save the best performing version of the model to disk, then add a new command to publish to hugging face. The model should be save-able in the following formats: gguf for ollama, lora adapter, merged HF checkpoint, AWQ for vllm. Make a new publishing config file that has saving and publishing data."

## Clarifications

### Session 2026-03-05

- Q: Should each format published to HuggingFace be its own model repo, with the repo name being a concatenation of the model name and format? → A: Yes. Each format is its own HuggingFace repository named `<model-name>-<format-slug>` (e.g., `my-classifier-gguf`).
- Q: Should a model card be generated when saving to disk, and should the saved card be used when publishing? → A: Yes. A model card is generated alongside every saved format, auto-populated from the dataset, training, and publishing config files. The publish command uses this saved card rather than regenerating it.
- Q: Should the publishing config include a description field for use in model cards? → A: Yes. The publishing config MUST include a `description` field that appears as the primary description in every generated model card.
- Q: Should training evaluation metrics be included in the model card? → A: Yes — both pre-training and post-training classification metrics (e.g., accuracy, F1) MUST be included in the model card.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Auto-Save After Training (Priority: P1)

A researcher runs the train command. When training completes, the best-performing model
checkpoint is automatically saved to local disk in the formats they have configured in
a publishing config file. For each format, a model card is generated alongside the
artifacts, populated from the dataset, training, and publishing config data.
The researcher can immediately use those artifacts with Ollama, vLLM, or other serving
tools without any additional steps.

**Why this priority**: This is the foundation of the feature. Without local artifacts
and model cards there is nothing to publish. It also delivers standalone value: local
deployment does not require HuggingFace credentials or network access.

**Independent Test**: Run `train` with a publishing config that specifies at least one
format. Verify the expected subdirectory structure, model files, and `README.md` model
card appear under `output_models/`.

**Acceptance Scenarios**:

1. **Given** a publishing config with one or more save formats enabled,
   **When** the train command finishes,
   **Then** `output_models/<hf-model-name>/<format>/` directories exist containing
   the correct artifacts and a `README.md` model card for each enabled format.

2. **Given** a publishing config with no formats enabled,
   **When** the train command finishes,
   **Then** no `output_models/` directory is created and training completes normally.

3. **Given** a training run where `output_models/<format>/` already contains a previous
   save,
   **When** training completes again,
   **Then** the existing artifacts and model card are overwritten with the new best
   checkpoint and freshly generated card.

---

### User Story 2 - Publish Saved Model to HuggingFace (Priority: P2)

After saving a model locally, a researcher runs a separate `publish` command, passing
a training config name and a publishing config name. The command uploads each enabled
format to its own dedicated HuggingFace repository, using the locally saved model card
as the repository README, without regenerating it.

**Why this priority**: Publishing is a downstream step that requires saved artifacts
and model cards (US1). Having it as a separate command lets researchers inspect local
artifacts before uploading.

**Independent Test**: With locally saved artifacts and model cards (from US1), run
`publish` and verify that a separate HuggingFace repository exists for each format,
each containing the correct model files and the matching `README.md`.

**Acceptance Scenarios**:

1. **Given** saved artifacts and model cards on disk and valid HuggingFace credentials,
   **When** the publish command is run with the correct config names,
   **Then** each enabled publish format is uploaded to its own HuggingFace repository
   named `<hf-username>/<model-name>-<format-slug>`, and the repository README is
   the locally saved model card for that format.

2. **Given** no locally saved model exists for the specified config,
   **When** the publish command is run,
   **Then** the command fails with a clear error message indicating the model must
   be saved first.

3. **Given** invalid or missing HuggingFace credentials,
   **When** the publish command is run,
   **Then** the command fails with a clear message describing the authentication problem.

---

### User Story 3 - Configure Save, Publish, and Model Card Content (Priority: P3)

A researcher creates or edits a publishing config YAML file to control which formats
are saved locally and which are uploaded to HuggingFace, the destination HuggingFace
username, and a human-readable description that will appear in the generated model
cards for all formats.

**Why this priority**: The config is required for US1 and US2 to work, but it is a
one-time setup step rather than a repeated action, making it lower urgency relative
to the core save/publish flows.

**Independent Test**: Write a publishing config YAML with a `description` field, run
validation, and confirm the config loads without errors and that the generated model
card contains the specified description text.

**Acceptance Scenarios**:

1. **Given** a YAML file in the publishing config directory with valid fields including
   a `description`,
   **When** the train command loads it and saves a format,
   **Then** the generated model card contains the `description` text from the config.

2. **Given** a YAML file with an unrecognised format name or missing required field,
   **When** the train or publish command loads it,
   **Then** the command fails immediately with a validation error listing the invalid
   field(s).

---

### Edge Cases

- What if disk space is exhausted during a GGUF or merged-checkpoint save? The save
  operation should fail with an informative message; partially-written files should
  be cleaned up.
- What if the HuggingFace repository named for a format does not yet exist?
  The publish command should create it automatically (private by default).
- What if only some formats are selected in the publishing config — should unselected
  format directories be left alone or deleted? Unselected directories are left untouched;
  only configured formats are written.
- What if the train command is interrupted before saving? Saving occurs only after a
  successful training completion; interrupted runs produce no artifacts.
- What if the publishing config `description` field is empty? The model card is still
  generated with an empty description section; no error is raised.
- What if the locally saved model card is missing when the publish command runs?
  The publish command fails for that format with a clear error indicating the save
  must be re-run.
- What if the training run did not capture pre-training metrics (e.g., `run_comparison_before_training=False`)?
  The model card is still generated, but the pre-training metrics section is omitted
  with a note that pre-training evaluation was not performed.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The train command MUST automatically save the best-performing model
  checkpoint to disk in all formats enabled in the publishing config, immediately
  after training completes.
- **FR-002**: Saved artifacts MUST be stored under `output_models/<hf-model-name>/
  <format-slug>/` relative to the project root, where `hf-model-name` is the
  HuggingFace model name from `TrainingInfo`.
- **FR-003**: The system MUST support saving in four formats: GGUF (slug: `gguf`,
  for Ollama), LoRA adapter (slug: `lora`), merged HuggingFace checkpoint
  (slug: `merged`), and AWQ (slug: `awq`, for vLLM).
- **FR-004**: A new `publish` CLI command MUST upload saved model artifacts to
  HuggingFace Hub, creating one separate repository per format.
- **FR-005**: Each HuggingFace repository MUST be named `<hf-username>/<model-name>-
  <format-slug>` (e.g., `alice/my-classifier-gguf`), where `hf-username` comes from
  `TrainingInfo` and `model-name` is the base model name.
- **FR-006**: The publishing config MUST independently control which formats are saved
  locally and which are published to HuggingFace; a format may be published without
  being separately retained locally.
- **FR-007**: A new publishing config YAML schema MUST be defined and validated via
  a Pydantic model, stored in a `publishing_info/` directory at the project root.
- **FR-008**: The publishing config MUST include: enabled save formats, enabled publish
  formats, HuggingFace username, a `description` string, and per-format options
  (e.g., GGUF quantisation level).
- **FR-009**: The train command MUST skip saving entirely if no publishing config is
  supplied or if all save formats are disabled, with no error.
- **FR-010**: For every format saved to disk, the system MUST generate a `README.md`
  model card in that format's save directory alongside the model artifacts.
- **FR-011**: The model card MUST be auto-populated from the dataset config, training
  config, and publishing config; the `description` field from the publishing config
  MUST appear as the primary description section of the card. The card MUST also
  include both pre-training and post-training classification metrics (e.g., accuracy,
  F1) captured during the training run that produced the saved checkpoint.
- **FR-012**: The publish command MUST use the locally saved `README.md` model card
  when uploading to HuggingFace; it MUST NOT regenerate the card during publish.
- **FR-013**: The publish command MUST fail if the locally saved model card for a
  format is missing, with a clear error directing the user to re-run the save step.
- **FR-014**: The publish command MUST emit a progress indicator for each format being
  uploaded and report success or failure per format.
- **FR-015**: The publish command MUST fail fast with a human-readable error if no
  locally saved artifacts exist for the requested model and format.

### Key Entities

- **PublishingInfo**: Config model loaded from `publishing_info/<name>.yaml`. Fields
  include: `save_formats` (list), `publish_formats` (list), `huggingface_username`
  (string), `description` (string), and per-format options (e.g., `gguf_quantization`).
- **SaveFormat**: An enumerated value representing one of the four supported output
  formats with associated slugs: `gguf`, `lora`, `merged`, `awq`.
- **SavedModelArtifact**: A collection of files on disk representing one saved format
  of a trained model, living at `output_models/<model-name>/<format-slug>/`, including
  a `README.md` model card.
- **ModelCard**: A `README.md` file generated per saved format, containing the model
  description (from publishing config), base model details, dataset summary, training
  configuration summary, and pre/post-training classification metrics captured during
  the training run.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: After any training run that references a publishing config with at least
  one format enabled, the corresponding `output_models/` artifact directories exist,
  are non-empty, and each contains a `README.md` model card that includes both
  pre-training and post-training classification metrics.
- **SC-002**: A researcher can run the publish command and have each enabled format
  appear as its own HuggingFace repository (named per FR-005), with the correct model
  files and the matching `README.md` as the repository description.
- **SC-003**: An invalid publishing config (bad format name, missing required field)
  is caught at command startup — before any model loading or training begins — and
  reported with a clear error message.
- **SC-004**: The train command's existing behaviour (training, evaluation, WandB
  reporting) is unaffected when no publishing config is specified.
- **SC-005**: The `description` text written in the publishing config appears verbatim
  in every generated model card for that publishing run.

## Assumptions

- HuggingFace credentials are available via the standard `HF_TOKEN` environment
  variable or the `huggingface-cli login` flow; the publish command does not manage
  credential storage.
- GGUF quantisation defaults to Q8_0 unless overridden in the publishing config.
- The best-performing checkpoint is the one selected by `load_best_model_at_end=True`
  in the existing training setup (lowest eval loss by default).
- Publishing config YAML files are stored in a new `publishing_info/` directory
  at the project root, consistent with `training_info/` and `dataset_info/`.
- AWQ quantisation requires sufficient VRAM; if unavailable, the save step for
  that format fails with an actionable error.
- Model card content derived from config files uses the YAML field values directly
  (e.g., dataset name, base model name, training hyperparameters); no LLM-generated
  prose is involved.
