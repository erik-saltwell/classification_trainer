# Research: Configuration File Reorganization

**Branch**: `001-reorganize-config` | **Date**: 2026-03-05

## Findings

### Decision 1: Reference resolution via `TrainingInfo` properties

**Decision**: Add `base_model: str`, `inference: str`, and `publishing: str | None` string fields to `TrainingInfo`. Resolution to actual config objects is done via `@property` methods on the model (e.g., `training_info.base_model_info`, `training_info.inference_info`, `training_info.publishing_info`).

**Rationale**: `BaseModelInfo` already uses this exact pattern — its `chat_template_info` property calls `load_chat_template_info(self.chat_template)`. This feature extends the same established pattern rather than inventing a new one. Keeps `TrainingInfo` as the single "bundle" file while preserving the pure-Pydantic nature of the config layer.

**Alternatives considered**:
- A separate `load_training_bundle()` factory function returning a tuple of all resolved configs — rejected because it requires callers to unpack multiple return values, and the property pattern already exists.
- A new "run profile" YAML grouping all config names — rejected (added unnecessary new config category; user explicitly chose Option A in clarification).

---

### Decision 2: `model_card_description` moves to `TrainingInfo`

**Decision**: Add `model_card_description: str` field to `TrainingInfo`. Remove `description: str` from `PublishingInfo`. `publishing_helper.py` reads description from `training_info` instead.

**Rationale**: `model_name` and `hugging_face_user_name` are already in `TrainingInfo`; the description is equally model-specific and belongs alongside them. This makes `PublishingInfo` contain only reusable artifact-format settings.

**Alternatives considered**:
- Keep description optional in `PublishingInfo` with `None` default — rejected; still puts model-specific text in a reusable file and creates an awkward None-vs-empty state.

---

### Decision 3: CLI args `--base-model`, `--inference-info`, `--publishing-info` removed

**Decision**: Remove these three optional/required args from `train`, `sweep`, `analyze-dataset`, and `compute-batch-size`. The `publish` command retains `--publishing-info` as an override since it runs independently of a training session and a user may want to republish with different format settings.

**Rationale**: The spec (FR-002a, FR-002b) permits removal. Removing them simplifies the invocation to the two mandatory args (`--dataset`, `--training-info`) without losing any capability — the info is now in the YAML. The `publish` command is a special case: it already takes `--training-info` to get model identity, and a user might legitimately want to re-publish to different formats without editing their training config. Retaining `--publishing-info` there preserves that flexibility.

**Alternatives considered**:
- Deprecation warnings with backwards-compatible pass-through for old args — rejected; adds complexity for no real benefit since all existing callers are internal scripts easily updated.

---

### Decision 4: Kebab-case file naming enforced during migration

**Decision**: Rename `dataset_info/rpg_reddit_post_classification.yaml` → `dataset_info/rpg-reddit-post-classification.yaml`. All new files created during migration use kebab-case. No code enforcement (just convention + documentation in example files).

**Rationale**: The spec (FR-010) mandates kebab-case. Code-level enforcement (e.g., a validator on the name string) is not necessary — a convention documented in `example.yaml` and the comments is sufficient, consistent with Principle V (Simplicity).

---

### Decision 5: `inference_info` added to `ensure_all_dirs_exist()`

**Decision**: `inference_info/` is currently NOT in `ensure_all_dirs_exist()` (it exists in `CommonPaths` as a property but isn't auto-created). Since `inference_info` is now a required reference target (all training configs must name an inference config), it should be created on startup alongside the other config dirs.

**Rationale**: Principle I requires config directories to be registered in `CommonPaths.ensure_all_dirs_exist()`. The inference_info directory was previously optional from the CLI perspective; it is now always required.

---

### Decision 6: `example.yaml` field required for `publishing: str | None`

**Decision**: The `publishing` field in `TrainingInfo` is optional (`None` by default). The example documents it as optional with `null` as the default and explains that setting it causes model artifacts to be saved/published after training.

**Rationale**: Publishing is not always desired (e.g., during a hyperparameter sweep or exploratory training run). The optional pattern is already established for `publishing_info` in the CLI today.
