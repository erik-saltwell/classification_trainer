# Implementation Plan: Configuration File Reorganization

**Branch**: `001-reorganize-config` | **Date**: 2026-03-05 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-reorganize-config/spec.md`

## Summary

Reorganize configuration files so that training a new model requires only two new YAML files
(`dataset_info` and `training_info`). All other configs (base model, inference, publishing) are
declared by reference inside `training_info` and reused across models. `TrainingInfo` gains
reference fields and lazy-resolution properties. `PublishingInfo` loses its model-specific
`description` field (moved to `TrainingInfo`). CLI commands are simplified to two mandatory args.
All existing real configs are migrated to the new schema and renamed to kebab-case.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: Pydantic v2, Typer, PyYAML, Unsloth, TRL, HuggingFace datasets
**Storage**: YAML config files in `training_info/`, `dataset_info/`, `base_model_info/`,
`inference_info/`, `publishing_info/`, `chat_template_info/`
**Testing**: pytest (unit tests in `tests/unit/`)
**Target Platform**: Linux (local GPU workstation)
**Project Type**: CLI tool / fine-tuning framework
**Performance Goals**: N/A (config loading is not performance-sensitive)
**Constraints**: Must not break existing training workflows; no new dependencies required
**Scale/Scope**: 6 config directories, ~15 YAML files, 4 CLI commands modified

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|---|---|---|
| I. Configuration-First | ✅ Pass | All params remain in YAML. `TrainingInfo` reference fields follow the existing `BaseModelInfo.chat_template` pattern. `inference_info/` added to `ensure_all_dirs_exist()`. No new directories needed beyond what `CommonPaths` already declares. |
| II. Protocol-Based Interfaces | ✅ Pass | No new cross-layer boundaries. No protocol changes. |
| III. Separation of Concerns | ✅ Pass | Reference-resolution properties (`training_info.base_model_info`, etc.) stay in `configuration/`. Commands call `load_training_info()` as before; they then access resolved configs via properties rather than loading them independently. Helper `publishing_helper.py` reads `description` from `training_info` — source changes but the helper layer stays clean. |
| IV. Observability | ✅ Pass | No changes to metrics, WandB, seeds, or logging. |
| V. Simplicity & Scope | ✅ Pass | Changes reduce CLI surface area. New fields/properties are minimal. YAGNI satisfied. |

## Project Structure

### Documentation (this feature)

```text
specs/001-reorganize-config/
├── plan.md              # This file
├── research.md          # Phase 0 — decisions and rationale
├── data-model.md        # Phase 1 — schema changes
├── contracts/
│   └── cli.md           # Phase 1 — new CLI signatures
├── checklists/
│   └── requirements.md  # Spec quality checklist
└── tasks.md             # Phase 2 — /speckit.tasks output
```

### Source Code (affected files)

```text
src/classification_trainer/
├── configuration/
│   ├── training_info.py          # Add base_model, inference, publishing, model_card_description + resolution properties
│   └── publishing_info.py        # Remove description field
├── console/
│   └── main.py                   # Remove --base-model, --inference-info, --publishing-info from train/sweep/analyze-dataset/compute-batch-size
├── helpers/
│   └── publishing_helper.py      # Read description from training_info, not publishing_info
└── utils/
    └── common_paths.py           # Add inference_info to ensure_all_dirs_exist()

training_info/
├── example.yaml                  # Full schema with extensive comments (incl. new fields)
└── reddit-rpg-rules-questions-classifier.yaml   # Add base_model, inference, publishing, model_card_description

dataset_info/
├── example.yaml                  # Full schema with extensive comments
├── rpg-reddit-post-classification.yaml          # Renamed from snake_case
└── imdb.yaml                     # Verify already kebab-case (no rename needed)

base_model_info/
└── example.yaml                  # Full schema with extensive comments

inference_info/
├── example.yaml                  # Full schema with extensive comments
└── simple-classification.yaml    # Verify name (already kebab-case)

publishing_info/
├── example.yaml                  # Full schema with extensive comments (description removed)
└── reddit-rpg-rules-question-classifier.yaml    # Remove description field

chat_template_info/
└── example.yaml                  # Full schema with extensive comments

tests/
└── unit/
    └── test_cli.py               # Verify help/version tests still pass (likely no change needed)
```

## Complexity Tracking

No constitution violations. No complexity justification required.

---

## Implementation Phases

### Phase A — Pydantic Model Changes (foundation; do first, no other phase depends on anything else)

**A1. Update `TrainingInfo`**
- Add fields: `base_model: str`, `inference: str`, `publishing: str | None = None`, `model_card_description: str`
- Add resolution properties:
  - `base_model_info` → `load_base_model_info(self.base_model)`
  - `inference_info` → `load_inference_info(self.inference)`
  - `publishing_info` → `load_publishing_info(self.publishing) if self.publishing else None`
- Imports: add `load_base_model_info`, `load_inference_info`, `load_publishing_info` inside the property bodies (lazy imports to avoid circular import; same pattern as `load_chat_template_info` in `BaseModelInfo`)

**A2. Update `PublishingInfo`**
- Remove `description: str` field
- Update `model_config` if `extra="forbid"` would now reject old YAML files that still have `description` (it will; migrated files must have description removed)

**A3. Update `common_paths.py`**
- Add `self.inference_info.mkdir(parents=True, exist_ok=True)` to `ensure_all_dirs_exist()`

### Phase B — Helper Update

**B1. Update `publishing_helper.py`**
- Change the one call site (`publishing_info.description` at line 192) to read from `training_info.model_card_description`
- Update the function signature of the affected function to accept `training_info: TrainingInfo` if it doesn't already receive it (check current signature)

### Phase C — CLI Simplification

**C1. Update `console/main.py`**
- `train`: Remove `--base-model`, `--inference-info`, `--publishing-info` params. Load via `tr_info.base_model_info`, `tr_info.inference_info`, `tr_info.publishing_info`.
- `sweep`: Remove `--base-model`, `--inference-info`. Load via training info properties.
- `analyze-dataset`: Remove `--base-model`. Load via training info properties.
- `compute-batch-size`: Remove `--base-model`. Load via training info properties.
- `publish`: No change (retains explicit `--publishing-info`).

### Phase D — YAML File Migration

**D1. Migrate `training_info/` real configs**
- `reddit-rpg-rules-questions-classifier.yaml`: Add `base_model`, `inference`, `publishing`, `model_card_description`. Move description text from `publishing_info/reddit-rpg-rules-question-classifier.yaml`.

**D2. Migrate `publishing_info/` real configs**
- `reddit-rpg-rules-question-classifier.yaml`: Remove `description` field.
- `example.yaml`: Remove `description` field; update all comments.

**D3. Rename dataset config**
- Rename `dataset_info/rpg_reddit_post_classification.yaml` → `dataset_info/rpg-reddit-post-classification.yaml`
- Update any references (check if the dataset name is referenced in training config or docs)

**D4. Update `training_info/imdb.yaml`**
- Add `base_model`, `inference`, `publishing`, `model_card_description` fields (or verify if it's a real config that needs migration vs. just an example)

### Phase E — Example YAML Files (all config directories)

Each `example.yaml` must have inline comments for every field covering:
1. What the field represents and how it is used at runtime
2. Required vs. optional (and the default if optional)
3. Allowed values or value range
4. A concrete example where the meaning might otherwise be ambiguous

**E1. `training_info/example.yaml`** — add new fields with full comments; update existing field comments to be complete
**E2. `dataset_info/example.yaml`** — full comments on all fields
**E3. `base_model_info/example.yaml`** — full comments on all fields
**E4. `inference_info/example.yaml`** — full comments on all fields
**E5. `publishing_info/example.yaml`** — full comments on all fields (description removed)
**E6. `chat_template_info/example.yaml`** — full comments on all fields

### Phase F — Tests

**F1. Verify existing tests still pass** — `test_cli.py` (help/version) should be unaffected.
**F2. Add unit tests for `TrainingInfo` resolution properties** — test that `base_model_info`, `inference_info`, `publishing_info` properties load the correct config, and that a missing referenced config raises `FileNotFoundError` with a clear message.
**F3. Add unit tests for `PublishingInfo`** — verify that YAML without `description` field loads without error (i.e., Pydantic `extra="forbid"` does not reject existing fields).

## Dependency Order

```
A (Pydantic models) → B (helpers) → C (CLI)
A (Pydantic models) → D (YAML migration)
E (example files) — independent, can be done any time
F (tests) — after A, B, C
```

Phases A, D, E can be worked in parallel by a single developer.
B and C depend on A being complete.
F runs last.
