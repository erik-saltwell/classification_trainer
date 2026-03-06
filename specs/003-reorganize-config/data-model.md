# Data Model: Configuration File Reorganization

**Branch**: `001-reorganize-config` | **Date**: 2026-03-05

## Config Schema Changes

### `TrainingInfo` (modified)

Fields added:

| Field | Type | Required | Description |
|---|---|---|---|
| `base_model` | `str` | Yes | Filename stem of the `base_model_info` YAML to load (e.g., `qwen2.5-0.5b-instruct`) |
| `inference` | `str` | Yes | Filename stem of the `inference_info` YAML to load (e.g., `simple-classification`) |
| `publishing` | `str \| None` | No (default: `null`) | Filename stem of the `publishing_info` YAML to load; `null` disables save/publish |
| `model_card_description` | `str` | Yes | Human-readable description used in the HuggingFace model card |

Properties added (lazy loaders following the `BaseModelInfo.chat_template_info` pattern):

| Property | Return Type | Behaviour |
|---|---|---|
| `base_model_info` | `BaseModelInfo` | Calls `load_base_model_info(self.base_model)` |
| `inference_info` | `InferenceInfo` | Calls `load_inference_info(self.inference)` |
| `publishing_info` | `PublishingInfo \| None` | Calls `load_publishing_info(self.publishing)` if `self.publishing` is not None, else `None` |

No fields removed from `TrainingInfo`.

---

### `PublishingInfo` (modified)

Fields removed:

| Field | Moved To |
|---|---|
| `description: str` | `TrainingInfo.model_card_description` |

All other fields unchanged: `gguf_quantizations`, `merged_save_method`, `save_gguf`, `save_lora`, `save_merged`, `publish_gguf`, `publish_lora`, `publish_merged`.

---

### All other config models

`DatasetInfo`, `BaseModelInfo`, `InferenceInfo`, `ChatTemplateInfo` — no schema changes.

---

## YAML File Changes

### `training_info/*.yaml` — fields added

```yaml
# NEW: name of the base_model_info YAML to load (without .yaml extension)
base_model: qwen2.5-0.5b-instruct

# NEW: name of the inference_info YAML to load (without .yaml extension)
inference: simple-classification

# NEW: name of the publishing_info YAML to load; set to null to skip save/publish
publishing: standard-publish   # or: null

# MOVED FROM publishing_info: human-readable text for the HuggingFace model card
model_card_description: >
  Binary text classifier fine-tuned with LoRA on a custom dataset.
```

### `publishing_info/*.yaml` — field removed

```yaml
# REMOVED: description field no longer appears here
```

---

## File Renames (migration)

| Old path | New path | Reason |
|---|---|---|
| `dataset_info/rpg_reddit_post_classification.yaml` | `dataset_info/rpg-reddit-post-classification.yaml` | kebab-case convention (FR-010) |

All other existing YAML files already use kebab-case and require no rename.

---

## `CommonPaths` changes

`ensure_all_dirs_exist()` updated to include `inference_info` directory creation (currently missing — it is a config dir, so auto-creation is required per Principle I).

No new directories introduced; no new constants needed.

---

## Relationship Diagram

```
TrainingInfo (always-new)
 ├── base_model ──────────────► BaseModelInfo (reusable)
 │                                  └── chat_template ──► ChatTemplateInfo (reusable)
 ├── inference ───────────────► InferenceInfo (reusable)
 ├── publishing ──────────────► PublishingInfo (reusable, optional)
 └── model_card_description    (stays in TrainingInfo)

DatasetInfo (always-new) — no references to other configs
```
