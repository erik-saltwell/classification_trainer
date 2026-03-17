# Feature Specification: GGUF HuggingFace Metadata Files

**Feature Branch**: `012-gguf-hf-metadata`
**Created**: 2026-03-17
**Status**: Draft
**Input**: User description: "HuggingFace and Ollama documentation say that when a gguf repo is created, there should be separate template, params, and system files. This will allow using the model in ollama without having to download the model myself. Please research this, and then update the gguf publishing code to publish this IN ADDITION to the existing Modelfile publishing."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Run GGUF Model in Ollama Without Local Download (Priority: P1)

A practitioner publishes a fine-tuned GGUF model to HuggingFace. Another user (or the same practitioner on a different machine) runs `ollama run hf.co/<user>/<model>-gguf` and immediately gets correct inference behavior — the right system prompt, chat template, and sampling parameters — without needing to manually download any files or create a Modelfile.

**Why this priority**: This is the entire purpose of the feature. The three metadata files (`template`, `system`, `params`) are what enable Ollama's HuggingFace integration to configure the model automatically. Without them, the user gets Ollama's generic defaults, which produce different (likely worse) behavior than what was measured during training evaluation.

**Independent Test**: Publish a GGUF model, confirm `template`, `system`, and `params` files appear in the published HuggingFace repo alongside the `.gguf` files. Run `ollama run hf.co/<user>/<repo>` and verify the model uses the correct system prompt and produces classification labels matching the training evaluation.

**Acceptance Scenarios**:

1. **Given** a GGUF model is saved locally, **When** the save completes, **Then** `template`, `system`, and `params` files are written to the GGUF save directory alongside the `.gguf` files and `Modelfile`.
2. **Given** a GGUF HuggingFace repository, **When** Ollama imports it via `ollama run hf.co/user/repo`, **Then** Ollama uses the `template`, `system`, and `params` files to configure the model without any user intervention.
3. **Given** the `params` file in the repository, **When** a user inspects it, **Then** it is valid JSON containing `temperature`, `top_p`, `num_predict`, `num_ctx`, and `stop` values matching the training configuration.

---

### User Story 2 - Pre-Existing GGUF Saves Get Metadata on Next Publish (Priority: P2)

A practitioner has a GGUF model already saved locally (before this feature was built). When they run the publish command, the three metadata files are generated and uploaded automatically — no re-training or re-saving required.

**Why this priority**: Ensures backward compatibility. Users who already have GGUF saves on disk benefit from the feature without any extra steps.

**Independent Test**: Create a GGUF save directory with only the `.gguf`, `README.md`, and `Modelfile` present (simulating a pre-feature save). Run the publish command and confirm `template`, `system`, and `params` are present in the uploaded repository.

**Acceptance Scenarios**:

1. **Given** a GGUF save directory with no `template`, `system`, or `params` files, **When** the publish command runs, **Then** all three files are generated and uploaded as part of the folder upload.

---

### User Story 3 - Metadata Files Not Generated for Non-GGUF Formats (Priority: P3)

When publishing LoRA or merged format models, no `template`, `system`, or `params` files are generated. These files are specific to the HuggingFace GGUF + Ollama integration.

**Why this priority**: Correctness boundary. The merged format already uses a `Modelfile` with `FROM <hf-repo-id>`. Generating extra files there would be incorrect. LoRA adapters don't support this pattern at all.

**Independent Test**: Publish only a merged model (GGUF disabled). Confirm the merged save directory contains no `template`, `system`, or `params` files.

**Acceptance Scenarios**:

1. **Given** only merged format publishing is enabled, **When** save and publish complete, **Then** no `template`, `system`, or `params` files appear in the merged save directory.
2. **Given** both GGUF and merged are enabled, **When** save and publish complete, **Then** the three files appear only in the GGUF directory.

---

### Edge Cases

- What happens if the `template`, `system`, or `params` files already exist in the GGUF save directory? They are overwritten unconditionally, matching the behavior of the `Modelfile` and `README.md`.
- What happens if the GGUF publish directory has some but not all three metadata files (e.g., only `template` exists)? The publish step regenerates all three files if any is missing, ensuring the set is always complete.
- What happens if `stop_strings` is empty? The `params` file includes `"stop": []` — an empty JSON array, which is valid.
- What happens if the system prompt contains double quotes or special JSON characters? The `params` file only references stop strings (which could contain special characters); `json.dumps()` handles escaping correctly. The `system` file is plain text and requires no escaping.
- What happens if generation of any of the three files fails? The entire GGUF format save fails and the save directory is cleaned up, matching the existing failure behavior for `Modelfile` and model card generation.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: During the GGUF save phase, the system MUST write a `template` file to the GGUF save directory containing the Go template body derived from `ChatTemplateInfo`.
- **FR-002**: During the GGUF save phase, the system MUST write a `system` file to the GGUF save directory containing the verbatim system prompt from `TrainingInfo`.
- **FR-003**: During the GGUF save phase, the system MUST write a `params` file to the GGUF save directory containing a JSON object with `temperature`, `top_p`, `num_predict`, `num_ctx`, and `stop` derived from `InferenceInfo` and `TrainingInfo`.
- **FR-004**: If `InferenceInfo.repetition_penalty` is not null, the `params` file MUST include `"repeat_penalty"`. If null, it MUST be omitted.
- **FR-005**: The three metadata files MUST be uploaded to HuggingFace as part of the standard GGUF folder upload — no separate upload step.
- **FR-006**: During the publish step, if any of the three metadata files is absent from the GGUF save directory, the system MUST generate all three before uploading.
- **FR-007**: The system MUST NOT generate `template`, `system`, or `params` files for LoRA or merged format models.
- **FR-008**: The `template` file content MUST be identical to the Go template body used in the `Modelfile` TEMPLATE block (same derivation logic, no divergence).
- **FR-009**: Generation of the three metadata files MUST be logged to the console alongside other artifact save messages.
- **FR-010**: If any metadata file generation fails, the system MUST fail the entire GGUF format save (clean up directory and re-raise), identical to how `Modelfile` or model card failures are handled.

### Key Entities

- **`template` file**: Plain Go template text at GGUF repo root. Derived from `ChatTemplateInfo` (instruction separator, response separator, system separator, stop strings). Identical body to the Modelfile TEMPLATE block.
- **`system` file**: Plain UTF-8 text at GGUF repo root. Contains verbatim `TrainingInfo.system_prompt`.
- **`params` file**: JSON object at GGUF repo root. Contains sampling parameters matching `InferenceInfo` and `TrainingInfo.max_sequence_length`.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: After a GGUF save completes, all three files (`template`, `system`, `params`) are present in the save directory with zero additional manual steps.
- **SC-002**: `ollama run hf.co/<user>/<model>-gguf` produces classification outputs matching the post-training evaluation results without any user-supplied Modelfile or parameter configuration.
- **SC-003**: The `params` file is valid JSON and all parameter values are verifiably derived from the training configuration — no hardcoded defaults substituted for values present in config.
- **SC-004**: The `template` file body is byte-for-byte identical to the TEMPLATE body embedded in the co-located `Modelfile` — no divergence between the two.
- **SC-005**: Non-GGUF formats (LoRA, merged) produce no `template`, `system`, or `params` files, confirmed by automated test coverage.
- **SC-006**: Pre-existing GGUF saves (lacking the three files) receive them automatically on the next publish run with no re-training required.

## Assumptions

- All three files must be present at the repository root (not in a subdirectory) for Ollama's HuggingFace integration to discover them.
- The three files are generated together atomically — the system never writes a partial set.
- Ollama falls back to GGUF built-in metadata if the files are absent; the files are an override/enhancement, not a requirement for basic operation.
- The existing `Modelfile` continues to be generated alongside these files — this feature is additive.
