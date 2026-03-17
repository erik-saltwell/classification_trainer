# Feature Specification: Modelfile Generation on Publish

**Feature Branch**: `011-generate-modelfile`
**Created**: 2026-03-17
**Status**: Draft
**Input**: User description:

## Clarifications

### Session 2026-03-17

- Q: What should the `FROM` line reference in the merged model Modelfile? → A: HuggingFace repo ID (e.g., `FROM username/model-merged`) — works for online Ollama import; local user edits one line.
- Q: If Modelfile generation fails, what happens to the overall save/publish operation? → A: Fail the entire format — same behavior as a model card or artifact save failure (clean up directory, raise exception).
- Q: Should `publish_model` regenerate the Modelfile if it is missing from the save directory? → A: Yes — regenerate if missing. Publish detects no Modelfile and generates it before uploading, so pre-existing saved artifacts get a Modelfile without requiring a re-train. "When we publish a model to hugging face with the publish command, we need to also generate a modelfile used for things like ollama and upload that as well. You need to do research on what should go into the model file and how to set parameters from our BaseModelInfo, TrainingInfo and InferenceInfo, and you need to figure out which model formats should get a model file. I want to make sure that if i download a model and modelfile, then load them into (for example) ollama, i get good high quality results similar to what i see when i test the model"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - GGUF Model Ready for Ollama Out of the Box (Priority: P1)

A practitioner trains and publishes a GGUF model. When they download the GGUF file and the accompanying Modelfile from HuggingFace, they can immediately load the model into Ollama and get inference behavior—temperature, stop sequences, system prompt, token limits—that exactly replicates what was observed during training evaluation.

**Why this priority**: GGUF is the native Ollama format. Without a correct Modelfile, users must manually reconstruct generation parameters and the chat template, which leads to degraded or inconsistent output quality compared to what was measured during training.

**Independent Test**: Can be fully tested by publishing a GGUF model, downloading the GGUF file and the generated Modelfile, importing both into Ollama, and running a sample classification prompt. The resulting output should match the label produced by the in-training evaluator.

**Acceptance Scenarios**:

1. **Given** a training run has produced a GGUF model saved locally, **When** the publish command is run, **Then** a `Modelfile` is generated in the GGUF save directory alongside the `.gguf` files before upload begins.
2. **Given** a published GGUF repository, **When** a user downloads the `.gguf` file and the `Modelfile` and runs `ollama create mymodel -f Modelfile`, **Then** Ollama accepts the file without errors and the resulting model uses the correct system prompt, stop sequences, temperature, and token generation limit.
3. **Given** a GGUF model with multiple quantization files in the same repository, **When** the Modelfile is generated, **Then** it references the default quantization file (the first/highest quality quantization in `gguf_quantizations`) via a relative path so it works when both files are in the same directory.

---

### User Story 2 - Merged Model Published with Modelfile (Priority: P2)

A practitioner publishes a merged (full-weight) HuggingFace Safetensors model. A Modelfile is included in the repository pointing to the uploaded merged model, so that someone using Ollama's import-from-HuggingFace feature gets correct generation behavior immediately.

**Why this priority**: Merged models are a common deployment format for users who want full-precision weights. Including a Modelfile enables consistent behavior without requiring the user to reverse-engineer the training configuration.

**Independent Test**: Publish a merged model, inspect the generated Modelfile in the save directory, verify it contains the correct chat template, system prompt, inference parameters, and a `FROM` line referencing the HuggingFace repo ID. Confirm Ollama can import using `ollama run <hf-repo-id>` with the embedded Modelfile.

**Acceptance Scenarios**:

1. **Given** a merged model is being saved and published, **When** model artifacts are written to disk, **Then** a `Modelfile` is included in the merged save directory.
2. **Given** the published merged repository on HuggingFace, **When** a user inspects the `Modelfile`, **Then** the `FROM` line references the HuggingFace repo ID (e.g., `FROM username/model-name-merged`) enabling direct Ollama online import without modification.

---

### User Story 3 - LoRA Adapter Excluded from Modelfile Generation (Priority: P3)

When a LoRA adapter is published, no Modelfile is generated for it. LoRA adapters require a compatible base model to be loaded first, and Ollama's support for separate LoRA adapters is limited and version-dependent; generating a Modelfile for a raw LoRA adapter would produce a non-functional result for most users.

**Why this priority**: Generating an incorrect Modelfile is worse than generating none. Clearly scoping the feature to GGUF and merged formats prevents confusion. This story ensures the feature boundary is correctly enforced.

**Independent Test**: Publish only a LoRA adapter (with GGUF and merged disabled). Verify no `Modelfile` is written to the LoRA save directory and no Modelfile-related log messages appear.

**Acceptance Scenarios**:

1. **Given** only `publish_lora` is enabled, **When** the publish command runs, **Then** no `Modelfile` is written to the LoRA save directory.
2. **Given** both `publish_lora` and `publish_gguf` are enabled, **When** the publish command runs, **Then** a `Modelfile` is written for the GGUF format only, and the LoRA directory contains no Modelfile.

---

### Edge Cases

- What happens when a chat template is not one of the known templates (ChatML, Llama, Mistral, Gemma, Phi)? The Modelfile TEMPLATE section should still be generated using the raw separator strings from `ChatTemplateInfo`, with no silent fallback or omission.
- What happens when `temperature` is 0.0 and `do_sample` is false? The Modelfile should set `temperature 0` and `top_p 1.0` to ensure deterministic greedy-equivalent decoding in Ollama.
- What happens when `repetition_penalty` is `null` in `InferenceInfo`? The `repeat_penalty` PARAMETER should be omitted from the Modelfile (use Ollama's default) rather than writing a null or zero value.
- What happens when the GGUF repo has multiple quantizations? The Modelfile references the first entry in `gguf_quantizations` as the default, since it is assumed to be the highest-quality option listed first.
- What happens if the Modelfile already exists in the save directory (e.g., from a prior run)? It is overwritten unconditionally during save, matching the behavior of the model card `README.md`.
- What happens if a user publishes artifacts saved before this feature existed (no Modelfile present)? The publish step detects the missing Modelfile, generates it from current config, and uploads it as part of the folder.
- What happens if `system_prompt` text contains special characters or newlines? The system prompt must be written as a quoted string or heredoc in the Modelfile format, preserving the exact content used during training.
- What happens if Modelfile generation raises an error? The entire format save fails and the save directory is cleaned up, identical to how model card or artifact save failures are handled. The publish step will then find no artifacts and report a failure for that format.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST generate a `Modelfile` file in the save directory for GGUF format models during the save phase (alongside the `.gguf` files and `README.md`).
- **FR-002**: The system MUST generate a `Modelfile` file in the save directory for merged format models during the save phase.
- **FR-003**: The system MUST NOT generate a `Modelfile` for LoRA adapter format models.
- **FR-004**: The generated Modelfile MUST include a `FROM` instruction that references the model. For GGUF, this is the relative filename of the primary GGUF file (e.g., `FROM mymodel-gguf-q8_0.gguf`). For merged, this is the HuggingFace repo ID (e.g., `FROM username/model-merged`) so the Modelfile works correctly when co-located in the published repository and accessed via Ollama's online import. Local users who download the folder may update the FROM line to a local path.
- **FR-005**: The generated Modelfile MUST include a `SYSTEM` instruction containing the verbatim system prompt from `TrainingInfo.system_prompt`.
- **FR-006**: The generated Modelfile MUST include a `TEMPLATE` instruction containing a Go-template-formatted chat template derived from the model's `ChatTemplateInfo` (instruction separator, response separator, and conditional system block).
- **FR-007**: The generated Modelfile MUST include `PARAMETER stop` instructions for each stop string defined in the model's `ChatTemplateInfo.stop_strings`.
- **FR-008**: The generated Modelfile MUST include `PARAMETER temperature` set from `InferenceInfo.temperature`.
- **FR-009**: The generated Modelfile MUST include `PARAMETER top_p` set from `InferenceInfo.top_p`.
- **FR-010**: The generated Modelfile MUST include `PARAMETER num_predict` set from `InferenceInfo.max_new_tokens`.
- **FR-011**: The generated Modelfile MUST include `PARAMETER num_ctx` set from `TrainingInfo.max_sequence_length` (the context window used during training).
- **FR-012**: If `InferenceInfo.repetition_penalty` is not null, the generated Modelfile MUST include `PARAMETER repeat_penalty` set to that value. If null, the parameter MUST be omitted.
- **FR-013**: The `Modelfile` must be uploaded to HuggingFace as part of the standard folder upload for its format (GGUF or merged), requiring no separate upload step.
- **FR-014**: Modelfile generation MUST be logged to the console alongside existing artifact save messages.
- **FR-015**: If Modelfile generation raises an error, the system MUST fail the entire format save (clean up the save directory and re-raise), identical to the existing failure behavior for model card or artifact save errors. No partial publish with a missing Modelfile is permitted.
- **FR-016**: During the publish step, if no `Modelfile` is present in a qualifying format's save directory (GGUF or merged), the system MUST generate it from the current `TrainingInfo` before uploading. This ensures models saved before this feature was introduced receive a Modelfile on their next publish without requiring a re-train.

### Key Entities

- **Modelfile**: A plain-text configuration file consumed by Ollama and compatible tools. Contains a `FROM` reference, `SYSTEM` prompt, `TEMPLATE` definition, and `PARAMETER` settings. One Modelfile is produced per qualifying format (GGUF, merged).
- **Chat Template**: The structured conversation format (e.g., ChatML, Llama, Mistral) defined in `ChatTemplateInfo`, including instruction/response separators and stop strings. Must be faithfully reproduced in the Modelfile `TEMPLATE` block.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: After publish completes, a valid `Modelfile` exists in the save directory for every enabled GGUF or merged format, with zero additional manual steps required.
- **SC-002**: A model loaded into Ollama using only the published `Modelfile` and model file produces classification outputs that match the post-training evaluation results for 100% of the label classes (positive/negative) on a representative sample.
- **SC-003**: All Modelfile parameters (`temperature`, `top_p`, `num_predict`, `num_ctx`, stop sequences, system prompt) are verifiably derived from the training configuration with no hardcoded defaults substituted for values that exist in config.
- **SC-004**: The LoRA format produces no Modelfile, and this is confirmed by automated test coverage.
- **SC-005**: Existing publish behavior (model card generation, HuggingFace upload) is unaffected when Modelfile generation is added—no regressions in existing published artifacts.

## Assumptions

- The primary consumer of the Modelfile is Ollama, but the file format is compatible with other tools (e.g., llama.cpp workflows) that accept the same syntax.
- The `gguf_quantizations` list is ordered from highest-quality to lowest; the first entry is used as the Modelfile's default `FROM` target.
- Merged models published to HuggingFace are in a format Ollama can import directly (Safetensors), so a `FROM <hf-repo-id>` reference is valid.
- No new configuration flags are required; Modelfile generation is automatic for qualifying formats.
- The system prompt stored in `TrainingInfo` is the same prompt used at inference time and should appear verbatim in the Modelfile `SYSTEM` block.
