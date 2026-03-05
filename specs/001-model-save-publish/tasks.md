---

description: "Task list for Model Save and Publish feature"
---

# Tasks: Model Save and Publish

**Input**: Design documents from `specs/001-model-save-publish/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/cli-contracts.md ✅

**Organization**: Tasks are grouped by user story to enable independent implementation
and testing of each story. No test tasks are included (not requested in spec).

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- All file paths are relative to the repository root

---

## Phase 1: Setup

**Purpose**: Register the new optional dependency before any code is written.

- [x] T001 ~~Add `autoawq` as an optional dependency~~ (removed — AWQ format not included)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Infrastructure that MUST exist before any user story implementation can begin.
All three tasks are sequential — each depends on the previous.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [x] T002 Extend `src/classification_trainer/utils/common_paths.py`: add class-level constants `PUBLISHING_INFO_DIR = Path("publishing_info")` and `OUTPUT_MODELS_DIR = Path("output_models")`; add `publishing_info` property returning `CommonPaths.PUBLISHING_INFO_DIR`; add `output_models` property returning `CommonPaths.OUTPUT_MODELS_DIR` (does NOT auto-create); add `self.publishing_info.mkdir(parents=True, exist_ok=True)` inside `ensure_all_dirs_exist()`
- [x] T003 Create `src/classification_trainer/configuration/publishing_info.py`: define `SaveFormat(StrEnum)` with values `GGUF = "gguf"`, `LORA = "lora"`, `MERGED = "merged"`; define `PublishingInfo(BaseModel)` with `description: str`, `gguf_quantizations: list[str] = ["q8_0"]`, `merged_save_method: str = "merged_16bit"`, and six boolean flags `save_gguf/lora/merged` and `publish_gguf/lora/merged` (all default `False`), plus helper properties `any_save_enabled` and `any_publish_enabled`; define `load_publishing_info(name: str) -> PublishingInfo`
- [x] T004 Update `src/classification_trainer/configuration/__init__.py`: add imports for `PublishingInfo`, `SaveFormat`, `load_publishing_info` from `.publishing_info`; add all three to `__all__`

**Checkpoint**: `PublishingInfo` loads from YAML, validates format names, and is importable from `classification_trainer.configuration`.

---

## Phase 3: User Story 1 — Auto-Save After Training (Priority: P1) 🎯 MVP

**Goal**: After training completes, the best model is saved in configured formats with a
generated model card alongside each artifact.

**Independent Test**: Run `python -m classification_trainer train --dataset <ds> --base-model <bm> --training-info <ti> --inference-info <inf> --publishing-info <pi>` where the publishing config enables at least one format. Confirm `output_models/<model-name>/<format>/` exists, is non-empty, and contains `README.md` with the description text and pre/post metrics.

### Implementation for User Story 1

- [x] T005 [US1] Implement format-specific private save functions in `src/classification_trainer/helpers/publishing_helper.py`: `_save_lora(model, tokenizer, save_dir: Path) -> None`; `_save_gguf_quant(model, tokenizer, save_dir: Path, quantization_method: str, output_filename: str) -> None` (saves to temp dir, renames to output_filename in save_dir; all quants share one `gguf/` dir with files named `<model>-gguf-<quant>.gguf`); `_save_merged(model, tokenizer, save_dir: Path, save_method: str) -> None`
- [x] T006 [US1] Implement `generate_model_card(save_dir: Path, format_slug: str, training_info: TrainingInfo, dataset_info: DatasetInfo, base_model_info: BaseModelInfo, publishing_info: PublishingInfo, pre_metrics: list[MetricResult], post_metrics: list[MetricResult]) -> None` in `src/classification_trainer/helpers/publishing_helper.py`: builds a Markdown string with sections — Title (`{training_info.model_name} ({format_slug})`), Description (publishing_info.description), Model Details (base model name, LoRA rank, quantization), Dataset (dataset_info.huggingface_name, splits, positive class), Training Configuration (epochs/steps, batch size, learning rate, max_sequence_length), Pre-Training Metrics (omitted with note if pre_metrics is empty), Post-Training Metrics, Usage (format-specific code snippet); writes to `save_dir / "README.md"` using `huggingface_hub.ModelCard(content).save(str(save_dir / "README.md"))`
- [x] T007 [US1] Implement `save_model(model, tokenizer, training_info: TrainingInfo, dataset_info: DatasetInfo, base_model_info: BaseModelInfo, publishing_info: PublishingInfo, pre_metrics: list[MetricResult], post_metrics: list[MetricResult], logger: LoggingProtocol) -> None` in `src/classification_trainer/helpers/publishing_helper.py`: skip entirely if `not publishing_info.any_save_enabled`; for each enabled format (lora/gguf/merged), compute `save_dir = CommonPaths.get().output_models / training_info.model_name / slug`; for gguf, loop over `gguf_quantizations` calling `_save_gguf_quant` with filename `<model>-gguf-<quant>.gguf` inside the shared `gguf/` dir, then generate one model card covering all quants; for other formats call their save function then generate a model card; on failure clean up partial dir with `shutil.rmtree(save_dir, ignore_errors=True)` and re-raise
- [x] T008 [US1] Update `TrainCommand` dataclass in `src/classification_trainer/commands/train.py`: add field `publishing_info: PublishingInfo | None = None`; at the end of `execute()` after `self.report_results(...)`, add a call to `publishing_helper.save_model(model, tokenizer, self.training_info, self.dataset_info, self.base_model_info, self.publishing_info, pre_run_results, post_run_results, logger)` guarded by `if self.publishing_info is not None`; add `from classification_trainer.helpers import publishing_helper` import
- [x] T009 [US1] Update the `train` CLI command in `src/classification_trainer/console/main.py`: add `publishing_info: Annotated[str | None, typer.Option("--publishing-info", help="Publishing info yaml name (no extension)")] = None` parameter; load it with `load_config_or_exit(load_publishing_info, publishing_info, "publishing info", console)` when not None; pass the result (or None) as `publishing_info=` to `TrainCommand`; add `load_publishing_info` to the configuration imports

**Checkpoint**: User Story 1 fully functional — train with a publishing config saves all enabled formats locally with model cards.

---

## Phase 4: User Story 2 — Publish Saved Model to HuggingFace (Priority: P2)

**Goal**: A separate `publish` command uploads locally saved artifacts to HuggingFace Hub,
one repository per format, using the saved model card.

**Independent Test**: With artifacts already on disk from US1, run `python -m classification_trainer publish --training-info <ti> --publishing-info <pi>`. Verify a HuggingFace repository named `<username>/<model-name>-<format>` exists for each publish format and contains the model files and matching `README.md`.

### Implementation for User Story 2

- [x] T010 [US2] Implement `publish_model(training_info: TrainingInfo, publishing_info: PublishingInfo, logger: LoggingProtocol) -> None` in `src/classification_trainer/helpers/publishing_helper.py`: iterate enabled publish formats (lora/gguf/merged) — for each: compute `save_dir = CommonPaths.get().output_models / training_info.model_name / slug`; fail clearly if save_dir or `README.md` missing; compute `repo_id = f"{training_info.hugging_face_user_name}/{training_info.model_name}-{slug}"`; call `HfApi().create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=True)` then `HfApi().upload_folder(...)`; log progress per format; collect all failures and report at end. For gguf, uploads the entire `gguf/` directory (all quant files + README.md) to the single `<model>-gguf` repo.
- [x] T011 [US2] Create `src/classification_trainer/commands/publish.py`: define `PublishCommand` as an `@dataclass` implementing `CommmandProtocol`; fields: `training_info: TrainingInfo`, `publishing_info: PublishingInfo`; `execute(self, logger: LoggingProtocol) -> None` calls `publishing_helper.publish_model(self.training_info, self.publishing_info, logger)` with appropriate log messages before and after
- [x] T012 [US2] Register a `publish` CLI command in `src/classification_trainer/console/main.py`: add `@app.command("publish")` function with required `--training-info` and `--publishing-info` options; load both configs with `load_config_or_exit`; instantiate and execute `PublishCommand`; add `PublishCommand` to imports

**Checkpoint**: User Stories 1 and 2 are both independently functional.

---

## Phase 5: User Story 3 — Config Validation (Priority: P3)

**Goal**: Publishing configs fail fast with clear messages on invalid input, and an example
YAML provides a working template.

**Independent Test**: Create a publishing config YAML with an invalid format name (e.g., `"tensorrt"`). Run any command that loads it. Confirm it exits immediately with a validation error naming the bad field before any model loading occurs.

### Implementation for User Story 3

- [x] T013 [US3] Verify `PublishingInfo` in `src/classification_trainer/configuration/publishing_info.py` produces clear Pydantic validation errors for unknown fields (already handled by Pydantic's `extra = "forbid"` or default strict mode); add `model_config = ConfigDict(frozen=True, extra="forbid")` so unknown YAML keys are rejected with a clear field name in the error message
- [x] T014 [P] [US3] Create `publishing_info/example.yaml` with all fields populated and inline comments explaining each flag, the `gguf_quantizations` list, and when to enable each format

**Checkpoint**: All three user stories are independently functional and testable.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Stability and observability improvements across US1 and US2.

- [x] T015 Add GPU memory flush between format saves in `save_model()` in `src/classification_trainer/helpers/publishing_helper.py`: import `from classification_trainer.utils.flush_gpu_memory import flush_gpu_memory`; call `flush_gpu_memory()` after each format's save completes (inside the format loop, after card generation) to reduce VRAM pressure on sequential GGUF + merged saves
- [x] T016 Update `src/classification_trainer/console/console_validation.py` (or `main.py`) to ensure authentication errors from `huggingface_hub` during `publish` are caught and surfaced as a user-readable message rather than a raw `HfHubHTTPError` traceback

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Phase 1 completion — BLOCKS all user stories
- **User Story 1 (Phase 3)**: Depends on Phase 2 completion
- **User Story 2 (Phase 4)**: Depends on Phase 3 completion (publish requires saved artifacts)
- **User Story 3 (Phase 5)**: Depends on Phase 2 completion; T013 enhances the Foundational config model
- **Polish (Phase 6)**: Depends on Phases 3 and 4 completion

### Within Each Phase

- **Phase 2**: T002 → T003 → T004 (strictly sequential; each builds on the previous)
- **Phase 3**: T005 → T006 → T007 → T008 → T009 (sequential within publishing_helper.py; T008 and T009 are different files but T009 depends on T008's interface)
- **Phase 4**: T010 → T011 → T012 (sequential; T011 calls T010's function, T012 registers T011's class)
- **Phase 5**: T013 and T014 [P] can run in parallel (different files, no dependencies)
- **Phase 6**: T015 → T016 (both touch different files, but run sequentially for clarity)

### Parallel Opportunities

```bash
# Phase 5 — US3 tasks can run in parallel:
Task T013: Add validators to publishing_info.py
Task T014: Create publishing_info/example.yaml  ← independent file
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001)
2. Complete Phase 2: Foundational (T002–T004) — CRITICAL, blocks everything
3. Complete Phase 3: User Story 1 (T005–T009)
4. **STOP and VALIDATE**: Run `train --publishing-info example` and confirm artifacts appear under `output_models/`
5. Inspect `README.md` for correct description and metrics

### Incremental Delivery

1. Setup + Foundational → Config model ready
2. User Story 1 → Local save + model cards working → **Local deployment possible**
3. User Story 2 → HuggingFace publish working → **Full end-to-end**
4. User Story 3 → Validation hardened → **Production ready**
5. Polish → Stability + UX improvements

---

## Notes

- `[P]` tasks in the same phase can run in parallel (different files)
- Story labels map to spec.md user stories: US1=P1, US2=P2, US3=P3
- All GGUF quantizations share a single `gguf/` directory and a single HF repo; files are named `<model>-gguf-<quant>.gguf`
- GPU memory flush (T015) is important for sequential GGUF + merged saves on small-VRAM machines
- The `output_models/` directory is NOT auto-created by `CommonPaths.ensure_all_dirs_exist()` — it is created on-demand by `save_model()` (per constitution Principle I amendment)
- The `publishing_info/` directory IS auto-created by `CommonPaths.ensure_all_dirs_exist()` (it's a config directory)
