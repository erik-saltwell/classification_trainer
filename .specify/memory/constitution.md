<!--
SYNC IMPACT REPORT
==================
Version change: 1.0.0 → 1.1.0
Modified principles:
  - I. Configuration-First: expanded to include CommonPaths as mandatory path registry
Added sections: none
Removed sections: none
Templates requiring updates:
  - .specify/templates/plan-template.md ✅ Constitution Check gates still align
  - .specify/templates/spec-template.md ✅ No conflicts
  - .specify/templates/tasks-template.md ✅ No conflicts
  - .specify/templates/constitution-template.md ✅ Source template unchanged
Follow-up TODOs:
  - TODO(RATIFICATION_DATE): Exact project start date unknown; marked as project inception estimate.
  - plan.md for 001-model-save-publish should be updated to reflect CommonPaths requirement
    explicitly in Phase 1 design notes.
-->

# Classification Trainer Constitution

## Core Principles

### I. Configuration-First

All training hyperparameters, dataset bindings, base model selection, and inference settings
MUST be declared in YAML files loaded and validated by Pydantic models
(`TrainingInfo`, `DatasetInfo`, `BaseModelInfo`, `InferenceInfo`).
No training-relevant parameters may be hardcoded in command or helper modules.
YAML files live under `training_info/`, `dataset_info/`, and `base_model_info/` directories.

All project-known directory paths MUST be declared as constants and properties in
`utils/common_paths.py` (`CommonPaths`). No module outside `common_paths.py` may
hard-code or construct a project directory path independently. Any new feature that
introduces a new directory (e.g., `publishing_info/`, `outputs/`) MUST add the
corresponding constant, property, and — if it is a config directory that should always
exist — a call in `ensure_all_dirs_exist()`. Runtime-only output directories MUST be
exposed as a property but MUST NOT be auto-created by `ensure_all_dirs_exist()`.

**Rationale**: Reproducible experiments require that every run be fully described by its
config files and that all path references resolve from a single authoritative registry.
Scattered path strings create silent breakage when directories are reorganised and
undermine auditability.

### II. Protocol-Based Interfaces

Cross-layer communication MUST use Python `Protocol` types (not concrete classes) at
system boundaries: `CommmandProtocol` (execute), `LoggingProtocol`, and
`MetricsReportingProtocol`. Helpers MUST accept these protocols, not concrete implementations.
New reporters or loggers MUST implement the relevant protocol rather than subclass a base class.

**Rationale**: Protocols enable independent testing of commands and helpers by substituting
lightweight fakes without coupling to Rich console, WandB, or any specific I/O backend.

### III. Separation of Concerns (NON-NEGOTIABLE)

The codebase MUST maintain three distinct layers with no cross-contamination:

- **Commands** (`commands/`): Orchestration only — load configs, call helpers in sequence,
  report results. Zero domain logic.
- **Helpers** (`helpers/`): Domain logic only — dataset prep, training, inference, evaluation,
  reporting. Zero orchestration or config loading.
- **Configuration** (`configuration/`): Pydantic models + YAML loaders only. No side effects.

A helper MUST NOT call another helper that belongs to a different domain without going through
a command. A command MUST NOT contain algorithmic logic that belongs in a helper.

**Rationale**: This layering keeps helpers independently testable and commands readable as
plain orchestration scripts.

### IV. Observability

Every training run MUST capture and report classification metrics both before and after
training (pre/post comparison). WandB integration is first-class: when `wandb_config` is
present in `TrainingInfo`, all metrics MUST be logged with monotonically increasing step
values. All runs MUST use an explicit `seed` value in `TrainingInfo` to ensure reproducibility.
Log messages MUST use the `LoggingProtocol`; direct `print()` calls are prohibited.

**Rationale**: Fine-tuning outcomes can only be validated by comparing pre- and post-training
performance. Reproducible seeds and W&B tracking allow experiments to be audited and repeated.

### V. Simplicity & Scope

This project's scope is classification fine-tuning of open-source LLMs via LoRA/QLoRA using
Unsloth. Features MUST NOT be added that fall outside this scope. New abstractions require
justification — three similar lines of code are preferable to a premature abstraction. YAGNI
applies. The Composite pattern (`CompositeMetricsReporter`) is the approved extension point
for adding reporters; new global patterns require constitution amendment.

**Rationale**: A focused, simple codebase is easier to maintain, debug, and reason about
than a generalized framework. Scope creep degrades reliability.

## Technology Stack

- **Language**: Python 3.11+
- **Fine-tuning**: Unsloth `FastLanguageModel`, TRL `SFTTrainer`, HuggingFace Transformers
- **Config validation**: Pydantic v2 (YAML → model)
- **Experiment tracking**: Weights & Biases (WandB) — optional, controlled by `wandb_config`
- **CLI**: Typer with Rich console output
- **Dataset source**: HuggingFace Hub (`datasets` library)
- **Chat templates**: ChatML (response separator: `<|im_start|>assistant\n`)
- **Quantization**: 4-bit QLoRA via `load_in_4bit=True` (default); LoftQ optional
- **Path registry**: `utils/common_paths.py` (`CommonPaths`) — single source of truth for
  all project directory paths

New dependencies MUST be justified against an existing dependency's capabilities before
introduction. GPU memory efficiency MUST be considered for any change touching training
or inference code.

## Development Workflow

- New features begin with a YAML config change or a new config model — not with code.
- Any new feature directory MUST be registered in `CommonPaths` before any code references it.
  Config directories are added to `ensure_all_dirs_exist()`; runtime output directories
  are exposed as properties only.
- Protocol definitions (`protocols/`) MUST be updated before implementing code that
  depends on the new interface contract.
- Known training pitfalls (e.g., `eval_loss = NaN` when `train_on_outputs_only=True`
  truncates response separators) MUST be documented in `memory/previous_bugs.md`.
- All PRs MUST verify constitution compliance against Principles I–V before merge.
- Complexity violations (e.g., a helper that also orchestrates) MUST be documented in
  the plan's Complexity Tracking table with justification.
- Commit messages MUST be imperative-mood, lowercase, ≤72 characters.

## Governance

This constitution supersedes all other coding conventions for this project.
Amendments require: (1) a documented rationale, (2) version bump per semantic versioning
rules (MAJOR: principle removal/redefinition; MINOR: new principle/section or materially
expanded mandatory guidance; PATCH: clarification/wording), (3) update of this file and
propagation to dependent templates.
All PRs that touch `helpers/`, `commands/`, or `configuration/` MUST include a
constitution compliance check. Use `.specify/memory/MEMORY.md` and
`.claude/projects/…/memory/MEMORY.md` for runtime development guidance.

**Version**: 1.1.0 | **Ratified**: TODO(RATIFICATION_DATE): project inception date unknown | **Last Amended**: 2026-03-05
