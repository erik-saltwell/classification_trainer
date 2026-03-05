<!--
SYNC IMPACT REPORT
==================
Version change: (none) → 1.0.0 (initial ratification from blank template)
Modified principles: N/A (first version)
Added sections:
  - Core Principles (5 principles derived from codebase analysis)
  - Technology Stack
  - Development Workflow
  - Governance
Removed sections: N/A
Templates requiring updates:
  - .specify/templates/plan-template.md ✅ Constitution Check gates align with principles below
  - .specify/templates/spec-template.md ✅ No conflicts; acceptance criteria pattern matches FR/SC format
  - .specify/templates/tasks-template.md ✅ Phase/parallel structure consistent with principle III
  - .specify/templates/constitution-template.md ✅ Source template unchanged (this is the filled instance)
Follow-up TODOs:
  - TODO(RATIFICATION_DATE): Exact project start date unknown; marked as project inception estimate.
-->

# Classification Trainer Constitution

## Core Principles

### I. Configuration-First

All training hyperparameters, dataset bindings, base model selection, and inference settings
MUST be declared in YAML files loaded and validated by Pydantic models
(`TrainingInfo`, `DatasetInfo`, `BaseModelInfo`, `InferenceInfo`).
No training-relevant parameters may be hardcoded in command or helper modules.
YAML files live under `training_info/`, `dataset_info/`, and `base_model_info/` directories.

**Rationale**: Reproducible experiments require that every run be fully described by its
config files. Hardcoded values create invisible experiment variance and undermine auditability.

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

New dependencies MUST be justified against an existing dependency's capabilities before
introduction. GPU memory efficiency MUST be considered for any change touching training
or inference code.

## Development Workflow

- New features begin with a YAML config change or a new config model — not with code.
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
rules (MAJOR: principle removal/redefinition; MINOR: new principle/section; PATCH:
clarification/wording), (3) update of this file and propagation to dependent templates.
All PRs that touch `helpers/`, `commands/`, or `configuration/` MUST include a
constitution compliance check. Use `.specify/memory/MEMORY.md` and
`.claude/projects/…/memory/MEMORY.md` for runtime development guidance.

**Version**: 1.0.0 | **Ratified**: TODO(RATIFICATION_DATE): project inception date unknown | **Last Amended**: 2026-03-05
