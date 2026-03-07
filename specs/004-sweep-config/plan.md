# Implementation Plan: User-Configurable Sweep Parameters

**Branch**: `004-sweep-config` | **Date**: 2026-03-06 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/004-sweep-config/spec.md`

## Summary

Move the hardcoded sweep search space out of `SFTParameters.get_default_sweep_config()` and into an optional `sweep` block in the training config YAML. The sweep block uses opt-in semantics: only parameters explicitly listed are varied; unlisted parameters use their `sft_parameters` fixed values. Three parameter formats are supported (discrete list, continuous range, fixed scalar). The search method (random/bayes/grid) is configurable. Validation catches all invalid configs before any trials run. Trial progress and parameters are reported to the terminal.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: Pydantic v2, wandb (sweep API), Typer, PyYAML, Unsloth, TRL
**Storage**: YAML config files (`training_info/`)
**Testing**: pytest
**Target Platform**: Linux (GPU workstation)
**Project Type**: CLI tool
**Performance Goals**: N/A (config parsing, no hot path)
**Constraints**: Backward compatible — existing configs without `sweep` block must work unchanged
**Scale/Scope**: Single-user CLI tool; sweep configs are small YAML documents

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Configuration-First | PASS | Feature moves hardcoded sweep params into YAML config — directly aligned. No new directories needed; `sweep` block lives inside existing `training_info` YAML. |
| II. Protocol-Based Interfaces | PASS | Trial progress logging uses existing `LoggingProtocol`. No new cross-layer interfaces needed. |
| III. Separation of Concerns | PASS | New `SweepConfig` Pydantic model in `configuration/`. Sweep config building logic stays in `helpers/sweep_helper.py`. Orchestration stays in `commands/sweep.py`. |
| IV. Observability | PASS | FR-010/FR-011 add trial counter and parameter display via `LoggingProtocol`. |
| V. Simplicity & Scope | PASS | Feature is within classification fine-tuning scope. Adds one Pydantic model and one helper function — no new abstractions or patterns. |

## Project Structure

### Documentation (this feature)

```text
specs/004-sweep-config/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   └── sweep-block.md   # YAML contract for sweep block
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
src/classification_trainer/
├── configuration/
│   ├── sweep_config.py          # NEW: SweepConfig + SweepParameterSpec Pydantic models
│   ├── sft_parameters.py        # MODIFIED: preserve get_default_sweep_config as backward-compat fallback
│   └── training_info.py         # MODIFIED: add optional sweep field
├── helpers/
│   └── sweep_helper.py          # MODIFIED: build_sweep_config reads from SweepConfig
├── commands/
│   └── sweep.py                 # MODIFIED: trial counter, parameter display, validation
└── ...

training_info/
└── example.yaml                 # MODIFIED: add documented sweep block

tests/
└── unit/
    ├── test_sweep_config.py     # NEW: validation, parameter format parsing, edge cases
    └── test_sft_parameters.py   # MODIFIED: remove get_default_sweep_config tests if any
```

**Structure Decision**: Single-project CLI structure. All changes fit within existing `configuration/`, `helpers/`, and `commands/` directories. One new file (`sweep_config.py`) and one new test file (`test_sweep_config.py`).

## Complexity Tracking

No constitution violations. No complexity justifications needed.
