# Implementation Plan: Configurable Sequence Length Analysis

**Branch**: `005-configurable-sequence-lengths` | **Date**: 2026-03-07 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/005-configurable-sequence-lengths/spec.md`

## Summary

Move the hardcoded sequence lengths `[1024, 1536, 2048]` from the `analyze_dataset.py` command into an optional `sequence_lengths` field on `DatasetInfo`. When omitted, the default matches today's behavior. The field is validated at config load time (positive integers, non-empty). The analyze-dataset command iterates the configured list instead of hardcoded values. Documentation is added to `example.yaml` and the CLI help text.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: Pydantic v2, PyYAML, Typer
**Storage**: YAML config files (`dataset_info/`)
**Testing**: pytest
**Target Platform**: Linux (GPU workstation)
**Project Type**: CLI tool
**Performance Goals**: N/A (config parsing, no hot path)
**Constraints**: Backward compatible — existing configs without `sequence_lengths` must work unchanged
**Scale/Scope**: Single-user CLI tool; dataset configs are small YAML documents

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Configuration-First | PASS | Feature moves hardcoded values into YAML config — directly aligned. No new directories; field lives on existing `DatasetInfo`. |
| II. Protocol-Based Interfaces | PASS | No new cross-layer interfaces. Existing `LoggingProtocol` used for output. |
| III. Separation of Concerns | PASS | Config field added to `configuration/dataset_info.py`. Command reads it in `commands/analyze_dataset.py`. No helper changes needed. |
| IV. Observability | PASS | Coverage reporting via `LoggingProtocol` unchanged. |
| V. Simplicity & Scope | PASS | One new field on an existing model, one loop change in a command. Minimal change. |

## Project Structure

### Documentation (this feature)

```text
specs/005-configurable-sequence-lengths/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
src/classification_trainer/
├── configuration/
│   └── dataset_info.py         # MODIFIED: add sequence_lengths field with validation
├── commands/
│   └── analyze_dataset.py      # MODIFIED: iterate configured list instead of hardcoded values
└── console/
    └── main.py                 # MODIFIED: update analyze-dataset help text

dataset_info/
└── example.yaml                # MODIFIED: add documented sequence_lengths field

tests/
└── unit/
    └── test_dataset_helper.py  # MODIFIED: add tests for sequence_lengths validation
```

**Structure Decision**: Single-project CLI structure. All changes fit within existing files. No new files needed.

## Complexity Tracking

No constitution violations. No complexity justifications needed.
