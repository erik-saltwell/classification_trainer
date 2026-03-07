# Implementation Plan: File-Based Logger

**Branch**: `007-file-logger` | **Date**: 2026-03-07 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/007-file-logger/spec.md`

## Summary

Create a `FileLogger` class that implements `LoggingProtocol`, writing all text output to a file. The implementation reuses Rich's `Console` by constructing it with `file=` pointing to the output file and `no_color=True` / `force_terminal=False` to produce clean plain text. This means table formatting, markup stripping, and traceback rendering all use the same Rich code paths as the console logger — ensuring the file output matches the console output as closely as possible (minus colors and interactive elements). Progress bars and spinners are no-ops (reusing the existing `_NullStatus` and `_NullProgress` classes from the protocol module). The file is truncated on creation.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: Rich (Console with file output, Table, Traceback — reused from existing dependency)
**Storage**: Plain text file output
**Testing**: pytest
**Target Platform**: Linux (GPU workstation)
**Project Type**: CLI tool
**Performance Goals**: N/A (logging is not performance-critical)
**Constraints**: Must implement all methods of LoggingProtocol
**Scale/Scope**: Single new class + test file

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Configuration-First | PASS | No config changes needed. The logger is instantiated programmatically with a filename. |
| II. Protocol-Based Interfaces | PASS | Implements `LoggingProtocol` — the existing protocol. No new protocols needed. Follows the pattern established by `RichConsoleLogger` and `NullLogger`. |
| III. Separation of Concerns | PASS | New logger lives in `console/` alongside `RichConsoleLogger`. No domain logic. |
| IV. Observability | PASS | This IS an observability feature — adds file-based output as an alternative to console. |
| V. Simplicity & Scope | PASS | One new class, reuses existing no-op handles. No new patterns or abstractions. |

## Project Structure

### Documentation (this feature)

```text
specs/007-file-logger/
├── plan.md              # This file
├── research.md          # Phase 0 output
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
src/classification_trainer/
├── console/
│   ├── file_logging_protocol.py  # NEW: FileLogger class
│   └── rich_logging_protocol.py  # EXISTING: for reference pattern
├── protocols/
│   └── logging_protocol.py       # EXISTING: _NullStatus, _NullProgress reused

tests/
└── unit/
    └── test_file_logger.py       # NEW: unit tests for FileLogger
```

**Structure Decision**: Single-project CLI structure. One new file in `console/` (matching where `RichConsoleLogger` lives) and one new test file.

## Complexity Tracking

No constitution violations. No complexity justifications needed.
