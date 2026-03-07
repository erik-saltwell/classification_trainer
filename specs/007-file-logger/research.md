# Research: File-Based Logger

**Feature**: 007-file-logger | **Date**: 2026-03-07

## R1: Rich Console as File Writer

**Decision**: Construct a `rich.console.Console` with `file=<file_handle>`, `no_color=True`, and `force_terminal=False`. Use this console for all output — `console.print()` handles markup stripping, table rendering, and traceback formatting automatically.

**Rationale**: The `RichConsoleLogger` already uses `Console.print()` for everything including `Table` objects and `Traceback`. By constructing a Console that targets a file instead of stdout, we get identical formatting with colors removed. This means:
- Markup stripping is handled by Rich itself (no custom regex needed)
- Table formatting uses the same `Table` + `box.SQUARE` style as the console logger
- Exception tracebacks use the same `Traceback.from_exception()` rendering
- No risk of format divergence between console and file output

**Alternatives considered**:
- Custom regex for markup stripping + manual table formatting: More code, risk of divergence from console output, duplicates logic already in Rich.
- `rich.text.Text.from_markup(msg).plain`: Would strip markup but wouldn't handle tables or tracebacks.

## R2: File Handle Management

**Decision**: Open the file in write mode (`"w"`) at creation time, pass the handle to `Console(file=...)`. Rich's Console handles flushing internally.

**Rationale**: Opening in `"w"` mode truncates the file automatically (FR-003). The Console takes ownership of writing to the file handle. `Console.print()` flushes by default when `force_terminal=False`.

**Alternatives considered**:
- Open/close per write: Higher overhead, incompatible with Console's file handle model.
- Buffered writes with flush at end: Risk losing output if the process crashes.

## R3: No-Op Progress/Status Handles

**Decision**: Reuse `_NullStatus` and `_NullProgress` from `protocols/logging_protocol.py`.

**Rationale**: These classes already exist and implement the required protocols. No reason to duplicate them. They are private by convention (underscore prefix) but are in the same package and intended for reuse by logger implementations — `NullLogger` already uses them.

## R4: Table Formatting

**Decision**: Reuse the exact same `Table` construction from `RichConsoleLogger` — same `show_header`, `show_lines`, `box.SQUARE` settings. The Rich Console targeting a file will render the table as ASCII art automatically.

**Rationale**: Maximum consistency between file and console output. The user sees the same table structure in the file as they would on screen, just without colors.

## R5: Console Width

**Decision**: Set `Console(width=120)` to ensure consistent table formatting regardless of the terminal width at the time of execution.

**Rationale**: When Console targets a file, there is no terminal to detect width from. Without an explicit width, Rich defaults to 80 columns, which may truncate wide tables. 120 columns provides comfortable room for most table layouts while keeping lines readable.
