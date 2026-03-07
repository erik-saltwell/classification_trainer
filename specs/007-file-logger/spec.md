# Feature Specification: File-Based Logger

**Feature Branch**: `007-file-logger`
**Created**: 2026-03-07
**Status**: Draft
**Input**: User description: "Create a version of LoggingProtocol that logs to a file instead of the console. Do not log progress bars or spinners. When created, it gets passed a filename and should overwrite any existing content (not append)."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Log Training Output to a File (Priority: P1)

A practitioner wants to capture all log output from a training run (or any command) to a file for later review, archival, or automated processing. They configure the system to use the file logger instead of the console logger. All messages, warnings, errors, and tables are written to the specified file in plain text. Progress bars and spinners are silently ignored since they are only meaningful in an interactive terminal.

**Why this priority**: This is the entire feature — a file-based implementation of the logging interface. Without this, the feature doesn't exist.

**Independent Test**: Can be fully tested by creating a file logger with a target filename, calling each logging method (report_message, report_warning, report_error, report_exception, report_table_message, report_multicolumn_table, add_break), and verifying the file contains the expected text output. Then verify that status() and progress() context managers produce no file output.

**Acceptance Scenarios**:

1. **Given** a file logger initialized with a filename, **When** `report_message("hello")` is called, **Then** the file contains "hello" followed by a newline.
2. **Given** a file logger initialized with a filename, **When** `report_warning("caution")` is called, **Then** the file contains the warning text.
3. **Given** a file logger initialized with a filename, **When** `report_error("failure")` is called, **Then** the file contains the error text.
4. **Given** a file logger initialized with a filename, **When** `report_table_message({"key": "value"})` is called, **Then** the file contains the key-value data in a readable format.
5. **Given** a file logger initialized with a filename, **When** `report_multicolumn_table(headers, rows)` is called, **Then** the file contains the table data in a readable format.
6. **Given** a file logger initialized with a filename, **When** `status("working...")` context manager is entered and exited, **Then** no output is written to the file.
7. **Given** a file logger initialized with a filename, **When** `progress("processing", total=100)` context manager is entered and exited, **Then** no output is written to the file.

---

### User Story 2 - Overwrite Existing Log File on Creation (Priority: P2)

A practitioner runs a command multiple times with the same output file. Each run starts fresh — the log file is overwritten, not appended to. This ensures the file always contains output from the most recent run only.

**Why this priority**: Without overwrite behavior, repeated runs would accumulate output, making the log file confusing and increasingly large.

**Independent Test**: Can be fully tested by creating a file logger that writes to a file with existing content, then verifying the old content is gone and only the new output remains.

**Acceptance Scenarios**:

1. **Given** a file that already contains previous log output, **When** a new file logger is created with the same filename, **Then** the existing content is deleted and the file starts empty.
2. **Given** a file logger that has written output, **When** a new file logger is created with the same filename, **Then** only the output from the second logger is present in the file.

---

### User Story 3 - Strip Markup from Messages (Priority: P3)

The existing console logger uses Rich markup tags (e.g., `[blue]`, `[red]`, `[green]`) in messages. When writing to a file, these markup tags MUST be stripped so the file contains clean, readable plain text.

**Why this priority**: Without stripping, the log file would contain unreadable markup artifacts like `[blue]Loading...[/blue]` instead of `Loading...`.

**Independent Test**: Can be fully tested by calling `report_message("[blue]Loading...[/blue]")` and verifying the file contains `Loading...` without any markup tags.

**Acceptance Scenarios**:

1. **Given** a file logger, **When** `report_message("[blue]Loading model...[/blue]")` is called, **Then** the file contains `Loading model...` with no markup tags.
2. **Given** a file logger, **When** `report_error("[red]Error:[/red] something failed")` is called, **Then** the file contains `Error: something failed` with no markup tags.

---

### Edge Cases

- What happens when the file logger is given a path to a directory that does not exist? The system MUST create the parent directories automatically.
- What happens when `add_break(3)` is called? The file MUST contain 3 blank lines.
- What happens when `report_exception("context", exc)` is called? The file MUST contain both the context string and the exception details.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The file logger MUST implement the full `LoggingProtocol` interface — every method defined in the protocol MUST be implemented.
- **FR-002**: The file logger MUST accept a filename (or path) at creation time and write all output to that file.
- **FR-003**: When created, the file logger MUST overwrite any existing content in the target file (truncate, not append).
- **FR-004**: The `status()` and `progress()` context managers MUST produce no file output. They MUST return no-op handles that satisfy the `StatusHandle` and `ProgressTask` protocols.
- **FR-005**: All text written to the file MUST have Rich markup tags stripped, producing clean plain text.
- **FR-006**: `report_message`, `report_warning`, `report_error`, and `report_exception` MUST write their content to the file followed by a newline.
- **FR-007**: `report_table_message` MUST write key-value pairs in a readable text format (e.g., `key: value`, one per line).
- **FR-008**: `report_multicolumn_table` MUST write columnar data in a readable text format with headers.
- **FR-009**: `add_break` MUST write the specified number of blank lines to the file.
- **FR-010**: If the target file's parent directory does not exist, the file logger MUST create it.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The file logger can be used as a drop-in replacement for the console logger anywhere a `LoggingProtocol` is accepted — all methods work without errors.
- **SC-002**: A command run with the file logger produces a file containing all logged messages, warnings, errors, and tables — with zero markup artifacts.
- **SC-003**: Running the same command twice with the same output file produces a file containing only the second run's output.
- **SC-004**: Progress bars and spinners produce zero bytes of file output.

## Assumptions

- The file logger is a new class that lives alongside the existing console logger. It does not replace the console logger — both are available for different use cases.
- The file logger writes synchronously (no buffering beyond normal file I/O). This is acceptable since logging is not a performance-critical path.
- The file is opened at creation time and flushed after each write to ensure output is visible immediately (useful when tailing the log file).
- Rich markup stripping covers the `[tag]...[/tag]` pattern used throughout the codebase. Other Rich formatting (e.g., emoji shortcodes) is not used and is out of scope.
