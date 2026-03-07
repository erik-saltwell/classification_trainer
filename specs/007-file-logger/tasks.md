# Tasks: File-Based Logger

**Input**: Design documents from `/specs/007-file-logger/`
**Prerequisites**: plan.md (required), spec.md (required), research.md

**Tests**: Tests are included — the logger must satisfy all `LoggingProtocol` methods and the spec has detailed acceptance scenarios.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Phase 1: Foundational (Blocking Prerequisites)

**Purpose**: Create the `FileLogger` class with core structure — MUST be done before any user story work

- [x] T001 Create `FileLogger` class in `src/classification_trainer/console/file_logging_protocol.py` that implements `LoggingProtocol`. Constructor accepts a filename (str or Path), creates parent directories if needed, opens the file in write mode (`"w"` — truncates), and constructs a `rich.console.Console` with `file=<handle>`, `no_color=True`, `force_terminal=False`, `width=120`. Implement `status()` and `progress()` context managers using `_NullStatus` and `_NullProgress` from `protocols/logging_protocol.py`.
- [x] T002 [P] Create test file `tests/unit/test_file_logger.py` with imports for `FileLogger`, `pytest`, and `tmp_path` fixture.

**Checkpoint**: `FileLogger` class exists with constructor and no-op progress/status. Can be instantiated.

---

## Phase 2: User Story 1 - Log Training Output to a File (Priority: P1)

**Goal**: All text logging methods write to the file using Rich Console. Progress/status produce no output.

**Independent Test**: Create a FileLogger, call each method, read the file, verify expected content.

### Implementation for User Story 1

- [x] T003 [US1] Implement `report_message`, `report_warning`, `report_error`, and `report_exception` in `src/classification_trainer/console/file_logging_protocol.py`. Use `self._console.print()` for messages. For warnings, prefix with `WARNING`. For errors, prefix with `ERROR`. For exceptions, prefix with `EXCEPTION {context}` then print `Traceback.from_exception()` — matching the `RichConsoleLogger` pattern.
- [x] T004 [US1] Implement `report_table_message` and `report_multicolumn_table` in `src/classification_trainer/console/file_logging_protocol.py`. Construct `Table` objects with `show_header=True`, `show_lines=True`, `box=box.SQUARE` — identical to `RichConsoleLogger`. Print via `self._console.print(table)`.
- [x] T005 [US1] Implement `add_break` in `src/classification_trainer/console/file_logging_protocol.py` — print empty strings for the specified count, matching the `RichConsoleLogger` pattern.
- [x] T006 [US1] Write unit tests in `tests/unit/test_file_logger.py`: (a) `report_message("hello")` writes "hello" to file, (b) `report_warning("caution")` writes "WARNING caution", (c) `report_error("failure")` writes "ERROR failure", (d) `report_exception("ctx", ValueError("oops"))` writes both context and exception details, (e) `report_table_message({"key": "val"})` writes table with key/value columns, (f) `report_multicolumn_table(["a","b"], [["1","2"]])` writes columnar data, (g) `add_break(3)` writes 3 blank lines, (h) `status()` context manager produces no file output, (i) `progress()` context manager produces no file output.

**Checkpoint**: All logging methods work. File contains expected text. Progress/status produce zero output.

---

## Phase 3: User Story 2 - Overwrite on Creation (Priority: P2)

**Goal**: Creating a new FileLogger with the same filename overwrites existing content.

**Independent Test**: Write to a file, create a new FileLogger for same file, verify old content is gone.

### Implementation for User Story 2

- [x] T007 [US2] Write unit tests in `tests/unit/test_file_logger.py`: (a) create a file with existing content, create a FileLogger for that file, verify the existing content is gone, (b) write with one FileLogger, create a second FileLogger for the same file, write new content, verify only the second logger's content is present.

**Checkpoint**: Overwrite behavior is already implemented by the `"w"` open mode in T001. This phase just adds test coverage.

---

## Phase 4: User Story 3 - Strip Rich Markup (Priority: P3)

**Goal**: Rich markup tags are stripped in the file output, producing clean plain text.

**Independent Test**: Call `report_message("[blue]Loading...[/blue]")` and verify file contains `Loading...`.

### Implementation for User Story 3

- [x] T008 [US3] Write unit tests in `tests/unit/test_file_logger.py`: (a) `report_message("[blue]Loading model...[/blue]")` produces `Loading model...` without markup, (b) `report_error("[red]Error:[/red] something failed")` produces `ERROR Error: something failed` without markup.

**Checkpoint**: Markup stripping is already handled by `Console(no_color=True)` from T001. This phase just adds test coverage.

---

## Phase 5: Polish & Cross-Cutting Concerns

**Purpose**: Edge case tests and final validation

- [x] T009 Write edge case tests in `tests/unit/test_file_logger.py`: (a) FileLogger creates parent directories if they don't exist (`tmp_path / "sub" / "dir" / "log.txt"`), (b) `add_break(0)` produces no output.
- [x] T010 Run all tests with `cd src && pytest` to verify no regressions across all existing test files and new `test_file_logger.py`.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Foundational (Phase 1)**: No dependencies — can start immediately. BLOCKS all user stories.
- **US1 (Phase 2)**: Depends on Phase 1 (needs FileLogger class with Console)
- **US2 (Phase 3)**: No implementation needed — covered by T001 `"w"` mode. Tests only.
- **US3 (Phase 4)**: No implementation needed — covered by T001 `no_color=True`. Tests only.
- **Polish (Phase 5)**: Depends on all phases being complete

### Parallel Opportunities

- T001 and T002 can run in parallel (different new files)
- T003, T004, T005 modify the same file — run sequentially
- T007, T008, T009 add tests to the same file — run sequentially after T006

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Foundational (T001, T002)
2. Complete Phase 2: User Story 1 (T003–T006)
3. **STOP and VALIDATE**: Create a FileLogger, call all methods, inspect the output file

### Incremental Delivery

1. Foundational → Class skeleton ready
2. US1 → All logging methods work → **MVP**
3. US2 → Overwrite behavior confirmed via tests
4. US3 → Markup stripping confirmed via tests
5. Polish → Edge cases + full test run

---

## Notes

- US2 and US3 require no implementation — the behaviors are inherent in `Console(file=..., no_color=True)` and `open("w")` from T001. Those phases only add test coverage.
- The FileLogger mirrors `RichConsoleLogger`'s method implementations almost exactly — the only difference is the Console construction (file vs stdout, no_color vs colors).
- Total: 10 tasks, 5 implementation + 5 test
