# Tasks: Configurable Sequence Length Analysis

**Input**: Design documents from `/specs/005-configurable-sequence-lengths/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md

**Tests**: Tests are included — validation logic requires unit test coverage.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Phase 1: Foundational (Blocking Prerequisites)

**Purpose**: Add the `sequence_lengths` field to `DatasetInfo` with validation — MUST be done before any user story work

- [x] T001 Add `sequence_lengths: list[int] = [1024, 1536, 2048]` field to `DatasetInfo` in `src/classification_trainer/configuration/dataset_info.py`. Add a `field_validator` that: (a) rejects empty lists, (b) rejects non-positive values (all values must be > 0), (c) deduplicates while preserving order.
- [x] T002 Write unit tests in `tests/unit/test_dataset_helper.py` (or a new `tests/unit/test_dataset_info.py` if more appropriate) for the `sequence_lengths` field: default value is `[1024, 1536, 2048]` when omitted, custom list accepted, empty list rejected with `ValidationError`, zero rejected, negative value rejected, duplicates deduplicated preserving order, valid list of positive integers accepted.

**Checkpoint**: `DatasetInfo` model accepts optional `sequence_lengths` with full validation. All tests pass.

---

## Phase 2: User Story 1 - Configure Custom Sequence Lengths (Priority: P1)

**Goal**: The analyze-dataset command uses `sequence_lengths` from the dataset config instead of hardcoded values.

**Independent Test**: Add `sequence_lengths: [512, 768, 1024]` to a dataset config, run analyze-dataset, confirm output reports coverage for exactly those three values.

### Implementation for User Story 1

- [x] T003 [US1] Update `AnalyzeDatasetCommand.execute()` in `src/classification_trainer/commands/analyze_dataset.py`: replace the three hardcoded `self.produce_coverage_report_from_target(dataset, 1024, ...)` calls with a loop over `self.dataset_info.sequence_lengths`.

**Checkpoint**: Analyze-dataset command reports coverage for the configured list. Hardcoded values are removed.

---

## Phase 3: User Story 2 - Default Sequence Lengths (Priority: P2)

**Goal**: Existing configs without `sequence_lengths` produce identical output to current behavior.

**Independent Test**: Run analyze-dataset with an existing dataset config that has no `sequence_lengths` field, confirm output shows coverage for 1024, 1536, 2048.

### Implementation for User Story 2

No implementation tasks needed — backward compatibility is already handled by the default value `[1024, 1536, 2048]` set in T001. The T002 test covers this case.

**Checkpoint**: Existing configs work unchanged. Verified by T002 default-value test.

---

## Phase 4: User Story 3 - Documentation (Priority: P3)

**Goal**: Users can discover the feature via the example YAML and the CLI help text.

**Independent Test**: Read `dataset_info/example.yaml` and `analyze-dataset --help` — both explain the `sequence_lengths` option.

### Implementation for User Story 3

- [x] T004 [P] [US3] Add a fully commented `sequence_lengths` field to `dataset_info/example.yaml` with inline comments explaining: what it does (which token-count thresholds are reported by analyze-dataset), format (list of positive integers), default when omitted (`[1024, 1536, 2048]`), and a concrete example value (e.g., `[512, 1024, 1536, 2048, 4096]`).
- [x] T005 [P] [US3] Update the analyze-dataset command's docstring or help text in `src/classification_trainer/console/main.py` to mention that sequence lengths for the coverage report are configured via the `sequence_lengths` field in the dataset config YAML.

**Checkpoint**: Example YAML and CLI help both document the feature.

---

## Phase 5: Polish & Cross-Cutting Concerns

**Purpose**: Final validation

- [x] T006 Run all tests with `cd src && pytest` to verify no regressions across all existing test files and new sequence_lengths tests.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Foundational (Phase 1)**: No dependencies — can start immediately. BLOCKS user stories.
- **US1 (Phase 2)**: Depends on Phase 1 (needs field on DatasetInfo)
- **US2 (Phase 3)**: No implementation needed — covered by Phase 1 default value
- **US3 (Phase 4)**: Independent of US1 — can run in parallel after Phase 1
- **Polish (Phase 5)**: Depends on all phases being complete

### Parallel Opportunities

- T004 and T005 can run in parallel (different files)
- T004 and T005 can also run in parallel with T003 (all modify different files)

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Foundational (T001, T002)
2. Complete Phase 2: User Story 1 (T003)
3. **STOP and VALIDATE**: Test with a dataset config containing custom `sequence_lengths` and one without

### Incremental Delivery

1. Foundational → Model and validation ready
2. US1 → Command uses config values → **MVP**
3. US3 → Documentation in example.yaml and CLI help
4. Polish → Full test run

---

## Notes

- This is a small, focused feature — 6 tasks total
- US2 requires no implementation (backward compat handled by default value)
- All existing dataset config YAMLs continue to work unchanged
- The `produce_coverage_report_from_target` method is not changed — only its callers
