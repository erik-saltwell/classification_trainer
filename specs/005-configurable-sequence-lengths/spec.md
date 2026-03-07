# Feature Specification: Configurable Sequence Length Analysis

**Feature Branch**: `005-configurable-sequence-lengths`
**Created**: 2026-03-07
**Status**: Draft
**Input**: User description: "Move the analyze-dataset command's hardcoded sequence lengths (1024, 1536, 2048) to a configurable list in the dataset_info YAML. Update the command's help text and add a well-documented example in the dataset_info example YAML."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Configure Custom Sequence Lengths for Dataset Analysis (Priority: P1)

A practitioner wants to analyze how much of their dataset fits within specific sequence lengths relevant to their use case. Instead of being limited to the hardcoded values (1024, 1536, 2048), they specify their own list of sequence lengths in the dataset config YAML. When they run the analyze-dataset command, the system reports coverage for each of their configured lengths.

**Why this priority**: This is the entire feature — moving from hardcoded to configurable. Without this, the feature doesn't exist.

**Independent Test**: Can be fully tested by adding a `sequence_lengths` list to a dataset config YAML (e.g., `[512, 768, 1024, 2048, 4096]`), running the analyze-dataset command, and confirming the output shows coverage reports for exactly those five lengths.

**Acceptance Scenarios**:

1. **Given** a dataset config with `sequence_lengths: [512, 1024, 2048]`, **When** the user runs the analyze-dataset command, **Then** the system reports coverage for 512, 1024, and 2048 — exactly the specified values in the specified order.
2. **Given** a dataset config with `sequence_lengths: [256, 4096]`, **When** the user runs the analyze-dataset command, **Then** the system reports coverage for 256 and 4096 only — not the old hardcoded values.

---

### User Story 2 - Use Default Sequence Lengths When None Configured (Priority: P2)

A practitioner has an existing dataset config YAML that does not include a `sequence_lengths` field. When they run the analyze-dataset command, the system uses the default sequence lengths (1024, 1536, 2048) — identical to today's behavior. No existing configs break.

**Why this priority**: Backward compatibility ensures existing workflows are not disrupted. This is critical for adoption.

**Independent Test**: Can be fully tested by running the analyze-dataset command with an existing dataset config that has no `sequence_lengths` field and confirming the output matches the current behavior (coverage for 1024, 1536, 2048).

**Acceptance Scenarios**:

1. **Given** a dataset config with no `sequence_lengths` field, **When** the user runs the analyze-dataset command, **Then** the system reports coverage for 1024, 1536, and 2048 — the current default values.

---

### User Story 3 - Discover the Feature via Documentation (Priority: P3)

A practitioner browsing the dataset config example YAML or the analyze-dataset command's help text discovers the `sequence_lengths` option. The example YAML explains the field's purpose, format, default behavior, and includes a representative example. The command help text mentions that sequence lengths come from the dataset config.

**Why this priority**: Without documentation, users won't know the feature exists. The example YAML is the primary discovery mechanism for config options in this project.

**Independent Test**: Can be fully tested by reading the `dataset_info/example.yaml` file and the analyze-dataset command help text, and confirming both explain the `sequence_lengths` field clearly without consulting source code.

**Acceptance Scenarios**:

1. **Given** the `dataset_info/example.yaml` file, **When** a user reads it, **Then** they find a `sequence_lengths` field with inline comments explaining: what it does, what format to use (list of positive integers), what the default is when omitted, and a concrete example value.
2. **Given** the analyze-dataset command, **When** a user views its help output, **Then** the help text mentions that sequence lengths for the coverage report are configured in the dataset config YAML.

---

### Edge Cases

- What happens when a user specifies an empty list (`sequence_lengths: []`)? The system MUST reject the config with a clear error requiring at least one sequence length.
- What happens when a user specifies a non-positive value (e.g., `sequence_lengths: [0]` or `sequence_lengths: [-1]`)? The system MUST reject the config with a clear error stating that sequence lengths must be positive integers.
- What happens when a user specifies duplicate values (e.g., `sequence_lengths: [1024, 1024]`)? The system SHOULD accept the config and report coverage once per unique value (deduplicate silently).
- What happens when a user specifies non-integer values (e.g., `sequence_lengths: [1024.5]`)? The system MUST reject the config with a clear error.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The dataset config YAML MUST support an optional `sequence_lengths` field containing a list of positive integers. When omitted, the system MUST default to `[1024, 1536, 2048]`.
- **FR-002**: The analyze-dataset command MUST report coverage for each sequence length in the configured list, in the order specified.
- **FR-003**: The system MUST validate that all values in `sequence_lengths` are positive integers. Invalid values MUST be rejected with a clear error at config load time.
- **FR-004**: An empty `sequence_lengths` list MUST be rejected with a clear error at config load time.
- **FR-005**: Duplicate values in the list MUST be deduplicated — each unique sequence length is reported once.
- **FR-006**: The `dataset_info/example.yaml` MUST include a fully commented `sequence_lengths` field documenting its purpose, format, default value, and a concrete example.
- **FR-007**: The analyze-dataset command's help text MUST reference that sequence lengths are configured in the dataset config YAML.

### Key Entities

- **Sequence Length List**: An ordered list of positive integers in the dataset config, specifying which token-count thresholds to evaluate dataset coverage against.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A user can specify any list of positive integer sequence lengths in their dataset config YAML and receive coverage reports for exactly those values — no source code changes required.
- **SC-002**: Existing dataset configs without a `sequence_lengths` field produce identical output to the current hardcoded behavior (coverage for 1024, 1536, 2048).
- **SC-003**: Invalid sequence length configurations (empty list, non-positive values, non-integers) are caught at config load time with clear error messages — before the analysis runs.
- **SC-004**: The `example.yaml` for dataset config includes a fully commented `sequence_lengths` field that a new user can understand and customize without consulting source code or external documentation.

## Assumptions

- The field is named `sequence_lengths` (plural, snake_case) to be consistent with the project's YAML naming conventions and clearly communicate that a list of values is expected.
- The default value `[1024, 1536, 2048]` matches the current hardcoded behavior exactly — no change in default output.
- Sequence lengths are always specified in the dataset config, not as CLI arguments. There is no need for a CLI override.
- The coverage report format (percentage and sample loss count per length) does not change — only which lengths are evaluated changes.
