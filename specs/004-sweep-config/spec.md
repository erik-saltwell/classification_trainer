# Feature Specification: User-Configurable Sweep Parameters

**Feature Branch**: `004-sweep-config`
**Created**: 2026-03-06
**Status**: Draft
**Input**: User description: "Allow users to configure sweep parameter ranges, search method, and hyperparameter formats in their training config YAML instead of using hardcoded defaults"

## Clarifications

### Session 2026-03-06

- Q: When a user provides a `parameters` sub-block, should unmentioned parameters use hardcoded default sweep ranges (merge) or use the fixed value from `sft_parameters` (override/opt-in)? → A: Override — unmentioned parameters are NOT swept; they use the value from `sft_parameters`. The `parameters` block is an opt-in list.
- Q: What should happen when a `sweep` block has `method` but no `parameters` sub-block (zero parameters to sweep)? → A: Reject at startup with an error — the sweep block requires at least one parameter in `parameters`.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Customize Sweep Hyperparameter Ranges (Priority: P1)

A practitioner wants to sweep over a specific set of learning rates and LoRA ranks tailored to their task, rather than using the application's hardcoded defaults. They add a `sweep` block to their training config YAML, specifying custom value lists and continuous ranges for the parameters they want to vary. Parameters not listed in the sweep block are held constant at their `sft_parameters` values. They then run the existing sweep command and the sweep explores their custom search space.

**Why this priority**: This is the core problem — the hardcoded sweep search space cannot be customized without editing source code, which blocks practitioners from tuning sweeps to their specific use case.

**Independent Test**: Can be fully tested by adding a `sweep` block to a training config with custom parameter ranges, running a sweep with 3 trials, and confirming in wandb that trial hyperparameters fall within the user-specified ranges rather than the hardcoded defaults.

**Acceptance Scenarios**:

1. **Given** a training config with a `sweep` block specifying `learning_rate: {min: 1e-5, max: 1e-3}` and `rank: {values: [8, 64]}`, **When** the sweep command runs, **Then** all trials use learning rates between 1e-5 and 1e-3, and ranks of either 8 or 64.
2. **Given** a training config with a `sweep` block that specifies only `rank` and `learning_rate`, **When** the sweep command runs, **Then** all other parameters (optimizer, warmup_ratio, etc.) are held constant at their `sft_parameters` values and are not swept.
3. **Given** a training config with no `sweep` block, **When** the sweep command runs, **Then** the sweep uses the existing hardcoded default search space — full backward compatibility.

---

### User Story 2 - Fix a Parameter to a Single Value During Sweep (Priority: P2)

A practitioner knows they want to use a specific optimizer and does not want the sweep to vary it. They specify the parameter as a single fixed value in the sweep block. The sweep holds that parameter constant across all trials while varying the others.

**Why this priority**: Fixing known-good parameters reduces the search space and makes sweeps faster and more focused. This is essential for practitioners who have already narrowed down some hyperparameters.

**Independent Test**: Can be fully tested by setting `optim: "adamw_bnb_8bit"` (a single value, not a list) in the sweep block, running 5 trials, and confirming every trial uses that exact optimizer.

**Acceptance Scenarios**:

1. **Given** a sweep block with `optim: "adamw_bnb_8bit"`, **When** the sweep runs, **Then** every trial uses `adamw_bnb_8bit` as the optimizer — the parameter is not varied.
2. **Given** a sweep block with `rank: 32`, **When** the sweep runs, **Then** every trial uses rank 32 — a single scalar value is treated as a fixed constant.

---

### User Story 3 - Select a Sweep Search Method (Priority: P3)

A practitioner wants to use Bayesian optimization instead of random search to find good hyperparameters more efficiently. They set the search method in the sweep block and the sweep uses that strategy.

**Why this priority**: Different search strategies suit different scenarios. Bayesian search converges faster for many hyperparameter landscapes. Grid search is useful for exhaustive coverage of small discrete spaces. Without method selection, users are locked into random search.

**Independent Test**: Can be fully tested by running two sweeps — one with `method: random` and one with `method: bayes` — and confirming the wandb sweep dashboard shows the correct search method for each.

**Acceptance Scenarios**:

1. **Given** a sweep block with `method: bayes`, **When** the sweep runs, **Then** the wandb sweep is configured with Bayesian search.
2. **Given** a sweep block with `method: grid`, **When** the sweep runs with discrete-only parameters, **Then** the wandb sweep is configured with grid search.
3. **Given** a sweep block with no `method` field, **When** the sweep runs, **Then** the sweep defaults to random search (backward compatible).

---

### User Story 4 - Use Continuous Distributions for Numeric Parameters (Priority: P4)

A practitioner wants to specify a continuous range for learning rate using a log-uniform distribution, and a uniform range for weight decay. They express these as min/max ranges in the sweep block and the system maps them to the appropriate wandb distribution types.

**Why this priority**: Continuous distributions are more statistically appropriate for numeric hyperparameters than discrete value lists, especially for learning rate (log-scale) and regularization parameters.

**Independent Test**: Can be fully tested by specifying `learning_rate: {min: 1e-5, max: 1e-3}` and `weight_decay: {min: 0.0, max: 0.1}` in the sweep block, running a sweep, and confirming trial values are sampled from the full continuous range rather than a discrete set.

**Acceptance Scenarios**:

1. **Given** a sweep block with `learning_rate: {min: 1e-5, max: 1e-3}`, **When** the sweep runs, **Then** learning rates are sampled from a log-uniform distribution between 1e-5 and 1e-3.
2. **Given** a sweep block with `weight_decay: {min: 0.0, max: 0.1}`, **When** the sweep runs, **Then** weight decay values are sampled from a uniform distribution between 0.0 and 0.1.
3. **Given** a sweep block with `lora_dropout: {min: 0.0, max: 0.5}`, **When** the sweep runs, **Then** dropout values are sampled from a uniform distribution between 0.0 and 0.5.

---

### User Story 5 - Monitor Sweep Progress and Verify Trial Parameters (Priority: P5)

A practitioner running a long sweep wants to know how far along it is and whether the sweep is actually varying parameters as expected. At the start of each trial, the system reports the current trial number out of the total requested (e.g., "Trial 3 of 10") and prints the training parameters that will be used for that trial. This lets the practitioner confirm the sweep is working without opening the wandb dashboard.

**Why this priority**: Observability during long-running sweeps reduces anxiety and catches misconfiguration early. Without progress reporting, a practitioner has no terminal feedback on whether the sweep is advancing or stuck. Without parameter display, they cannot verify the sweep is actually varying hyperparameters.

**Independent Test**: Can be fully tested by running a sweep with 3 trials and confirming that (a) each trial prints its number and total count, and (b) each trial prints the hyperparameter values it will use before training begins.

**Acceptance Scenarios**:

1. **Given** a sweep with `--count 10`, **When** each trial starts, **Then** the system prints the trial number and total (e.g., "Trial 3 of 10") to the terminal.
2. **Given** a sweep trial with specific hyperparameter values assigned by the sweep controller, **When** the trial starts but before training begins, **Then** the system prints all training parameter names and their values for that trial to the terminal.

---

### Edge Cases

- What happens when a user specifies an invalid parameter name in the sweep block (e.g., a typo like `lerning_rate`)? The system MUST reject the config at startup with a clear error listing the invalid field names and the valid options.
- What happens when a user specifies `method: grid` but includes a parameter with a continuous min/max range? The system MUST reject the config with an error explaining that grid search requires all parameters to use discrete value lists.
- What happens when a user provides an empty values list (e.g., `rank: {values: []}`)? The system MUST reject the config with a clear error.
- What happens when a user provides a min greater than max (e.g., `learning_rate: {min: 1e-3, max: 1e-5}`)? The system MUST reject the config with a clear error.
- What happens when a user specifies a value outside the parameter's valid domain (e.g., `lora_dropout: {values: [1.5]}` when dropout must be 0.0-1.0)? The system MUST validate against the parameter's constraints and reject invalid values.
- What happens when a `sweep` block contains `method` but no `parameters` sub-block (or an empty one)? The system MUST reject the config at startup with an error requiring at least one parameter to sweep.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The training config YAML MUST support an optional `sweep` block at the top level, alongside the existing `sft_parameters` block. Configs without a `sweep` block MUST continue to work identically to today.
- **FR-002**: The `sweep` block MUST support a `method` field accepting `random`, `bayes`, or `grid`. When omitted, the method MUST default to `random`.
- **FR-003**: The `sweep` block MUST support a `parameters` sub-block where each key is a valid `sft_parameters` field name. Invalid field names MUST be rejected with a clear error at startup.
- **FR-004**: Each parameter in the `parameters` sub-block MUST support three formats:
  - **Discrete list**: `{values: [v1, v2, ...]}` — the sweep samples from these values
  - **Continuous range**: `{min: <number>, max: <number>}` — the sweep samples from a continuous distribution between min and max
  - **Fixed value**: a bare scalar value — the parameter is held constant across all trials
- **FR-005**: For continuous range parameters, the system MUST select the appropriate distribution type automatically: log-uniform for `learning_rate`, uniform for all other numeric parameters. Users MUST NOT need to specify distribution names.
- **FR-006**: Parameters not mentioned in the user's `parameters` sub-block MUST NOT be swept. They MUST use the fixed value from the `sft_parameters` block in the same training config. The `parameters` sub-block is an opt-in list — only listed parameters are varied during the sweep.
- **FR-007**: The system MUST validate the sweep configuration at startup, before any trials run, rejecting configs with: empty value lists, min > max, invalid parameter names, grid search combined with continuous ranges, or a sweep block with no parameters (missing or empty `parameters` sub-block).
- **FR-008**: The existing sweep config builder MUST be updated to build the sweep config from the user's `parameters` sub-block, using `sft_parameters` values as fixed constants for any parameters not listed.
- **FR-009**: The sweep block MUST be documented in the training config `example.yaml` with inline comments explaining all three parameter formats, the method field, and a complete example.
- **FR-010**: At the start of each sweep trial, the system MUST report the trial number and total count to the terminal (e.g., "Trial 3 of 10").
- **FR-011**: At the start of each sweep trial, before training begins, the system MUST print all training parameter names and their values for that trial to the terminal, so the user can verify the sweep is varying parameters as expected.

### Key Entities

- **Sweep Block**: An optional top-level section in the training config YAML that defines the hyperparameter search space and search method for sweeps. Does not affect non-sweep training runs.
- **Parameter Format**: One of three ways to specify a sweep parameter — discrete values list, continuous min/max range, or fixed scalar value.
- **Search Method**: The algorithm used to explore the hyperparameter space — random sampling, Bayesian optimization, or exhaustive grid search.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A user can customize sweep parameter ranges by editing only their training config YAML — no source code changes required.
- **SC-002**: Existing training configs without a `sweep` block produce identical sweep behavior to the current hardcoded defaults.
- **SC-003**: A user can fix any parameter to a constant value by specifying a bare scalar, verified by running a sweep and confirming that parameter is identical across all trials.
- **SC-004**: A user can select between random, Bayesian, and grid search methods via a single YAML field, verified by the wandb sweep dashboard showing the correct method.
- **SC-005**: Invalid sweep configurations (bad parameter names, empty lists, min > max, grid + continuous) are caught at startup with clear error messages — before any trials run.
- **SC-006**: The `example.yaml` for training config includes a fully commented `sweep` block that a new user can copy and modify without consulting external documentation.
- **SC-007**: Each sweep trial prints its trial number (e.g., "Trial 3 of 10") and the full set of training parameter values to the terminal before training starts, verified by inspecting terminal output during a sweep run.

## Assumptions

- The sweep block only controls parameters that exist in `sft_parameters`. Sweeping over other training config fields (e.g., `max_sequence_length`, `per_device_batch_size`) is out of scope.
- The automatic distribution selection (log-uniform for learning rate, uniform for others) covers all practical use cases. Allowing users to specify arbitrary wandb distribution types is out of scope.
- When a `sweep` block with a `parameters` sub-block is present, it fully controls which parameters are swept (opt-in). When no `sweep` block exists at all, the existing hardcoded default search space is used for backward compatibility.
- The `sweep` block is ignored during non-sweep training runs (the `train` command uses `sft_parameters` directly).
- wandb's sweep API supports the method types (random, bayes, grid) used here. No custom search algorithms are needed.
- The `sweep_metric` and `sweep_metric_goal` fields remain in the inference config, not the sweep block — they control what metric to optimize, not how to search.
