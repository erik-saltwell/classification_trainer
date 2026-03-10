# Feature Specification: Sweep Parameter Simplification and Observability

**Feature Branch**: `010-sweep-params-observability`
**Created**: 2026-03-09
**Status**: Draft
**Input**: User description: "Spec out two sets of changes to the sweep feature. Change how sweep parameters are specified in SweepInfo and the related yaml. You should not require that all SFTParameter values are specified there, only the ones that will be swept. The general flow of a sweep run should be: Make a copy of the training_info SFTParameters that is in the SweepCommand (which is passed in from the yaml file). Update with any parameters sent by wandb. Do the training run. Add some observability to sweeps, specifically: When we initialize the sweep, log the sweep config we pass to wandb out using the logger. After the SFTTrainer is created for a single run of a sweep, we should look at the run.config object coming from wandb, then log a multi-column table using the logger with one row for every passed in value, and three columns: the name of the parameter, the value passed in by wandb, and the value that is actually in the trainer."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Swept Parameters Only in Sweep Config (Priority: P1)

A researcher configuring a hyperparameter sweep wants to list only the parameters being varied — not every possible training hyperparameter. The sweep config block should act as an override layer on top of the existing training configuration, not a full replacement of it.

**Why this priority**: This is the most impactful change. The current requirement to re-specify every `sft_parameters` field in the sweep block is redundant, error-prone (values can get out of sync), and confusing — it implies the sweep config fully owns the training parameters when it should only own the delta.

**Independent Test**: A researcher can write a sweep config with only `rank` and `learning_rate` listed, run the sweep, and each trial trains with those two values varied while all other training hyperparameters come unchanged from the `sft_parameters` block in `training_info`.

**Acceptance Scenarios**:

1. **Given** a sweep config that lists only `rank` and `learning_rate`, **When** a sweep trial runs, **Then** the trial uses the wandb-sampled values for `rank` and `learning_rate`, and all other training hyperparameters (e.g. `optim`, `warmup_ratio`, `weight_decay`) are taken from the `sft_parameters` block in `training_info` unchanged.

2. **Given** a sweep config that lists only `rank`, **When** the sweep is registered with wandb, **Then** the wandb sweep definition only includes `rank` as a variable parameter — not fixed-value entries for every other `sft_parameters` field.

3. **Given** a wandb trial config that contains only the swept fields, **When** the training parameters are built for that trial, **Then** the result is identical to the base `sft_parameters` with only the swept fields overridden.

4. **Given** a sweep config with no `parameters` block, **When** the sweep command is validated, **Then** a clear error is reported stating at least one swept parameter is required.

---

### User Story 2 - Sweep Initialization Logging (Priority: P2)

When a researcher kicks off a sweep, they want to see exactly what configuration was submitted to wandb — the full sweep definition including method, metric, and all parameter specs — so they can verify the sweep is configured as intended before committing trial compute.

**Why this priority**: Without this, the researcher has no feedback that the sweep was registered correctly. The wandb UI shows the sweep config but only after navigating to it. Immediate console output lets the researcher catch mistakes (wrong metric, wrong parameter ranges) before wasting GPU time.

**Independent Test**: Running the sweep command with a valid sweep config produces a log entry containing the sweep configuration (method, metric, parameter names and their specs) before any trial begins.

**Acceptance Scenarios**:

1. **Given** a valid sweep config, **When** the sweep is initialized, **Then** the sweep config submitted to wandb is written to the logger before the first trial starts.

2. **Given** the logged sweep config, **When** the researcher reads it, **Then** they can identify the sweep method, the optimization metric and goal, and each parameter with its sweep specification (values list, range, or fixed value).

3. **Given** sweep initialization that fails, **When** the error occurs, **Then** the sweep config is still logged (it was logged before the failure) so the researcher can diagnose the issue.

---

### User Story 3 - Per-Trial Parameter Verification Table (Priority: P2)

At the start of each sweep trial, a researcher wants to see a side-by-side comparison of what wandb requested for the trial versus what the trainer was actually configured with, so they can confirm parameters are being applied correctly and catch any mismatch.

**Why this priority**: Equally important to story 2 — without per-trial visibility, a silent parameter mapping error (e.g. a field name mismatch or type coercion issue) would silently corrupt every trial result with no way to diagnose it after the fact.

**Independent Test**: Running a single sweep trial produces a table in the log with one row per wandb-provided parameter, showing the wandb value and the corresponding trainer value side by side.

**Acceptance Scenarios**:

1. **Given** a sweep trial where wandb provides values for `rank` and `learning_rate`, **When** the trainer is created for that trial, **Then** the logger outputs a table with one row for `rank` and one row for `learning_rate`, each showing the wandb-provided value and the trainer's actual value.

2. **Given** a sweep trial where wandb provides a value that is correctly applied, **When** the table is logged, **Then** the wandb column and trainer column for that row show the same value.

3. **Given** a sweep trial where a wandb value fails to apply (type error, out-of-range, etc.), **When** the parameter table is logged, **Then** the discrepancy between the wandb value and the actual trainer value is visible in the table.

4. **Given** a sweep trial, **When** the parameter table is logged, **Then** the table has exactly three columns: parameter name, wandb-provided value, and actual trainer value.

---

### Edge Cases

- What happens when wandb provides a parameter name that does not correspond to any known training parameter? The trial should fail with a clear error identifying the unknown parameter name.
- What happens when a swept parameter value from wandb cannot be applied (e.g. a string where a number is expected)? The trial fails with a diagnostic error; the parameter table is still logged showing the wandb value and what the trainer actually has.
- What happens when the wandb trial config is empty (no parameters provided)? The trial runs using only the base `sft_parameters` from `training_info`; the parameter table is either empty or omitted with a log note.
- What if two sweep trials run concurrently (future scenario)? Each trial's parameter table is associated with its own trial identifier. (Out of scope for this feature — sweeps currently run sequentially.)

## Requirements *(mandatory)*

### Functional Requirements

**Sweep parameter simplification:**

- **FR-001**: The sweep config `parameters` block MUST only require the fields being varied. Fields not listed are automatically sourced from the `sft_parameters` block in `training_info`.
- **FR-002**: When building trial training parameters, the system MUST start from the base `sft_parameters` in `training_info` and apply only the wandb-provided values as overrides.
- **FR-003**: When registering the sweep with wandb, the sweep definition MUST only include fields listed in the `parameters` block — not fixed-value entries for every unlisted `sft_parameters` field.
- **FR-004**: The sweep config MUST still require at least one parameter in the `parameters` block (an empty sweep definition is invalid).
- **FR-005**: Validation MUST confirm that every parameter name in the sweep `parameters` block matches a known training hyperparameter field. Unknown parameter names MUST be rejected at startup with a clear error message.

**Sweep initialization logging:**

- **FR-006**: When a sweep is initialized, the sweep configuration submitted to wandb MUST be logged using the standard logger before any trial begins.
- **FR-007**: The logged sweep configuration MUST include: the sweep method, the optimization metric name and goal, and each parameter with its specification (values, range, or fixed value).

**Per-trial parameter table:**

- **FR-008**: After the trainer is configured for each sweep trial, a multi-column table MUST be logged via the standard logger.
- **FR-009**: The table MUST contain exactly one row per parameter provided by wandb for that trial.
- **FR-010**: The table MUST have exactly three columns: parameter name, the value provided by wandb, and the value present in the trainer for that parameter.

### Key Entities

- **Base training parameters**: The `sft_parameters` block from `training_info`. Serves as the default for all training hyperparameters.
- **Sweep parameter spec**: The `parameters` block in `sweep_config`. Defines only the subset of training hyperparameters to be varied, and how to vary them.
- **Trial parameters**: The merged result of base training parameters overridden by wandb's sampled values for one trial. Ephemeral — computed fresh for each trial.
- **Sweep config**: The full definition sent to wandb: method, metric, and parameter specs. Logged at sweep initialization.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A sweep config for 2 parameters requires only those 2 parameter entries — the number of entries in the YAML `parameters` block equals the number of parameters being swept, not the total count of all training hyperparameters.
- **SC-002**: Every sweep trial log contains a parameter table before training begins, with one row per wandb-provided parameter.
- **SC-003**: The sweep initialization log entry appears before the first trial starts and contains all fields required to reconstruct the sweep definition (method, metric, parameters).
- **SC-004**: A researcher can determine from the per-trial table alone whether each wandb-requested value was applied correctly, without inspecting any other log output or source code.

## Assumptions

- The standard logger (`LoggingProtocol`) already supports multi-column table output (it has `report_multicolumn_table`). No new logging capability needs to be added.
- "The value actually in the trainer" means the value of each parameter as it exists in the `SFTParameters` object used to configure the trainer for that trial, not a value read back from the `SFTTrainer` object itself.
- Swept parameters are always a subset of `SFTParameters` fields — no other categories of training parameters (e.g. sequence length, batch size) are in scope for sweeping.
- The wandb trial config provides parameter values as a flat mapping of field name to value. No nested structures are expected.
- Sequential trial execution is assumed. Concurrent trial logging is out of scope.
