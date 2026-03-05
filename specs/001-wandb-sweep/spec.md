# Feature Specification: WandB Hyperparameter Sweep Command

**Feature Branch**: `001-wandb-sweep`
**Created**: 2026-03-05
**Status**: Draft
**Input**: User description: "Add a new command to the application that uses wandb's sweep feature to do a sweep of training parameters. Use the parameters inside the SFTParameters class as options for the sweep. Use one of our custom metrics to define how good a sweep was, configurable in the inference_info yaml. It should be possible to look at the sweep results on wandb and grab the best configuration for training/publishing."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Run a Hyperparameter Sweep (Priority: P1)

A practitioner wants to find better training hyperparameters for a classification model without manually running dozens of experiments. They launch a single sweep command that automatically explores the hyperparameter search space, running multiple training trials and evaluating each one using the configured quality metric. All trial results appear in the wandb dashboard, ranked by quality.

**Why this priority**: This is the core value of the feature — automating the search for better training configurations. Without this, the feature doesn't exist.

**Independent Test**: Can be fully tested by running the sweep command with a small trial count and confirming that multiple trials complete, each with different hyperparameter combinations, and that results appear in the wandb dashboard ranked by the target metric.

**Acceptance Scenarios**:

1. **Given** a valid dataset, base model, training config, and inference config, **When** the user runs the sweep command with a trial count, **Then** the system runs that many training trials, each with a distinct hyperparameter combination drawn from the predefined search space.
2. **Given** a sweep is running, **When** each trial completes training, **Then** the system evaluates on the test set, computes the configured quality metric, and logs it to wandb so the sweep controller can rank the trial.
3. **Given** a sweep has completed, **When** the user views the wandb sweep dashboard, **Then** they can see all trials ranked by the quality metric and identify the best-performing hyperparameter combination.

---

### User Story 2 - Configure the Target Quality Metric (Priority: P2)

A practitioner wants the sweep to optimize for the metric that matters most to their use case — for example, F1 rather than accuracy. They specify the target metric and whether higher or lower is better in the inference configuration YAML, without touching any code.

**Why this priority**: Without a configurable target metric, the sweep may optimize for the wrong objective and deliver misleading results. This is critical to the feature's correctness.

**Independent Test**: Can be fully tested by running two sweeps with different target metric settings (e.g., one optimizing `f1`, one optimizing `accuracy`) and confirming each sweep's wandb page shows the correct metric as the optimization objective.

**Acceptance Scenarios**:

1. **Given** an inference config YAML with `sweep_metric: f1` and `sweep_metric_goal: maximize`, **When** the sweep command runs, **Then** wandb is configured to optimize for F1 and trials are ranked by F1 score in the dashboard.
2. **Given** a metric name in the YAML that is not in the available metrics list, **When** the sweep command is launched, **Then** the system exits with a clear error message before running any trials.
3. **Given** no `sweep_metric` is specified in the inference config YAML, **When** the sweep command runs, **Then** the system uses a sensible default metric (e.g., `f1`) and logs a notice to the user.

---

### User Story 3 - Apply the Best Configuration to a Training Run (Priority: P3)

After reviewing sweep results in the wandb dashboard, a practitioner wants to translate the best-performing hyperparameter combination into a `training_info` YAML that they can use with the existing `train` command. The sweep output should make this straightforward.

**Why this priority**: The sweep is only useful if its results can be acted on. Without a clear path from sweep results to a real training run, the feature is a dead end.

**Independent Test**: Can be fully tested by completing a sweep, identifying the best trial in wandb, and confirming that the hyperparameter values shown in wandb map directly to named fields in the `sft_parameters` section of the training YAML — enabling the user to copy them over manually.

**Acceptance Scenarios**:

1. **Given** a completed sweep, **When** the user views the best trial in the wandb dashboard, **Then** all hyperparameter names and values shown correspond exactly to fields in the `sft_parameters` block of the training YAML schema.
2. **Given** the user copies the best sweep trial's hyperparameters into a new training YAML, **When** the `train` command is run with that YAML, **Then** training completes successfully with those hyperparameters applied.

---

### Edge Cases

- What happens when a trial crashes mid-training (e.g., OOM)? The system should mark that trial as failed in wandb and continue to the next trial without halting the sweep.
- What happens if the user interrupts the sweep (Ctrl+C)? Trials completed so far remain visible in wandb; the sweep is left in a stopped state.
- What if the evaluation step produces no valid predictions for the quality metric (e.g., all outputs malformed)? The trial should log a sentinel value (e.g., 0.0 for maximize goals) and the error should be visible in the trial's wandb logs.
- What if `wandb_config` is not set in the training info? The sweep command should fail fast with a clear message before launching any trials.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The application MUST provide a new `sweep` CLI command accepting the same dataset, base model, training config, and inference config inputs as the existing `train` command, plus a trial count parameter.
- **FR-002**: The `sweep` command MUST use the predefined hyperparameter search space drawn from the training hyperparameter configuration, without requiring the user to define sweep ranges manually.
- **FR-003**: Each sweep trial MUST run a full training cycle followed by a classifier inference evaluation pass on the test set — identical to the post-training assessment in the `train` command — producing classification metrics (accuracy, precision, recall, F1) from predicted vs. ground-truth labels. The target quality metric from this pass MUST be logged to the experiment tracking system so the sweep controller can rank the trial. Trainer-internal loss metrics (e.g., eval_loss) MUST NOT be used as the sweep optimization objective.
- **FR-004**: The inference configuration YAML MUST support two new optional fields: `sweep_metric` (name of the metric to optimize) and `sweep_metric_goal` (either `maximize` or `minimize`), with `f1` / `maximize` as defaults.
- **FR-005**: The `sweep_metric` value MUST be validated against the list of available metrics at startup; if invalid, the command MUST exit with a descriptive error before any trials run.
- **FR-006**: Each trial MUST run in isolation — model and tokenizer are loaded fresh per trial to avoid state leaking between trials.
- **FR-007**: Trial outputs (checkpoints, intermediate files) MUST be stored in separate per-trial directories so trials do not overwrite each other.
- **FR-008**: The hyperparameter names logged to the experiment tracking system MUST match exactly the field names in the training YAML's `sft_parameters` section.
- **FR-009**: Failed trials (e.g., OOM or evaluation error) MUST be marked as failed without stopping the sweep; the sweep agent MUST continue to the next trial.
- **FR-010**: The sweep command MUST support a `--count` argument to cap the number of trials run by the local agent.

### Key Entities

- **Sweep**: A named experiment containing multiple trials, configured with a hyperparameter search space and a target metric. Owned by a wandb project.
- **Trial**: A single training run within a sweep, using one specific hyperparameter combination. Produces a quality metric score used to rank all trials.
- **Hyperparameter Search Space**: The set of candidate values for each training hyperparameter (e.g., learning rate ranges, LoRA rank options). Derived from the application's existing training hyperparameter defaults.
- **Target Metric**: A classification metric (accuracy, precision, recall, or F1) produced by running the trained model as a classifier on the test set and comparing predicted labels to ground truth — the same evaluation pipeline used by the `train` command's post-training assessment. This is distinct from trainer-internal metrics such as training loss or eval loss. The specific metric to optimize is designated per inference profile.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A user can launch a 10-trial sweep with a single command and have all trials complete and appear ranked in the wandb dashboard, with no manual intervention after the command starts.
- **SC-002**: Changing `sweep_metric` in the inference YAML changes which metric the sweep optimizes for, verifiable by comparing the wandb sweep objective shown for two sweeps with different settings.
- **SC-003**: 100% of hyperparameter names in the sweep trial results map directly to fields in the training YAML schema, enabling zero-friction copy of the best configuration.
- **SC-004**: A failed trial (e.g., deliberate OOM in testing) does not terminate the sweep; remaining trials continue and appear in wandb.
- **SC-005**: The `train` command can be run successfully using a training YAML built from the best sweep trial's hyperparameter values without any modifications to the command or its inputs.

## Assumptions

- The sweep always uses random search strategy (as already defined in the hyperparameter search space defaults); Bayesian or grid search strategies are out of scope for this feature.
- The user already has wandb configured and authenticated; the sweep command does not handle wandb login or project creation.
- Each trial trains on the full dataset as configured in the training info YAML (no automatic dataset slicing per trial); if the user wants cheaper trials, they configure a shorter training length in their training YAML.
- The sweep does not automatically apply the best configuration — it surfaces the results in wandb and the user applies them manually. Automated best-config extraction is out of scope.
- The `wandb_config` block must be present in the training info YAML for the sweep command to function; the sweep command does not create a wandb config on the user's behalf.
