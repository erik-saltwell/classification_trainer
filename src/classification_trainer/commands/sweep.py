from __future__ import annotations

import typer
from attr import dataclass
from datasets import DatasetDict

import wandb
from classification_trainer.configuration import BaseModelInfo, DatasetInfo, TrainingInfo
from classification_trainer.configuration.chat_template_info import ChatTemplateInfo
from classification_trainer.configuration.inference_info import InferenceInfo
from classification_trainer.helpers.dataset_helper import (
    add_eval_column,
    load_dataset_from_hf,
    prep_classification_dataset_for_training,
    split_dataset,
)
from classification_trainer.helpers.evaluation_helper import (
    _METRIC_REGISTRY,
    ClassificationCounts,
    MetricProtocol,
    add_classification_result_column,
    collect_classification_counts,
    generate_metrics,
    get_metrics_from_inference_info,
)
from classification_trainer.helpers.inference_helper import (
    add_inferred_column,
    setup_unsloth_inference,
)
from classification_trainer.helpers.reporting_helper import (
    CompositeMetricsReporter,
    LoggerMetricsReporter,
    WandBMetricsReporter,
)
from classification_trainer.helpers.sweep_helper import apply_trial_sft_parameters, build_sweep_config
from classification_trainer.helpers.tokenizer_helper import load_tokenizer_from_hf
from classification_trainer.helpers.training_helper import create_trainer, load_base_model, run_training
from classification_trainer.protocols import CommmandProtocol, LoggingProtocol
from classification_trainer.utils.common_paths import CommonPaths


@dataclass
class SweepCommand(CommmandProtocol):
    dataset_info: DatasetInfo
    base_model_info: BaseModelInfo
    training_info: TrainingInfo
    inference_info: InferenceInfo
    count: int = 10

    def execute(self, logger: LoggingProtocol) -> None:
        # --- Startup validation ---
        if self.training_info.wandb_config is None:
            logger.report_message(
                "[red]Error:[/red] wandb_config must be set in training_info to use the sweep command."
            )
            raise typer.Exit(code=1)

        sweep_metric = self.inference_info.sweep_metric
        sweep_metric_goal = self.inference_info.sweep_metric_goal

        if sweep_metric not in _METRIC_REGISTRY:
            logger.report_message(
                f"[red]Error:[/red] sweep_metric '{sweep_metric}' is not a valid metric. "
                f"Valid options: {sorted(_METRIC_REGISTRY.keys())}"
            )
            raise typer.Exit(code=1)

        if sweep_metric_goal not in ("maximize", "minimize"):
            logger.report_message(
                f"[red]Error:[/red] sweep_metric_goal '{sweep_metric_goal}' is invalid. "
                "Must be 'maximize' or 'minimize'."
            )
            raise typer.Exit(code=1)

        logger.report_message(f"[blue]Sweep optimising for: {sweep_metric} ({sweep_metric_goal})[/blue]")

        # --- Create sweep ---
        wandb_config = self.training_info.wandb_config
        sweep_config = build_sweep_config(self.inference_info)
        sweep_id = wandb.sweep(sweep=sweep_config, project=wandb_config.project)
        logger.report_message(f"[green]Sweep created. ID: {sweep_id}[/green]")

        # --- Per-trial function ---
        def _run_trial() -> None:
            try:
                run = wandb.init(
                    project=wandb_config.project,
                    group=wandb_config.group,
                    job_type=wandb_config.job_type,
                )
                trial_training_info = apply_trial_sft_parameters(self.training_info, dict(wandb.config))
                output_dir = str(CommonPaths.get().sweep_trial_outputs(self.training_info.model_name, run.id))

                tokenizer = load_tokenizer_from_hf(self.base_model_info)

                raw_datasets: DatasetDict = load_dataset_from_hf(self.dataset_info)
                dataset_info = self.dataset_info
                if self.dataset_info.validation_split_name is None:
                    raw_datasets, dataset_info = split_dataset(
                        self.dataset_info,
                        raw_datasets[self.dataset_info.training_split_name],
                        0.1,
                        0.1,
                        trial_training_info.seed,
                    )

                chat_template: ChatTemplateInfo = self.base_model_info.chat_template_info

                def _prep(ds):
                    ds = prep_classification_dataset_for_training(
                        dataset_info, trial_training_info, ds, tokenizer, chat_template
                    )
                    return add_eval_column(dataset_info, trial_training_info, ds, tokenizer)

                training_dataset = _prep(raw_datasets[dataset_info.training_split_name])
                validation_dataset = _prep(raw_datasets[dataset_info.validation_split_name])
                test_dataset = _prep(raw_datasets[dataset_info.test_split_name])

                model, tokenizer = load_base_model(self.base_model_info, trial_training_info)

                trainer = create_trainer(
                    dataset_info,
                    trial_training_info,
                    self.base_model_info,
                    model,
                    tokenizer,
                    training_dataset,
                    True,
                    validation_dataset,
                    output_dir=output_dir,
                )

                final_step = run_training(trainer, model)

                model, tokenizer = setup_unsloth_inference(model, tokenizer, self.inference_info)
                test_dataset = add_inferred_column(
                    test_dataset,
                    dataset_info,
                    model,
                    tokenizer,
                    self.inference_info,
                    chat_template,
                    logger=logger,
                )
                test_dataset = add_classification_result_column(dataset_info, test_dataset)
                counts: ClassificationCounts = collect_classification_counts(dataset_info, test_dataset)
                metric_creators: list[MetricProtocol] = list(get_metrics_from_inference_info(self.inference_info))
                results = list(generate_metrics(counts, metric_creators))

                reporter = CompositeMetricsReporter([LoggerMetricsReporter(logger), WandBMetricsReporter()])
                reporter.report(results, step=final_step + 1)

                wandb.finish()

            except Exception:
                wandb.finish(exit_code=1)
                raise

        # --- Launch agent ---
        wandb.agent(sweep_id=sweep_id, function=_run_trial, count=self.count)
