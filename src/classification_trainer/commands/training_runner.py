from __future__ import annotations

from dataclasses import dataclass

from datasets import DatasetDict
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from trl.trainer.sft_trainer import SFTTrainer

import classification_trainer.helpers.publishing_helper as publishing_helper
from classification_trainer.configuration import DatasetInfo, TrainingInfo
from classification_trainer.helpers.dataset_helper import DatasetSplits, load_dataset_from_hf, prepare_split_data
from classification_trainer.helpers.evaluation_helper import (
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
from classification_trainer.helpers.tokenizer_helper import load_tokenizer_from_hf
from classification_trainer.helpers.training_helper import create_trainer, load_base_model, run_training
from classification_trainer.protocols import LoggingProtocol, MetricResult
from classification_trainer.utils.common_paths import CommonPaths


@dataclass
class TrainingRunner:
    training_info: TrainingInfo
    dataset_info: DatasetInfo
    _data_splits: DatasetSplits | None = None
    _model: PreTrainedModel | None = None
    _tokenizer: PreTrainedTokenizerBase | None = None

    def prepare_data(self, logger: LoggingProtocol) -> None:
        datasets: DatasetDict = load_dataset_from_hf(self.dataset_info)
        tokenizer: PreTrainedTokenizerBase = load_tokenizer_from_hf(self.training_info.base_model_info)
        self._data_splits, self.dataset_info = prepare_split_data(
            self.training_info, self.dataset_info, datasets, tokenizer
        )

    def load_model(self, logger: LoggingProtocol) -> None:
        self._model, self._tokenizer = load_base_model(self.training_info)

    def train_model(self, logger: LoggingProtocol) -> int:
        if self._data_splits is None:
            raise ValueError("prepare_data must be called before train_model.  Datasets are None.")
        if self._model is None or self._tokenizer is None:
            raise ValueError("load_model must be called before train_model.  Model/Tokenizer are None.")
        logger.report_message("[blue]Begining Training...[/blue]")
        trainer: SFTTrainer = create_trainer(
            self.dataset_info,
            self.training_info,
            self._model,
            self._tokenizer,
            self._data_splits.training_dataset,
            self.training_info.wandb_config is not None,
            self._data_splits.validation_dataset,
            output_dir=str(CommonPaths.get().get_model_checkpoint_directory(self.training_info.model_name)),
        )
        return run_training(trainer, self._model)

    def evaluate_model(
        self, logger: LoggingProtocol, primary_logging_metric: MetricProtocol | None = None
    ) -> list[MetricResult]:
        if self._data_splits is None:
            raise ValueError("prepare_data must be called before evaluate_model.  Datasets are None.")
        if self._model is None or self._tokenizer is None:
            raise ValueError("load_model must be called before evaluate_model.  Model/Tokenizer are None.")

        logger.report_message("[blue]Evaluating Model...[/blue]")
        model, tokenizer = setup_unsloth_inference(self._model, self._tokenizer, self.training_info.inference_info)

        dataset = add_inferred_column(
            self._data_splits.test_dataset,
            self.dataset_info,
            model,
            tokenizer,
            self.training_info.inference_info,
            self.training_info.base_model_info.chat_template_info,
            logger=logger,
        )

        dataset = add_classification_result_column(self.dataset_info, dataset)
        counts: ClassificationCounts = collect_classification_counts(self.dataset_info, dataset)

        metric_creators = get_metrics_from_inference_info(self.training_info.inference_info)
        return_value = list(generate_metrics(counts, metric_creators))
        if primary_logging_metric is not None:
            primary_result = primary_logging_metric.compute_metric(counts)
            logger.report_message(
                f"[blue]Metric Result:{primary_result.metric_name}={primary_result.metric_result}[/blue]"
            )
        return return_value

    def save_model(
        self,
        pre_metrics: list[MetricResult],
        post_metrics: list[MetricResult],
        logger: LoggingProtocol,
    ) -> None:
        if self._data_splits is None:
            raise ValueError("prepare_data must be called before evaluate_model.  Datasets are None.")
        if self._model is None or self._tokenizer is None:
            raise ValueError("load_model must be called before evaluate_model.  Model/Tokenizer are None.")
        publishing_helper.save_model(
            self._model, self._tokenizer, self.training_info, self.dataset_info, pre_metrics, post_metrics, logger
        )
