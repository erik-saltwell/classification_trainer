from __future__ import annotations

from dataclasses import dataclass

import torch
from datasets import Dataset, DatasetDict
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from trl.trainer.sft_trainer import SFTTrainer

import classification_trainer.helpers.publishing_helper as publishing_helper
from classification_trainer.configuration import DatasetInfo, TrainingInfo, TrainingLengthType
from classification_trainer.helpers.dataset_helper import (
    DatasetSplits,
    load_dataset_from_hf,
    make_stress_split,
    prepare_split_data,
)
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
from classification_trainer.utils import flush_gpu_memory
from classification_trainer.utils.common_paths import CommonPaths


@dataclass
class TrainingRunner:
    training_info: TrainingInfo
    dataset_info: DatasetInfo
    pretokenize: bool = False
    _data_splits: DatasetSplits | None = None
    _model: PreTrainedModel | None = None
    _tokenizer: PreTrainedTokenizerBase | None = None

    def validate_training_ready(self, logger: LoggingProtocol) -> None:
        if self._data_splits is None:
            raise ValueError("TrainingRunner not ready.  Datasets are None.")
        if self._model is None or self._tokenizer is None:
            raise ValueError("TrainingRunner not ready.  Model/Tokenizer are None.")

    def prepare_data(self, logger: LoggingProtocol) -> None:
        datasets: DatasetDict = load_dataset_from_hf(self.dataset_info)
        tokenizer: PreTrainedTokenizerBase = load_tokenizer_from_hf(self.training_info.base_model_info)
        self._data_splits, self.dataset_info = prepare_split_data(
            self.training_info, self.dataset_info, datasets, tokenizer, pretokenize=self.pretokenize
        )

    def load_model(self, logger: LoggingProtocol) -> None:
        self._model, self._tokenizer = load_base_model(self.training_info)

    def train_model(self, logger: LoggingProtocol, report_to_wandb: bool | None = None) -> int:
        self.validate_training_ready(logger)
        assert self._model is not None and self._tokenizer is not None and self._data_splits is not None
        logger.report_message("[blue]Begining Training...[/blue]")

        if report_to_wandb is None:
            report_to_wandb = self.training_info.wandb_config is not None

        trainer: SFTTrainer = create_trainer(
            self.dataset_info,
            self.training_info,
            self._model,
            self._tokenizer,
            self._data_splits.training_dataset,
            report_to_wandb,
            self._data_splits.validation_dataset,
            output_dir=str(CommonPaths.get().get_model_checkpoint_directory(self.training_info.model_name)),
        )
        return run_training(trainer, self._model)

    def evaluate_model(
        self, logger: LoggingProtocol, primary_logging_metric: MetricProtocol | None = None
    ) -> list[MetricResult]:
        self.validate_training_ready(logger)
        assert self._model is not None and self._tokenizer is not None and self._data_splits is not None

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
        self.validate_training_ready(logger)
        assert self._model is not None and self._tokenizer is not None and self._data_splits is not None
        publishing_helper.save_model(
            self._model, self._tokenizer, self.training_info, self.dataset_info, pre_metrics, post_metrics, logger
        )

    def _prepare_data_for_stress_test(self, number_of_rows_to_test: int, logger: LoggingProtocol) -> None:
        assert self._model is not None and self._tokenizer is not None and self._data_splits is not None
        self._data_splits.training_dataset = make_stress_split(
            self.dataset_info, self._data_splits.training_dataset, number_of_rows_to_test, self._tokenizer
        )
        self._data_splits.validation_dataset = make_stress_split(
            self.dataset_info, self._data_splits.validation_dataset, number_of_rows_to_test, self._tokenizer
        )

    def _update_training_info_for_batch_stress_test(self, per_device_batch_size: int, logger: LoggingProtocol) -> None:
        self.training_info: TrainingInfo = self.training_info.model_copy(
            update={
                "training_length_type": TrainingLengthType.STEPS,
                "gradient_accumulation_steps": 1,
                "per_device_batch_size": per_device_batch_size,
                "evaluation_enabled": True,
                "evaluation_steps": 1,
                "training_length": 3.0,
            }
        )

    @staticmethod
    def _next_batch_size_candidate(last_good: int, last_failed: int | None) -> int | None:
        if last_failed is None:
            # Phase 1: exponential doubling
            return 1 if last_good == 0 else last_good * 2
        else:
            # Phase 2: binary search between last_good and last_failed
            mid = (last_good + last_failed) // 2
            return None if mid == last_good else mid

    def compute_max_batch_size(self, number_of_rows_to_test: int, logger: LoggingProtocol) -> int:
        self.validate_training_ready(logger)
        assert self._model is not None and self._tokenizer is not None and self._data_splits is not None
        self._prepare_data_for_stress_test(number_of_rows_to_test, logger)
        last_good, last_failed = 0, None

        while (candidate := TrainingRunner._next_batch_size_candidate(last_good, last_failed)) is not None:
            try:
                self._update_training_info_for_batch_stress_test(candidate, logger)
                logger.report_message(f"Probing Batch Size: {candidate}")
                self.train_model(logger, False)
                last_good = candidate
            except torch.cuda.OutOfMemoryError:
                last_failed = candidate
            finally:
                flush_gpu_memory()

        return last_good

    @property
    def training_split(self) -> Dataset:
        if self._data_splits is None:
            raise ValueError("Trying to access None data_splits.")
        return self._data_splits.training_dataset

    @property
    def validation_split(self) -> Dataset:
        if self._data_splits is None:
            raise ValueError("Trying to access None data_splits.")
        return self._data_splits.validation_dataset

    @property
    def test_split(self) -> Dataset:
        if self._data_splits is None:
            raise ValueError("Trying to access None data_splits.")
        return self._data_splits.test_dataset
