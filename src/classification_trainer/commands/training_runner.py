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
    prep_dataset,
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
from classification_trainer.utils import CommonPaths, flush_gpu_memory


@dataclass
class TrainingRunner:
    training_info: TrainingInfo
    dataset_info: DatasetInfo
    pretokenize: bool = False
    _data_splits: DatasetSplits | None = None
    _model: PreTrainedModel | None = None
    _tokenizer: PreTrainedTokenizerBase | None = None

    def report_blue_message(self, message: str, logger: LoggingProtocol) -> None:
        logger.report_message("[blue]" + message + "[/blue]")

    def prepare_data(self, logger: LoggingProtocol) -> None:
        self.report_blue_message("Preparing data...", logger)
        datasets: DatasetDict = load_dataset_from_hf(self.dataset_info)
        tokenizer: PreTrainedTokenizerBase = load_tokenizer_from_hf(self.training_info.base_model_info)
        self._data_splits, self.dataset_info = prepare_split_data(
            self.training_info, self.dataset_info, datasets, tokenizer, pretokenize=self.pretokenize
        )

    def prepare_stress_data(self, maximum_row_count: int, logger: LoggingProtocol) -> None:
        self.report_blue_message("Preparing data for stress testing...", logger)
        datasets: DatasetDict = load_dataset_from_hf(self.dataset_info)
        tokenizer: PreTrainedTokenizerBase = load_tokenizer_from_hf(self.training_info.base_model_info)
        training_dataset: Dataset = datasets[self.dataset_info.training_split_name]
        training_dataset = self.prepare_single_stress_dataset(training_dataset, maximum_row_count, tokenizer, logger)
        evaluation_dataset: Dataset = training_dataset
        if self.dataset_info.validation_split_name is not None:
            evaluation_dataset = self.prepare_single_stress_dataset(
                datasets[self.dataset_info.validate_split_name], maximum_row_count, tokenizer, logger
            )
        test_dataset: Dataset = training_dataset
        if self.dataset_info.validation_split_name is not None:
            test_dataset = self.prepare_single_stress_dataset(
                datasets[self.dataset_info.test_split_name], maximum_row_count, tokenizer, logger
            )
        self._data_splits = DatasetSplits(
            training_dataset=training_dataset, test_dataset=test_dataset, validation_dataset=evaluation_dataset
        )

    def prepare_single_stress_dataset(
        self, dataset: Dataset, maximum_row_count: int, tokenizer: PreTrainedTokenizerBase, logger: LoggingProtocol
    ) -> Dataset:
        dataset = prep_dataset(
            self.training_info,
            self.dataset_info,
            dataset,
            tokenizer,
            self.pretokenize,
        )
        dataset = make_stress_split(self.dataset_info, dataset, maximum_row_count, tokenizer)
        actual_count = len(dataset)
        if actual_count < maximum_row_count:
            logger.report_message(
                f"Warning: requested {maximum_row_count} stress rows but dataset only has {actual_count} rows."
            )

        return dataset

    def load_model(self, logger: LoggingProtocol) -> None:
        self.report_blue_message("Loading base model...", logger)
        self._model, self._tokenizer = load_base_model(self.training_info)

    def train_model(self, logger: LoggingProtocol, force_disable_wandb: bool = False) -> int:
        self.report_blue_message("Training model...", logger)
        if self._data_splits is None:
            raise ValueError("prepare_data must be called before train_model.  Datasets are None.")
        if self._model is None or self._tokenizer is None:
            raise ValueError("load_model must be called before train_model.  Model/Tokenizer are None.")

        CommonPaths.get().clear_cache_model_directories(self.training_info.model_name)
        enable_wandb = self.training_info.has_wandb and not force_disable_wandb
        trainer: SFTTrainer = create_trainer(
            self.dataset_info,
            self.training_info,
            self._model,
            self._tokenizer,
            self._data_splits.training_dataset,
            enable_wandb,
            self._data_splits.validation_dataset,
            output_dir=str(CommonPaths.get().get_model_checkpoint_directory(self.training_info.model_name)),
        )
        # trainer_lora_rank = trainer.model.peft_config["default"].r  # type:ignore
        # logger.report_message(f"Training with Lora Rank: {trainer_lora_rank}")
        return run_training(trainer, self._model)

    def evaluate_model(
        self, logger: LoggingProtocol, primary_logging_metric: MetricProtocol | None = None
    ) -> list[MetricResult]:
        if self._data_splits is None:
            raise ValueError("prepare_data must be called before evaluate_model.  Datasets are None.")
        if self._model is None or self._tokenizer is None:
            raise ValueError("load_model must be called before evaluate_model.  Model/Tokenizer are None.")

        self.report_blue_message("Evaluating model...", logger)
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

        counts.log_confusion_matrix(logger)

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
        self.report_blue_message("Saving model...", logger)
        publishing_helper.save_model(
            self._model, self._tokenizer, self.training_info, self.dataset_info, pre_metrics, post_metrics, logger
        )

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

    @staticmethod
    def _next_batch_size_candidate(last_good: int, last_failed: int | None) -> int | None:
        if last_failed is None:
            # Phase 1: exponential doubling
            return 1 if last_good == 0 else last_good * 2
        else:
            # Phase 2: binary search between last_good and last_failed
            mid = (last_good + last_failed) // 2
            return None if mid == last_good else mid

    def update_data_splits(self, data_splits: DatasetSplits, logger: LoggingProtocol) -> None:
        self._data_splits = data_splits

    @property
    def data_splits(self) -> DatasetSplits:
        if self._data_splits is None:
            raise ValueError("prepare_data must be called before accessing data splits.  Datasets are None.")
        return self._data_splits

    def find_max_batch_size(
        self,
        logger: LoggingProtocol,
    ) -> int:
        if self._data_splits is None:
            raise ValueError("prepare_data must be called before computing batch size.  Datasets are None.")
        if self._model is None or self._tokenizer is None:
            raise ValueError("load_model must be called before computing batch size.  Model/Tokenizer are None.")
        self.report_blue_message("Computing batch size...", logger)
        self.training_info = self.training_info.model_copy(
            update={
                "training_length_type": TrainingLengthType.STEPS,
                "gradient_accumulation_steps": 1,
                "per_device_batch_size": 1,
                "evaluation_enabled": True,
                "evaluation_steps": 1,
                "training_length": 3.0,
            }
        )

        last_good, last_failed = 0, None
        while (candidate := TrainingRunner._next_batch_size_candidate(last_good, last_failed)) is not None:
            try:
                logger.report_message(f"\tProbing Batch Size: {candidate}")
                self.training_info = self.training_info.model_copy(update={"per_device_batch_size": candidate})
                self.train_model(logger, True)
                last_good = candidate
            except torch.cuda.OutOfMemoryError:
                last_failed = candidate
            finally:
                flush_gpu_memory()

        return last_good
