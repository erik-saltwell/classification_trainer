from attr import dataclass
from datasets import Dataset, DatasetDict
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from trl.trainer.sft_trainer import SFTTrainer

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
from classification_trainer.protocols import CommmandProtocol, LoggingProtocol
from classification_trainer.protocols.metric_result import MetricResult


@dataclass
class DatasetSplits:
    training_dataset: Dataset
    test_dataset: Dataset
    validation_dataset: Dataset


@dataclass
class TrainCommand(CommmandProtocol):
    dataset_info: DatasetInfo
    base_model_info: BaseModelInfo
    training_info: TrainingInfo
    inference_info: InferenceInfo
    run_comparison_before_training: bool = True
    seed: int = 3414

    def extract_splits(self, datasets: DatasetDict) -> tuple[DatasetDict, DatasetInfo]:
        training_dataset: Dataset = datasets[self.dataset_info.training_split_name]
        dataset_info: DatasetInfo = self.dataset_info
        if self.dataset_info.validation_split_name is None:
            datasets, dataset_info = split_dataset(self.dataset_info, training_dataset, 0.1, 0.1, 3414)
            training_dataset = datasets[self.dataset_info.training_split_name]
        return datasets, dataset_info

    def prep_dataset(self, dataset: Dataset, tokenizer: PreTrainedTokenizerBase) -> Dataset:
        return_dataset = prep_classification_dataset_for_training(
            self.dataset_info, self.training_info, dataset, tokenizer, self.base_model_info.chat_template_info
        )
        return_dataset = add_eval_column(self.dataset_info, self.training_info, return_dataset, tokenizer)

        return return_dataset

    def extract_prepared_datasets(self, datasets: DatasetDict, tokenizer: PreTrainedTokenizerBase) -> DatasetSplits:
        training_dataset = self.prep_dataset(datasets[self.dataset_info.training_split_name], tokenizer)
        test_dataset = self.prep_dataset(datasets[self.dataset_info.test_split_name], tokenizer)
        validation_dataset = self.prep_dataset(datasets[self.dataset_info.validation_split_name], tokenizer)
        return DatasetSplits(
            training_dataset=training_dataset, test_dataset=test_dataset, validation_dataset=validation_dataset
        )

    def test_model(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        dataset: Dataset,
        metric_creators: list[MetricProtocol],
        logger: LoggingProtocol,
    ) -> list[MetricResult]:
        model, tokenizer = setup_unsloth_inference(model, tokenizer, self.inference_info)
        chat_template: ChatTemplateInfo = self.base_model_info.chat_template_info
        dataset = add_inferred_column(
            dataset, self.dataset_info, model, tokenizer, self.inference_info, chat_template, logger=logger
        )
        dataset = add_classification_result_column(self.dataset_info, dataset)
        counts: ClassificationCounts = collect_classification_counts(self.dataset_info, dataset)
        return_value = list(generate_metrics(counts, metric_creators))
        return return_value

    def train_model(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        training_dataset: Dataset,
        validation_dataset: Dataset,
    ) -> None:
        trainer: SFTTrainer = create_trainer(
            self.dataset_info,
            self.training_info,
            self.base_model_info,
            model,
            tokenizer,
            training_dataset,
            False,
            validation_dataset,
        )
        run_training(trainer, model)

    def report_results(
        self, pre_run_results: list[MetricResult], post_run_results: list[MetricResult], logger: LoggingProtocol
    ) -> None:
        pre_dict = {r.metric_name: r.metric_result for r in pre_run_results}
        post_dict = {r.metric_name: r.metric_result for r in post_run_results}
        if pre_dict.keys() != post_dict.keys():
            raise ValueError(
                f"Metric sets differ between pre and post results: "
                f"pre={sorted(pre_dict.keys())}, post={sorted(post_dict.keys())}"
            )
        logger.report_multicolumn_table(
            headers=["metric-name", "pre-result", "post-result"],
            rows=[[name, str(pre_dict[name]), str(post_dict[name])] for name in pre_dict],
        )

    def execute(self, logger: LoggingProtocol) -> None:
        logger.report_message("Loading Tokenizer...")
        tokenizer: PreTrainedTokenizerBase = load_tokenizer_from_hf(self.base_model_info)

        logger.report_message("Loading Datasets")
        datasets: DatasetDict = load_dataset_from_hf(self.dataset_info)
        datasets, self.dataset_info = self.extract_splits(datasets)
        data_splits = self.extract_prepared_datasets(datasets, tokenizer)

        logger.report_message("Loading base model...")
        model, tokenizer = load_base_model(self.base_model_info, self.training_info)

        logger.report_message("Pre Training Assessment...")
        metric_creators: list[MetricProtocol] = list(get_metrics_from_inference_info(self.inference_info))
        pre_run_results: list[MetricResult] = self.test_model(
            model, tokenizer, data_splits.test_dataset, metric_creators, logger
        )
        self.train_model(model, tokenizer, data_splits.training_dataset, data_splits.validation_dataset)
        post_run_results: list[MetricResult] = self.test_model(
            model, tokenizer, data_splits.test_dataset, metric_creators, logger
        )
        self.report_results(pre_run_results, post_run_results, logger)
