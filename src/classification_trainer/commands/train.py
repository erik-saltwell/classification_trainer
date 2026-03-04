from attr import dataclass
from datasets import Dataset, DatasetDict
from transformers import PreTrainedModel, PreTrainedTokenizerBase

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
    add_classification_result_column,
    classify_inference,
)
from classification_trainer.helpers.inference_helper import (
    add_inferred_column,
    generate_label_text,
)
from classification_trainer.helpers.tokenizer_helper import load_tokenizer_from_hf
from classification_trainer.protocols import CommmandProtocol, LoggingProtocol


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

    def test_model(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, dataset: Dataset) -> None: ...

    def execute(self, logger: LoggingProtocol) -> None:
        datasets: DatasetDict = load_dataset_from_hf(self.dataset_info)
        datasets, self.dataset_info = self.extract_splits(datasets)
        tokenizer: PreTrainedTokenizerBase = load_tokenizer_from_hf(self.base_model_info)
        self.extract_prepared_datasets(datasets, tokenizer)

    def compare_inference_methods(
        self,
        dataset: Dataset,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        inference_info: InferenceInfo,
        chat_template: ChatTemplateInfo,
        logger: LoggingProtocol,
        positive_case: str,
        case_invariant: bool,
        search_length: int,
    ) -> None:
        prompt_col = self.dataset_info.evaluation_instructions_column_name

        # Single-row inference
        single_results: list[str] = []
        with logger.progress("Single-row inference", total=len(dataset)) as progress:
            for row in dataset:
                result = generate_label_text(
                    model,
                    tokenizer,
                    row[prompt_col],  # pyright: ignore
                    inference_info,
                    chat_template,
                )
                single_results.append(result)
                progress.advance()

        # Batched inference via add_inferred_column
        batch_dataset = add_inferred_column(
            dataset,
            self.dataset_info,
            model,
            tokenizer,
            inference_info,
            chat_template,
            logger=logger,
        )
        batch_results: list[str] = batch_dataset[self.dataset_info.prediction_column_name]

        # Add classification result columns for single and batch inference
        single_prediction_col = self.dataset_info.new_column_prefix + "single_prediction"
        single_classification_result_col = self.dataset_info.new_column_prefix + "single_classification_result"
        ground_truth_col = self.dataset_info.string_labels_column_name

        batch_dataset = batch_dataset.map(
            lambda row, idx: {
                single_prediction_col: single_results[idx],
                single_classification_result_col: classify_inference(
                    single_results[idx], row[ground_truth_col], self.dataset_info
                ),
            },
            with_indices=True,
        )
        batch_dataset = add_classification_result_column(self.dataset_info, batch_dataset)

        # Output comparison table
        logger.report_multicolumn_table(
            headers=["input", "single", "batch"],
            rows=[
                [batch_dataset[i][self.dataset_info.content_column_name], single_results[i], batch_results[i]]
                for i in range(len(dataset))
            ],
        )
