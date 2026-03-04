from attr import dataclass
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase

from classification_trainer.configuration import (
    BaseModelInfo,
    ChatTemplateInfo,
    DatasetInfo,
    InferenceInfo,
    TrainingInfo,
    load_base_model_info,
    load_chat_template_info,
    load_dataset_info,
    load_inference_info,
    load_training_info,
)
from classification_trainer.helpers.dataset_helper import (
    add_eval_column,
    load_dataset_from_hf,
    log_dataset,
    prep_classification_dataset_for_training,
    split_dataset,
)
from classification_trainer.helpers.evaluation_helper import (
    ClassificationCounts,
    add_classification_result_column,
    classify_inference,
    collect_classification_counts,
    generate_metrics,
    get_metrics_from_inference_info,
)
from classification_trainer.helpers.inference_helper import (
    add_inferred_column,
    generate_label_text,
    setup_unsloth_inference,
)
from classification_trainer.helpers.reporting_helper import LoggerMetricsReporter
from classification_trainer.helpers.tokenizer_helper import load_tokenizer_from_hf
from classification_trainer.helpers.training_helper import load_base_model
from classification_trainer.protocols import CommmandProtocol, LoggingProtocol


@dataclass
class TestCommand(CommmandProtocol):
    dataset_info: DatasetInfo = load_dataset_info("imdb_full")
    base_model_info: BaseModelInfo = load_base_model_info("qwen2.5-1.5b-instruct")
    training_info: TrainingInfo = load_training_info("imdb")
    stress_set_rowcount: int = 100
    inference_info: InferenceInfo = load_inference_info("simple_classification")
    chat_template: ChatTemplateInfo = load_chat_template_info("chat-ml")
    positive_case: str = "pos"
    case_invariant: bool = True
    search_length: int = -1

    def execute(self, logger: LoggingProtocol) -> None:
        self.check_metric_reporting(logger)

    def log_row_after_prep(self, logger: LoggingProtocol) -> None:
        dataset: Dataset = load_dataset_from_hf(self.dataset_info)[self.dataset_info.training_split_name]
        tokenizer = load_tokenizer_from_hf(self.base_model_info)
        dataset = prep_classification_dataset_for_training(
            self.dataset_info, self.training_info, dataset, tokenizer, self.chat_template
        )
        dataset = add_eval_column(self.dataset_info, self.training_info, dataset, tokenizer)
        log_dataset(dataset, logger)

    def check_metric_reporting(self, logger: LoggingProtocol) -> None:
        datasets: DatasetDict = load_dataset_from_hf(self.dataset_info)
        training_dataset: Dataset = datasets[self.dataset_info.training_split_name]

        if self.dataset_info.validation_split_name is None:
            datasets, self.dataset_info = split_dataset(self.dataset_info, training_dataset, 0.1, 0.1, 3414)
            training_dataset = datasets[self.dataset_info.training_split_name]
        test_dataset: Dataset = datasets[self.dataset_info.test_split_name]

        tokenizer: PreTrainedTokenizerBase = load_tokenizer_from_hf(self.base_model_info)
        test_dataset = prep_classification_dataset_for_training(
            self.dataset_info,
            self.training_info,
            test_dataset,
            tokenizer,
            self.base_model_info.chat_template_info,
            True,
        )
        test_dataset = add_eval_column(self.dataset_info, self.training_info, test_dataset, tokenizer)
        model, tokenizer = load_base_model(self.base_model_info, self.training_info)
        inference_info: InferenceInfo = InferenceInfo()
        model, tokenizer = setup_unsloth_inference(model, tokenizer, inference_info)

        chat_template: ChatTemplateInfo = self.base_model_info.chat_template_info
        test_dataset = add_inferred_column(
            test_dataset,
            self.dataset_info,
            model,
            tokenizer,
            inference_info,
            chat_template,
            logger=logger,
        )
        test_dataset = add_classification_result_column(self.dataset_info, test_dataset)
        counts: ClassificationCounts = collect_classification_counts(self.dataset_info, test_dataset)
        metric_creators = get_metrics_from_inference_info(self.inference_info)
        reporter: LoggerMetricsReporter = LoggerMetricsReporter(logger)
        reporter.report(generate_metrics(counts, metric_creators))

    def compare_inference_methods(
        self, logger: LoggingProtocol, number_of_instances_of_each_classification_type: int = -1
    ) -> None:
        datasets: DatasetDict = load_dataset_from_hf(self.dataset_info)
        training_dataset: Dataset = datasets[self.dataset_info.training_split_name]

        if self.dataset_info.validation_split_name is None:
            datasets, self.dataset_info = split_dataset(self.dataset_info, training_dataset, 0.05, 0.75, 3414)
            training_dataset = datasets[self.dataset_info.training_split_name]
        test_dataset: Dataset = datasets[self.dataset_info.test_split_name]
        tokenizer: PreTrainedTokenizerBase = load_tokenizer_from_hf(self.base_model_info)
        test_dataset = prep_classification_dataset_for_training(
            self.dataset_info,
            self.training_info,
            test_dataset,
            tokenizer,
            self.base_model_info.chat_template_info,
            True,
        )
        inference_info: InferenceInfo = InferenceInfo()
        test_dataset = add_eval_column(self.dataset_info, self.training_info, test_dataset, tokenizer)
        model, tokenizer = load_base_model(self.base_model_info, self.training_info)
        model, tokenizer = setup_unsloth_inference(model, tokenizer, inference_info)

        chat_template: ChatTemplateInfo = self.base_model_info.chat_template_info
        prompt_col = self.dataset_info.evaluation_instructions_column_name

        # Single-row inference
        single_results: list[str] = []
        with logger.progress("Single-row inference", total=len(test_dataset)) as progress:
            for row in test_dataset:
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
            test_dataset,
            self.dataset_info,
            model,
            tokenizer,
            inference_info,
            chat_template,
            logger=logger,
        )

        # Add classification result columns for single and batch inference
        single_prediction_col = self.dataset_info.new_column_prefix + "single_prediction"
        single_classification_result_col = self.dataset_info.new_column_prefix + "single_classification_result"
        ground_truth_col = self.dataset_info.string_labels_column_name

        batch_dataset = batch_dataset.map(
            lambda row, idx: {
                single_prediction_col: single_results[idx],
                single_classification_result_col: classify_inference(
                    single_results[idx], row[ground_truth_col], self.dataset_info
                ).name,
            },
            with_indices=True,
        )
        batch_dataset = add_classification_result_column(self.dataset_info, batch_dataset)
        # log_dataset(batch_dataset, logger, 5)
        # logger.add_break()
        # Build list of row indices to report
        if number_of_instances_of_each_classification_type == -1:
            report_indices = list(range(len(test_dataset)))
        else:
            pair_counts: dict[tuple[str, str], int] = {}
            report_indices = []
            batch_classification_result_col = self.dataset_info.classification_result_column_name
            for i in range(len(test_dataset)):
                pair = (
                    batch_dataset[i][single_classification_result_col],
                    batch_dataset[i][batch_classification_result_col],
                )
                if pair_counts.get(pair, 0) < number_of_instances_of_each_classification_type:
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1
                    report_indices.append(i)
        # Output comparison table
        logger.report_multicolumn_table(
            headers=["ground_truth", "single", "single_result", "batch", "batch_result"],
            rows=[
                [
                    batch_dataset[i][self.dataset_info.content_column_name],
                    batch_dataset[i][self.dataset_info.string_labels_column_name],
                    batch_dataset[i][single_prediction_col],
                    batch_dataset[i][single_classification_result_col],
                    batch_dataset[i][self.dataset_info.prediction_column_name],
                    batch_dataset[i][self.dataset_info.classification_result_column_name],
                ]
                for i in report_indices
            ],
        )
