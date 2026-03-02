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
from classification_trainer.helpers.inference_helper import (
    add_inferred_column,
    generate_label_text,
    setup_unsloth_inference,
)
from classification_trainer.helpers.tokenizer_helper import load_tokenizer_from_hf
from classification_trainer.helpers.training_helper import load_base_model
from classification_trainer.protocols import CommmandProtocol, LoggingProtocol


@dataclass
class TestCommand(CommmandProtocol):
    dataset_info: DatasetInfo = load_dataset_info("imdb")
    base_model_info: BaseModelInfo = load_base_model_info("qwen2.5-1.5b-instruct")
    training_info: TrainingInfo = load_training_info("imdb")
    stress_set_rowcount: int = 100
    inference_info: InferenceInfo = load_inference_info("simple_classification")
    chat_template: ChatTemplateInfo = load_chat_template_info("chat-ml")

    def execute(self, logger: LoggingProtocol) -> None:
        self.compare_inference_methods(logger)

    def log_row_after_prep(self, logger: LoggingProtocol) -> None:
        dataset: Dataset = load_dataset_from_hf(self.dataset_info)[self.dataset_info.training_split_name]
        tokenizer = load_tokenizer_from_hf(self.base_model_info)
        dataset = prep_classification_dataset_for_training(
            self.dataset_info, self.training_info, dataset, tokenizer, self.chat_template
        )
        dataset = add_eval_column(self.dataset_info, self.training_info, dataset, tokenizer)
        log_dataset(dataset, logger)

    def compare_inference_methods(
        self,
        logger: LoggingProtocol,
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
        test_dataset = add_eval_column(self.dataset_info, self.training_info, test_dataset, tokenizer)
        model, tokenizer = load_base_model(self.base_model_info, self.training_info)
        model, tokenizer = setup_unsloth_inference(model, tokenizer)
        inference_info: InferenceInfo = InferenceInfo()
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
        batch_results: list[str] = batch_dataset[self.dataset_info.prediction_column_name]

        # Output comparison table
        logger.report_multicolumn_table(
            headers=["input", "single", "batch", "ground_truth"],
            rows=[
                [
                    batch_dataset[i][self.dataset_info.content_column_name],
                    single_results[i],
                    batch_results[i],
                    batch_dataset[i][self.dataset_info.string_labels_column_name],
                ]
                for i in range(len(test_dataset))
            ],
        )
