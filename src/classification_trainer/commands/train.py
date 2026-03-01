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
from classification_trainer.helpers.inference_helper import (
    add_inferred_column,
    generate_label_text,
    setup_unsloth_inference,
)
from classification_trainer.helpers.tokenizer_helper import load_tokenizer_from_hf
from classification_trainer.helpers.training_helper import load_base_model
from classification_trainer.protocols import CommmandProtocol, LoggingProtocol


@dataclass
class TrainCommand(CommmandProtocol):
    dataset_info: DatasetInfo
    base_model_info: BaseModelInfo
    training_info: TrainingInfo
    run_comparison_before_training: bool = True

    def execute(self, logger: LoggingProtocol) -> None:
        datasets: DatasetDict = load_dataset_from_hf(self.dataset_info)
        training_dataset: Dataset = datasets[self.dataset_info.training_split_name]

        if self.dataset_info.validation_split_name is None:
            datasets, self.dataset_info = split_dataset(self.dataset_info, training_dataset, 0.1, 0.1, 3414)
            training_dataset = datasets[self.dataset_info.training_split_name]
        test_dataset: Dataset = datasets[self.dataset_info.test_split_name]
        validation_dataset: Dataset = datasets[self.dataset_info.validation_split_name]
        tokenizer: PreTrainedTokenizerBase = load_tokenizer_from_hf(self.base_model_info)
        training_dataset = prep_classification_dataset_for_training(
            self.dataset_info,
            self.training_info,
            training_dataset,
            tokenizer,
            self.base_model_info.chat_template_info,
            True,
        )
        validation_dataset = prep_classification_dataset_for_training(
            self.dataset_info,
            self.training_info,
            validation_dataset,
            tokenizer,
            self.base_model_info.chat_template_info,
            True,
        )
        test_dataset = prep_classification_dataset_for_training(
            self.dataset_info,
            self.training_info,
            test_dataset,
            tokenizer,
            self.base_model_info.chat_template_info,
            True,
        )
        test_dataset = add_eval_column(self.dataset_info, self.training_info, test_dataset, tokenizer)
        # if self.run_comparison_before_training:
        #     log_dataset(test_dataset, logger, 1)
        model, tokenizer = load_base_model(self.base_model_info, self.training_info)
        model, tokenizer = setup_unsloth_inference(model, tokenizer)
        inference_info: InferenceInfo = InferenceInfo()
        chat_template: ChatTemplateInfo = self.base_model_info.chat_template_info
        # test_prompt: str = test_dataset[0][self.dataset_info.evaluation_instructions_column_name]
        # result: str = generate_label_text(
        #     model,
        #     tokenizer,
        #     test_prompt,
        #     inference_info,
        #     chat_template,
        # )
        self.compare_inference_methods(test_dataset, model, tokenizer, inference_info, chat_template, logger)

    def compare_inference_methods(
        self,
        dataset: Dataset,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        inference_info: InferenceInfo,
        chat_template: ChatTemplateInfo,
        logger: LoggingProtocol,
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

        # Output comparison table
        logger.report_multicolumn_table(
            headers=["input", "single", "batch"],
            rows=[
                [batch_dataset[i][self.dataset_info.content_column_name], single_results[i], batch_results[i]]
                for i in range(len(dataset))
            ],
        )
