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
    add_inferred_column_with_vllm,
    generate_label_text,
    generate_label_text_with_vllm,
    setup_unsloth_inference,
)
from classification_trainer.helpers.tokenizer_helper import load_tokenizer_from_hf
from classification_trainer.helpers.training_helper import load_base_model_with_vllm
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
        model, tokenizer = load_base_model_with_vllm(self.base_model_info, self.training_info)
        model, tokenizer = setup_unsloth_inference(model, tokenizer)
        inference_info: InferenceInfo = InferenceInfo()
        chat_template: ChatTemplateInfo = self.base_model_info.chat_template_info
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

        # --- Phase 1: HF inference (model already loaded) ---

        # Single-row HF inference
        single_hf_results: list[str] = []
        with logger.progress("Single-row HF inference", total=len(dataset)) as progress:
            for row in dataset:
                result = generate_label_text(
                    model,
                    tokenizer,
                    row[prompt_col],  # pyright: ignore
                    inference_info,
                    chat_template,
                )
                single_hf_results.append(result)
                progress.advance()

        # Batched HF inference via add_inferred_column
        batch_hf_dataset = add_inferred_column(
            dataset,
            self.dataset_info,
            model,
            tokenizer,
            inference_info,
            chat_template,
            logger=logger,
        )
        batch_hf_results: list[str] = batch_hf_dataset[self.dataset_info.prediction_column_name]

        # --- Phase 2: vLLM inference via Unsloth's vllm_engine ---

        vllm_engine = model.vllm_engine  # pyright: ignore[attr-defined]
        lora_request = model.load_lora("_lora_compare", load_tensors=True)  # pyright: ignore[reportCallIssue, reportAttributeAccessIssue]

        # Single-row vLLM inference
        single_vllm_results: list[str] = []
        with logger.progress("Single-row vLLM inference", total=len(dataset)) as progress:
            for row in dataset:
                result = generate_label_text_with_vllm(
                    vllm_engine,
                    row[prompt_col],  # pyright: ignore
                    inference_info,
                    chat_template,
                    lora_request=lora_request,
                )
                single_vllm_results.append(result)
                progress.advance()

        # Batched vLLM inference
        vllm_batch_dataset = add_inferred_column_with_vllm(
            dataset,
            self.dataset_info,
            vllm_engine,
            inference_info,
            chat_template,
            lora_request=lora_request,
            logger=logger,
        )
        batch_vllm_results: list[str] = vllm_batch_dataset[self.dataset_info.prediction_column_name]

        # --- Output comparison table ---
        logger.report_multicolumn_table(
            headers=["input", "single_hf", "batch_hf", "single_vllm", "batch_vllm"],
            rows=[
                [
                    batch_hf_dataset[i][self.dataset_info.content_column_name],
                    single_hf_results[i],
                    batch_hf_results[i],
                    single_vllm_results[i],
                    batch_vllm_results[i],
                ]
                for i in range(len(dataset))
            ],
        )
