from attr import dataclass
from datasets import Dataset

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
)
from classification_trainer.helpers.tokenizer_helper import load_tokenizer_from_hf
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
        self.log_row_after_prep(logger)

    def log_row_after_prep(self, logger: LoggingProtocol) -> None:
        dataset: Dataset = load_dataset_from_hf(self.dataset_info)[self.dataset_info.training_split_name]
        tokenizer = load_tokenizer_from_hf(self.base_model_info)
        dataset = prep_classification_dataset_for_training(
            self.dataset_info, self.training_info, dataset, tokenizer, self.chat_template
        )
        dataset = add_eval_column(self.dataset_info, self.training_info, dataset, tokenizer)
        log_dataset(dataset, logger)
