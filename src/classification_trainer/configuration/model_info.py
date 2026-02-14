from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from classification_trainer.utils import FragmentID


class BaseModelName(StrEnum):
    QWEN_25_14B_4BIT_BASE = "unsloth/Qwen2.5-14B-bnb-4bit"
    QWEN_25_14B_4BIT_INSTRUCT = "unsloth/Qwen2.5-14B-Instruct-bnb-4bit"
    QWEN_25_3B_4BIT_INSTRUCT = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
    QWEN_25_3B_05BIT_INSTRUCT = "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"
    QWEN_25_1_5B_INSTRUCT = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
    NONE = "none"


class TargetModelName(StrEnum):
    REDDIT_RPG_POST_CLASSIFICATION = "reddit_rpg_post_classification"
    IMDB_TEST = "imdb_sentiment_test"
    NONE = "none"


class InstructionSeperator(StrEnum):
    CHAT_ML = "<|im_start|>user\n"
    LLAMA = "<|start_header_id|>user<|end_header_id|>\n\n"
    MISTRAL = "[INST]"
    GEMMA = "<start_of_turn>user\n"
    PHI = "<|user|>\n"


class ResponseSeperator(StrEnum):
    CHAT_ML = "<|im_start|>assistant\n"
    LLAMA = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    MISTRAL = "[/INST]"
    GEMMA = "<start_of_turn>model\n"
    PHI = "<|assistant|>\n"


@dataclass(frozen=True)
class ModelInfo:
    is_instruct: bool
    instruction_seperator: str | None = None
    response_seperator: str | None = None
    training_fragment_id: FragmentID | None = None
    eval_fragment_id: FragmentID | None = None


_model_registry: dict[BaseModelName, ModelInfo] = {
    BaseModelName.QWEN_25_1_5B_INSTRUCT: ModelInfo(
        is_instruct=True,
        instruction_seperator=InstructionSeperator.CHAT_ML,
        response_seperator=ResponseSeperator.CHAT_ML,
    ),
    BaseModelName.QWEN_25_14B_4BIT_BASE: ModelInfo(
        is_instruct=False,
        training_fragment_id=FragmentID.ALPACA_PROMPT_TEMPLATE,
        eval_fragment_id=FragmentID.ALPACA_PROMPT_TEMPLATE,
    ),
    BaseModelName.QWEN_25_14B_4BIT_INSTRUCT: ModelInfo(
        is_instruct=True,
        instruction_seperator=InstructionSeperator.CHAT_ML,
        response_seperator=ResponseSeperator.CHAT_ML,
    ),
    BaseModelName.QWEN_25_3B_4BIT_INSTRUCT: ModelInfo(
        is_instruct=True,
        instruction_seperator=InstructionSeperator.CHAT_ML,
        response_seperator=ResponseSeperator.CHAT_ML,
    ),
    BaseModelName.QWEN_25_3B_05BIT_INSTRUCT: ModelInfo(
        is_instruct=True,
        instruction_seperator=InstructionSeperator.CHAT_ML,
        response_seperator=ResponseSeperator.CHAT_ML,
    ),
}


def get_model_info(model_name: BaseModelName) -> ModelInfo:
    if model_name not in _model_registry:
        raise KeyError(f"No model data registered for {model_name}")
    return _model_registry[model_name]
