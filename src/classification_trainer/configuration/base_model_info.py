"""Base model definitions, format separators, and registry lookup."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from classification_trainer.utils import FragmentID


class BaseModelName(StrEnum):
    """Identifiers for supported base models from HuggingFace."""

    QWEN_25_14B_4BIT_BASE = "unsloth/Qwen2.5-14B-bnb-4bit"
    QWEN_25_14B_4BIT_INSTRUCT = "unsloth/Qwen2.5-14B-Instruct-bnb-4bit"
    QWEN_25_3B_4BIT_INSTRUCT = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
    QWEN_25_3B_05BIT_INSTRUCT = "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"
    QWEN_25_1_5B_INSTRUCT = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
    NONE = "none"


class InstructionSeperator(StrEnum):
    """Chat-template tokens that mark the beginning of a user instruction."""

    CHAT_ML = "<|im_start|>user\n"
    LLAMA = "<|start_header_id|>user<|end_header_id|>\n\n"
    MISTRAL = "[INST]"
    GEMMA = "<start_of_turn>user\n"
    PHI = "<|user|>\n"


class ResponseSeperator(StrEnum):
    """Chat-template tokens that mark the beginning of an assistant response."""

    CHAT_ML = "<|im_start|>assistant\n"
    LLAMA = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    MISTRAL = "[/INST]"
    GEMMA = "<start_of_turn>model\n"
    PHI = "<|assistant|>\n"


@dataclass(frozen=True)
class BaseModelInfo:
    """Configuration for a base model including chat-template separators and prompt fragments.

    Attributes:
        is_instruct: Whether the model is an instruction-tuned variant.
        instruction_seperator: Token sequence that precedes user instructions, if applicable.
        response_seperator: Token sequence that precedes assistant responses, if applicable.
        training_fragment_id: Text fragment used to format training prompts for non-instruct models.
        eval_fragment_id: Text fragment used to format evaluation prompts for non-instruct models.
    """

    is_instruct: bool
    instruction_seperator: str | None = None
    response_seperator: str | None = None
    training_fragment_id: FragmentID | None = None
    eval_fragment_id: FragmentID | None = None


_base_model_registry: dict[BaseModelName, BaseModelInfo] = {
    BaseModelName.QWEN_25_1_5B_INSTRUCT: BaseModelInfo(
        is_instruct=True,
        instruction_seperator=InstructionSeperator.CHAT_ML,
        response_seperator=ResponseSeperator.CHAT_ML,
    ),
    BaseModelName.QWEN_25_14B_4BIT_BASE: BaseModelInfo(
        is_instruct=False,
        training_fragment_id=FragmentID.ALPACA_PROMPT_TEMPLATE,
        eval_fragment_id=FragmentID.ALPACA_PROMPT_TEMPLATE,
    ),
    BaseModelName.QWEN_25_14B_4BIT_INSTRUCT: BaseModelInfo(
        is_instruct=True,
        instruction_seperator=InstructionSeperator.CHAT_ML,
        response_seperator=ResponseSeperator.CHAT_ML,
    ),
    BaseModelName.QWEN_25_3B_4BIT_INSTRUCT: BaseModelInfo(
        is_instruct=True,
        instruction_seperator=InstructionSeperator.CHAT_ML,
        response_seperator=ResponseSeperator.CHAT_ML,
    ),
    BaseModelName.QWEN_25_3B_05BIT_INSTRUCT: BaseModelInfo(
        is_instruct=True,
        instruction_seperator=InstructionSeperator.CHAT_ML,
        response_seperator=ResponseSeperator.CHAT_ML,
    ),
}


def get_base_model_info(model_name: BaseModelName) -> BaseModelInfo:
    """Look up the configuration for a base model.

    Args:
        model_name: The base model identifier to look up.

    Returns:
        The corresponding BaseModelInfo with separators and fragment IDs.

    Raises:
        KeyError: If the model name is not registered.
    """
    if model_name not in _base_model_registry:
        raise KeyError(f"No model data registered for {model_name}")
    return _base_model_registry[model_name]
