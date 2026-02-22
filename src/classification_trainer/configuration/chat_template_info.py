from __future__ import annotations

from enum import StrEnum
from typing import NamedTuple


class ChatTemplateName(StrEnum):
    CHAT_ML = "CHAT_ML"
    LLAMA = "LLAMA"
    MISTRAL = "MISTRAL"
    GEMMA = "GEMMA"
    PHI = "PHI"
    NONE = "NONE"


class InstructionSeperator(StrEnum):
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


class ChatTemplateInfo(NamedTuple):
    instruction_seperator: str
    response_seperator: str


_chat_templates: dict[ChatTemplateName, ChatTemplateInfo] = {
    ChatTemplateName.CHAT_ML: ChatTemplateInfo(InstructionSeperator.CHAT_ML, ResponseSeperator.CHAT_ML),
    ChatTemplateName.LLAMA: ChatTemplateInfo(InstructionSeperator.LLAMA, ResponseSeperator.LLAMA),
    ChatTemplateName.MISTRAL: ChatTemplateInfo(InstructionSeperator.MISTRAL, ResponseSeperator.MISTRAL),
    ChatTemplateName.GEMMA: ChatTemplateInfo(InstructionSeperator.GEMMA, ResponseSeperator.GEMMA),
    ChatTemplateName.PHI: ChatTemplateInfo(InstructionSeperator.PHI, ResponseSeperator.PHI),
    ChatTemplateName.NONE: ChatTemplateInfo("NONE", "NONE"),
}


def get_chat_template_by_name(template_name: ChatTemplateName) -> ChatTemplateInfo:
    if template_name not in _chat_templates:
        raise KeyError(template_name)
    return _chat_templates[template_name]
