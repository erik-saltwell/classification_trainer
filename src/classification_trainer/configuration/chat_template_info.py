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


class InstructionSeparator(StrEnum):
    CHAT_ML = "<|im_start|>user\n"
    LLAMA = "<|start_header_id|>user<|end_header_id|>\n\n"
    MISTRAL = "[INST]"
    GEMMA = "<start_of_turn>user\n"
    PHI = "<|user|>\n"


class ResponseSeparator(StrEnum):
    """Chat-template tokens that mark the beginning of an assistant response."""

    CHAT_ML = "<|im_start|>assistant\n"
    LLAMA = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    MISTRAL = "[/INST]"
    GEMMA = "<start_of_turn>model\n"
    PHI = "<|assistant|>\n"


class ChatTemplateInfo(NamedTuple):
    instruction_separator: str
    response_separator: str


_chat_templates: dict[ChatTemplateName, ChatTemplateInfo] = {
    ChatTemplateName.CHAT_ML: ChatTemplateInfo(InstructionSeparator.CHAT_ML, ResponseSeparator.CHAT_ML),
    ChatTemplateName.LLAMA: ChatTemplateInfo(InstructionSeparator.LLAMA, ResponseSeparator.LLAMA),
    ChatTemplateName.MISTRAL: ChatTemplateInfo(InstructionSeparator.MISTRAL, ResponseSeparator.MISTRAL),
    ChatTemplateName.GEMMA: ChatTemplateInfo(InstructionSeparator.GEMMA, ResponseSeparator.GEMMA),
    ChatTemplateName.PHI: ChatTemplateInfo(InstructionSeparator.PHI, ResponseSeparator.PHI),
    ChatTemplateName.NONE: ChatTemplateInfo("NONE", "NONE"),
}


def get_chat_template_by_name(template_name: ChatTemplateName) -> ChatTemplateInfo:
    if template_name not in _chat_templates:
        raise KeyError(template_name)
    return _chat_templates[template_name]
