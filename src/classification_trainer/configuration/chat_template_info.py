"""Chat template definitions and YAML loader."""

from __future__ import annotations

import yaml
from pydantic import BaseModel, ConfigDict
from transformers import PreTrainedTokenizerBase

from classification_trainer.utils.common_paths import CommonPaths


class ChatTemplateInfo(BaseModel):
    """Validated configuration for a chat template."""

    model_config = ConfigDict(frozen=True)

    instruction_separator: str
    response_separator: str
    stop_strings: tuple[str, ...] = ()
    eos_token_strings: tuple[str, ...] = ()
    add_special_tokens: bool = False
    assistant_newline: bool = True

    def get_eos_token_ids(self, tokenizer: PreTrainedTokenizerBase) -> list[int]:
        ids: list[int] = []
        for s in self.eos_token_strings:
            tid: int = tokenizer.convert_tokens_to_ids(s)  # pyright: ignore
            if tid != -1:
                ids.append(tid)
        eos_id: int | None = getattr(tokenizer, "eos_token_id", None)
        if eos_id is not None:
            ids.append(eos_id)
        # de-dupe preserving order
        seen = set()
        out = []
        for x in ids:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def trim_at_stop_strings(self, text: str) -> str:
        stop_strings = self.stop_strings
        cut = None
        for s in stop_strings:
            idx = text.find(s)
            if idx != -1:
                cut = idx if cut is None else min(cut, idx)
        return text[:cut].strip() if cut is not None else text.strip()


def load_chat_template_info(name: str) -> ChatTemplateInfo:
    """Load a ChatTemplateInfo from a YAML file in the chat_template_info directory.

    Args:
        name: The yaml filename without extension (e.g. "chat-ml").

    Returns:
        A validated ChatTemplateInfo instance.

    Raises:
        FileNotFoundError: If the yaml file does not exist.
        pydantic.ValidationError: If the file data fails validation.
    """
    yaml_path = CommonPaths.get().chat_template_info / f"{name}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Chat template info file not found: {yaml_path}")
    with yaml_path.open() as f:
        data = yaml.safe_load(f)
    return ChatTemplateInfo(**data)
