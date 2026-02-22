from __future__ import annotations

from enum import StrEnum
from pathlib import Path


class FragmentID(StrEnum):
    """Identifiers for text fragment files stored in the fragments directory."""

    RPG_POST_CLASSIFICATION_PROMPT = "rpg_post_classification_prompt.md"
    ALPACA_PROMPT_TEMPLATE = "chat_template_alpaca.md"
    ALPACA_EVAL_PROMPT_TEMPLATE = "eval_template_alpaca.md"
    IMDB_TEST_PROMPT = "imdb_test_prompt.md"
    NONE = "none.md"


def get_fragment_path(fragment_id: FragmentID) -> Path:
    """Return the directory path where the given fragment file is stored."""
    from classification_trainer.utils.common_paths import CommonPaths

    return CommonPaths.get().fragments


def get_fragment(fragment_id: FragmentID) -> str:
    """Read and return the text content of the specified fragment file."""
    fragment_path = get_fragment_path(fragment_id) / Path(fragment_id.value)
    with open(fragment_path, encoding="utf-8") as f:
        return f.read()
