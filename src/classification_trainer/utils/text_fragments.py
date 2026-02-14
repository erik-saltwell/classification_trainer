from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from .common_paths import CommonPaths


class FragmentID(StrEnum):
    RPG_POST_CLASSIFICATION_PROMPT = "rpg_post_classification_prompt.md"
    ALPACA_PROMPT_TEMPLATE = "chat_template_alpaca.md"
    ALPACA_EVAL_PROMPT_TEMPLATE = "eval_template_alpaca.md"
    IMDB_TEST_PROMPT = "imdb_test_prompt.md"
    NONE = "none"


def get_fragment_path(fragment_id: FragmentID) -> Path:
    common_paths: CommonPaths = CommonPaths("_")
    return common_paths.fragments


def get_fragment(fragment_id: FragmentID) -> str:
    fragment_path = get_fragment_path(fragment_id) / Path(fragment_id.value)
    with open(fragment_path, encoding="utf-8") as f:
        return f.read()
