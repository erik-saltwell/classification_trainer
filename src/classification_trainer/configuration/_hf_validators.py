"""Shared HuggingFace repository name validation."""

from __future__ import annotations

import re

# HuggingFace Hub: each segment is 1-96 chars, only [a-zA-Z0-9_\-.],
# no leading/trailing - or ., no .git suffix.
HF_SEGMENT_RE = re.compile(r"^(?![-.])[a-zA-Z0-9_\-.]{1,96}(?<![-.])$")
HF_SEGMENT_FORBIDDEN = re.compile(r"\.\.|--")


def validate_hf_name(v: str) -> str:
    """Validate a HuggingFace repo identifier (owner/repo or repo)."""
    parts = v.split("/")
    if len(parts) > 2:
        raise ValueError("huggingface_name may contain at most one '/' (owner/repo)")
    for part in parts:
        if part.endswith(".git"):
            raise ValueError(f"Invalid huggingface_name segment '{part}': must not end with .git")
        if not HF_SEGMENT_RE.match(part):
            raise ValueError(
                f"Invalid huggingface_name segment '{part}': must be 1-96 chars, "
                "only [a-zA-Z0-9_-.], not start/end with - or ."
            )
        if HF_SEGMENT_FORBIDDEN.search(part):
            raise ValueError(f"Invalid huggingface_name segment '{part}': must not contain -- or ..")
    return v
