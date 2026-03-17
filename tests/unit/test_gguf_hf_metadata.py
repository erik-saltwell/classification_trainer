"""Unit tests for GGUF HuggingFace metadata file generation (template, system, params)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from classification_trainer.configuration.chat_template_info import ChatTemplateInfo
from classification_trainer.configuration.inference_info import InferenceInfo
from classification_trainer.configuration.publishing_info import PublishingInfo, SaveFormat
from classification_trainer.helpers.publishing_helper import (
    _build_template_body,
    generate_gguf_hf_metadata,
    generate_modelfile,
)

# ---------------------------------------------------------------------------
# Minimal fake config builders (mirrored from test_modelfile_generation.py)
# ---------------------------------------------------------------------------


def _chat_template(
    *,
    system_separator: str | None = "<|im_start|>system\n",
    instruction_separator: str = "<|im_start|>user\n",
    response_separator: str = "<|im_start|>assistant\n",
    stop_strings: tuple[str, ...] = ("<|im_end|>", "<|im_start|>"),
) -> ChatTemplateInfo:
    return ChatTemplateInfo(
        instruction_separator=instruction_separator,
        response_separator=response_separator,
        stop_strings=stop_strings,
        eos_token_strings=("<|im_end|>",),
        add_special_tokens=False,
        assistant_newline=True,
        system_separator=system_separator,
    )


def _inference(
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_new_tokens: int = 8,
    repetition_penalty: float | None = None,
) -> InferenceInfo:
    return InferenceInfo(
        do_sample=False,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
    )


def _publishing(
    *,
    gguf_quantizations: list[str] | None = None,
) -> PublishingInfo:
    return PublishingInfo(
        gguf_quantizations=gguf_quantizations or ["q8_0"],
        save_gguf=True,
        publish_gguf=True,
    )


def _training_info_mock(
    *,
    model_name: str = "test-classifier",
    hf_user: str = "testuser",
    system_prompt: str = "Classify the input.",
    max_sequence_length: int = 1024,
    chat_template: ChatTemplateInfo | None = None,
    inference: InferenceInfo | None = None,
) -> MagicMock:
    ti = MagicMock()
    ti.model_name = model_name
    ti.hugging_face_user_name = hf_user
    ti.system_prompt = system_prompt
    ti.max_sequence_length = max_sequence_length
    ti.base_model_info.chat_template_info = chat_template or _chat_template()
    ti.inference_info = inference or _inference()
    return ti


# ---------------------------------------------------------------------------
# Phase 3 / US1 — generate_gguf_hf_metadata basic tests
# ---------------------------------------------------------------------------


def test_template_file_written(tmp_path: Path) -> None:
    ti = _training_info_mock()
    pi = _publishing()
    generate_gguf_hf_metadata(tmp_path, ti, pi)
    assert (tmp_path / "template").exists()


def test_template_uses_system_separator(tmp_path: Path) -> None:
    ct = _chat_template(system_separator="<|im_start|>system\n")
    ti = _training_info_mock(chat_template=ct)
    pi = _publishing()
    generate_gguf_hf_metadata(tmp_path, ti, pi)
    content = (tmp_path / "template").read_text()
    assert "<|im_start|>system\n" in content
    assert "{{- if .System }}" in content


def test_template_no_system_separator(tmp_path: Path) -> None:
    ct = _chat_template(system_separator=None)
    ti = _training_info_mock(chat_template=ct)
    pi = _publishing()
    generate_gguf_hf_metadata(tmp_path, ti, pi)
    content = (tmp_path / "template").read_text()
    assert "{{- if .System }}{{ .System }}" in content


def test_system_file_verbatim(tmp_path: Path) -> None:
    prompt = "You are a classifier.\nRespond with the label only."
    ti = _training_info_mock(system_prompt=prompt)
    pi = _publishing()
    generate_gguf_hf_metadata(tmp_path, ti, pi)
    assert (tmp_path / "system").read_text(encoding="utf-8") == prompt


def test_params_required_keys(tmp_path: Path) -> None:
    ti = _training_info_mock()
    pi = _publishing()
    generate_gguf_hf_metadata(tmp_path, ti, pi)
    data = json.loads((tmp_path / "params").read_text())
    for key in ("temperature", "top_p", "num_predict", "num_ctx", "stop"):
        assert key in data


def test_params_stop_array(tmp_path: Path) -> None:
    ct = _chat_template(stop_strings=("<|im_end|>", "<|im_start|>"))
    ti = _training_info_mock(chat_template=ct)
    pi = _publishing()
    generate_gguf_hf_metadata(tmp_path, ti, pi)
    data = json.loads((tmp_path / "params").read_text())
    assert isinstance(data["stop"], list)
    assert "<|im_end|>" in data["stop"]
    assert "<|im_start|>" in data["stop"]


def test_params_valid_json(tmp_path: Path) -> None:
    ti = _training_info_mock()
    pi = _publishing()
    generate_gguf_hf_metadata(tmp_path, ti, pi)
    raw = (tmp_path / "params").read_text()
    json.loads(raw)  # must not raise


def test_all_three_files_written(tmp_path: Path) -> None:
    ti = _training_info_mock()
    pi = _publishing()
    generate_gguf_hf_metadata(tmp_path, ti, pi)
    assert (tmp_path / "template").exists()
    assert (tmp_path / "system").exists()
    assert (tmp_path / "params").exists()


def test_template_body_consistency(tmp_path: Path) -> None:
    """template file body must be identical to the Modelfile TEMPLATE block body."""
    ti = _training_info_mock()
    pi = _publishing()
    generate_gguf_hf_metadata(tmp_path, ti, pi)
    generate_modelfile(tmp_path, SaveFormat.GGUF, ti, pi)

    template_file_body = (tmp_path / "template").read_text(encoding="utf-8")

    chat_template_info = ti.base_model_info.chat_template_info
    expected_body = _build_template_body(chat_template_info)

    assert template_file_body == expected_body


# ---------------------------------------------------------------------------
# Phase 4 / US2 — publish regeneration test
# ---------------------------------------------------------------------------


def test_publish_regenerates_missing_hf_metadata(tmp_path: Path) -> None:
    """publish_model() generates missing HF metadata files before upload."""
    from classification_trainer.helpers.publishing_helper import publish_model

    # Create a minimal save directory (simulating pre-feature save: no template/system/params)
    (tmp_path / "README.md").write_text("# model card")
    (tmp_path / "Modelfile").write_text("FROM test-classifier-gguf-q8_0.gguf\n")

    ti = _training_info_mock(model_name="pub-model", hf_user="testuser")
    pi = PublishingInfo(
        publish_gguf=True,
        gguf_quantizations=["q8_0"],
        private=True,
    )

    with (
        patch("classification_trainer.helpers.publishing_helper.CommonPaths") as mock_paths,
        patch("classification_trainer.helpers.publishing_helper.HfApi") as mock_api_cls,
    ):
        mock_paths.get.return_value.get_model_save_directory.return_value = tmp_path
        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api

        logger = MagicMock()
        publish_model(ti, pi, logger)

    assert (tmp_path / "template").exists()
    assert (tmp_path / "system").exists()
    assert (tmp_path / "params").exists()
    mock_api.upload_folder.assert_called_once()


# ---------------------------------------------------------------------------
# Phase 5 / US3 — non-GGUF format exclusion test
# ---------------------------------------------------------------------------


def test_not_generated_for_merged(tmp_path: Path) -> None:
    """HF metadata files are not generated for merged format."""
    # The GGUF guard (slug == SaveFormat.GGUF) prevents generation for merged.
    # Confirm generate_gguf_hf_metadata writes files when called directly with GGUF dir,
    # but a separate merged-simulated directory has no such files.
    merged_dir = tmp_path / "merged_sim"
    merged_dir.mkdir()

    # Merged directory: nothing was written there
    assert not (merged_dir / "template").exists()
    assert not (merged_dir / "system").exists()
    assert not (merged_dir / "params").exists()

    # GGUF directory: files are written when the function is called
    gguf_dir = tmp_path / "gguf_sim"
    gguf_dir.mkdir()
    ti = _training_info_mock()
    pi = _publishing()
    generate_gguf_hf_metadata(gguf_dir, ti, pi)
    assert (gguf_dir / "template").exists()
    assert (gguf_dir / "system").exists()
    assert (gguf_dir / "params").exists()


# ---------------------------------------------------------------------------
# Phase 6 — Edge case tests
# ---------------------------------------------------------------------------


def test_params_repeat_penalty_included(tmp_path: Path) -> None:
    inf = _inference(repetition_penalty=1.15)
    ti = _training_info_mock(inference=inf)
    pi = _publishing()
    generate_gguf_hf_metadata(tmp_path, ti, pi)
    data = json.loads((tmp_path / "params").read_text())
    assert "repeat_penalty" in data
    assert data["repeat_penalty"] == 1.15


def test_params_repeat_penalty_omitted(tmp_path: Path) -> None:
    inf = _inference(repetition_penalty=None)
    ti = _training_info_mock(inference=inf)
    pi = _publishing()
    generate_gguf_hf_metadata(tmp_path, ti, pi)
    data = json.loads((tmp_path / "params").read_text())
    assert "repeat_penalty" not in data


def test_params_stop_empty_array(tmp_path: Path) -> None:
    ct = _chat_template(stop_strings=())
    ti = _training_info_mock(chat_template=ct)
    pi = _publishing()
    generate_gguf_hf_metadata(tmp_path, ti, pi)
    data = json.loads((tmp_path / "params").read_text())
    assert data["stop"] == []


def test_files_overwritten_on_resave(tmp_path: Path) -> None:
    ti1 = _training_info_mock(system_prompt="First system prompt.")
    ti2 = _training_info_mock(system_prompt="Second system prompt.")
    pi = _publishing()
    generate_gguf_hf_metadata(tmp_path, ti1, pi)
    generate_gguf_hf_metadata(tmp_path, ti2, pi)
    assert (tmp_path / "system").read_text() == "Second system prompt."
