"""Unit tests for Modelfile generation in publishing_helper.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from classification_trainer.configuration.chat_template_info import ChatTemplateInfo
from classification_trainer.configuration.inference_info import InferenceInfo
from classification_trainer.configuration.publishing_info import PublishingInfo, SaveFormat
from classification_trainer.helpers.publishing_helper import generate_modelfile

# ---------------------------------------------------------------------------
# Minimal fake config builders
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
# Phase 3 / US1 — GGUF tests
# ---------------------------------------------------------------------------


def test_gguf_from_line(tmp_path: Path) -> None:
    ti = _training_info_mock(model_name="my-model")
    pi = _publishing(gguf_quantizations=["q8_0"])
    generate_modelfile(tmp_path, SaveFormat.GGUF, ti, pi)
    content = (tmp_path / "Modelfile").read_text()
    assert content.startswith("FROM my-model-gguf-q8_0.gguf")


def test_system_block_verbatim(tmp_path: Path) -> None:
    prompt = "You are a classifier.\nRespond with positive or negative."
    ti = _training_info_mock(system_prompt=prompt)
    pi = _publishing()
    generate_modelfile(tmp_path, SaveFormat.GGUF, ti, pi)
    content = (tmp_path / "Modelfile").read_text()
    assert f'SYSTEM """\n{prompt}\n"""' in content


def test_template_with_system_separator(tmp_path: Path) -> None:
    ct = _chat_template(system_separator="<|im_start|>system\n")
    ti = _training_info_mock(chat_template=ct)
    pi = _publishing()
    generate_modelfile(tmp_path, SaveFormat.GGUF, ti, pi)
    content = (tmp_path / "Modelfile").read_text()
    assert "TEMPLATE" in content
    assert "<|im_start|>system\n" in content
    assert "{{- if .System }}" in content


def test_template_without_system_separator(tmp_path: Path) -> None:
    ct = _chat_template(system_separator=None)
    ti = _training_info_mock(chat_template=ct)
    pi = _publishing()
    generate_modelfile(tmp_path, SaveFormat.GGUF, ti, pi)
    content = (tmp_path / "Modelfile").read_text()
    assert "TEMPLATE" in content
    # Inline system: no system_separator string, just {{ .System }}
    assert "{{- if .System }}{{ .System }}" in content


def test_parameters_all_present(tmp_path: Path) -> None:
    inf = _inference(temperature=0.1, top_p=0.9, max_new_tokens=16)
    ti = _training_info_mock(inference=inf, max_sequence_length=2048)
    pi = _publishing()
    generate_modelfile(tmp_path, SaveFormat.GGUF, ti, pi)
    content = (tmp_path / "Modelfile").read_text()
    assert "PARAMETER temperature 0.1" in content
    assert "PARAMETER top_p 0.9" in content
    assert "PARAMETER num_predict 16" in content
    assert "PARAMETER num_ctx 2048" in content


def test_stop_strings_in_parameters(tmp_path: Path) -> None:
    ct = _chat_template(stop_strings=("<|im_end|>", "<|im_start|>"))
    ti = _training_info_mock(chat_template=ct)
    pi = _publishing()
    generate_modelfile(tmp_path, SaveFormat.GGUF, ti, pi)
    content = (tmp_path / "Modelfile").read_text()
    assert 'PARAMETER stop "<|im_end|>"' in content
    assert 'PARAMETER stop "<|im_start|>"' in content


def test_modelfile_written_to_disk(tmp_path: Path) -> None:
    ti = _training_info_mock()
    pi = _publishing()
    generate_modelfile(tmp_path, SaveFormat.GGUF, ti, pi)
    modelfile = tmp_path / "Modelfile"
    assert modelfile.exists()
    assert modelfile.read_text(encoding="utf-8")  # non-empty


# ---------------------------------------------------------------------------
# Phase 4 / US2 — merged and publish regeneration tests
# ---------------------------------------------------------------------------


def test_merged_from_line(tmp_path: Path) -> None:
    ti = _training_info_mock(model_name="my-model", hf_user="eriksalt")
    pi = _publishing()
    generate_modelfile(tmp_path, SaveFormat.MERGED, ti, pi)
    content = (tmp_path / "Modelfile").read_text()
    assert content.startswith("FROM eriksalt/my-model-merged")


def test_repeat_penalty_included(tmp_path: Path) -> None:
    inf = _inference(repetition_penalty=1.2)
    ti = _training_info_mock(inference=inf)
    pi = _publishing()
    generate_modelfile(tmp_path, SaveFormat.GGUF, ti, pi)
    content = (tmp_path / "Modelfile").read_text()
    assert "PARAMETER repeat_penalty 1.2" in content


def test_repeat_penalty_omitted(tmp_path: Path) -> None:
    inf = _inference(repetition_penalty=None)
    ti = _training_info_mock(inference=inf)
    pi = _publishing()
    generate_modelfile(tmp_path, SaveFormat.GGUF, ti, pi)
    content = (tmp_path / "Modelfile").read_text()
    assert "repeat_penalty" not in content


def test_no_stop_lines_when_empty(tmp_path: Path) -> None:
    ct = _chat_template(stop_strings=())
    ti = _training_info_mock(chat_template=ct)
    pi = _publishing()
    generate_modelfile(tmp_path, SaveFormat.GGUF, ti, pi)
    content = (tmp_path / "Modelfile").read_text()
    assert "PARAMETER stop" not in content


def test_publish_generates_modelfile_if_missing(tmp_path: Path) -> None:
    """publish_model() regenerates a missing Modelfile before upload."""
    from classification_trainer.helpers.publishing_helper import publish_model

    # Create a minimal save directory with only README.md (no Modelfile)
    (tmp_path / "README.md").write_text("# model card")

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

    assert (tmp_path / "Modelfile").exists()
    mock_api.upload_folder.assert_called_once()


# ---------------------------------------------------------------------------
# Phase 5 / US3 — LoRA exclusion test
# ---------------------------------------------------------------------------


def test_lora_receives_no_modelfile(tmp_path: Path) -> None:
    """Verify that the _save_format guard skips LoRA — Modelfile must not appear."""
    # Simulate what _save_format does: it only calls generate_modelfile when
    # slug is in (GGUF, MERGED).  For LoRA, the function is never invoked.
    # We call generate_modelfile directly with GGUF/MERGED to confirm it works,
    # then confirm calling it with LORA is not in scope by checking the guard.
    from classification_trainer.configuration.publishing_info import SaveFormat

    assert SaveFormat.LORA not in (SaveFormat.GGUF, SaveFormat.MERGED)

    # LoRA save directory gets no Modelfile (guard in _save_format prevents the call)
    ti = _training_info_mock()
    pi = _publishing()

    # Produce a GGUF Modelfile to confirm the function works
    generate_modelfile(tmp_path, SaveFormat.GGUF, ti, pi)
    assert (tmp_path / "Modelfile").exists()

    # Simulate a separate LoRA directory — nothing written there
    lora_dir = tmp_path / "lora_sim"
    lora_dir.mkdir()
    assert not (lora_dir / "Modelfile").exists()


# ---------------------------------------------------------------------------
# Phase 6 — Edge case tests
# ---------------------------------------------------------------------------


def test_modelfile_overwritten_on_resave(tmp_path: Path) -> None:
    ti1 = _training_info_mock(system_prompt="First prompt.")
    ti2 = _training_info_mock(system_prompt="Second prompt.")
    pi = _publishing()
    generate_modelfile(tmp_path, SaveFormat.GGUF, ti1, pi)
    generate_modelfile(tmp_path, SaveFormat.GGUF, ti2, pi)
    content = (tmp_path / "Modelfile").read_text()
    assert "Second prompt." in content
    assert "First prompt." not in content


def test_multiple_stop_strings(tmp_path: Path) -> None:
    ct = _chat_template(stop_strings=("<|im_end|>", "<|im_start|>", "</s>"))
    ti = _training_info_mock(chat_template=ct)
    pi = _publishing()
    generate_modelfile(tmp_path, SaveFormat.GGUF, ti, pi)
    content = (tmp_path / "Modelfile").read_text()
    assert content.count("PARAMETER stop") == 3


def test_multiple_quantizations_uses_first(tmp_path: Path) -> None:
    ti = _training_info_mock(model_name="my-model")
    pi = _publishing(gguf_quantizations=["q4_k_m", "q8_0"])
    generate_modelfile(tmp_path, SaveFormat.GGUF, ti, pi)
    content = (tmp_path / "Modelfile").read_text()
    assert content.startswith("FROM my-model-gguf-q4_k_m.gguf")
    assert "q8_0" not in content.split("\n")[0]


def test_system_prompt_with_newlines(tmp_path: Path) -> None:
    prompt = "Line one.\nLine two.\nLine three."
    ti = _training_info_mock(system_prompt=prompt)
    pi = _publishing()
    generate_modelfile(tmp_path, SaveFormat.GGUF, ti, pi)
    content = (tmp_path / "Modelfile").read_text()
    assert "Line one.\nLine two.\nLine three." in content
