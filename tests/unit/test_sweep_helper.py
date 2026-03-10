from __future__ import annotations

from classification_trainer.configuration.sft_parameters import SFTParameters
from classification_trainer.configuration.training_info import TrainingInfo, TrainingLengthType
from classification_trainer.helpers.sweep_helper import apply_trial_sft_parameters
from classification_trainer.utils.text_fragments import FragmentID


def _make_training_info(**sft_overrides: object) -> TrainingInfo:
    """Minimal TrainingInfo for testing apply_trial_sft_parameters."""
    sft = SFTParameters(**sft_overrides)  # type: ignore[arg-type]
    return TrainingInfo(
        model_name="test-model",
        hugging_face_user_name="test-user",
        system_prompt_name=FragmentID.NONE,
        base_model="test-base",
        inference="test-inference",
        model_card_description="test",
        training_length_type=TrainingLengthType.EPOCHS,
        training_length=1.0,
        max_sequence_length=512,
        per_device_batch_size=4,
        gradient_accumulation_steps=4,
        sft_parameters=sft,
    )


# ---------------------------------------------------------------------------
# apply_trial_sft_parameters — merge behaviour
# ---------------------------------------------------------------------------


def test_single_swept_field_overrides_base() -> None:
    base = _make_training_info(rank=16, learning_rate=2e-4)
    result = apply_trial_sft_parameters(base, {"rank": 32})
    assert result.sft_parameters.rank == 32
    assert result.sft_parameters.learning_rate == 2e-4  # base value preserved


def test_multiple_swept_fields_all_override() -> None:
    base = _make_training_info(rank=16, learning_rate=2e-4, warmup_ratio=0.05)
    result = apply_trial_sft_parameters(base, {"rank": 64, "learning_rate": 5e-5})
    assert result.sft_parameters.rank == 64
    assert result.sft_parameters.learning_rate == 5e-5
    assert result.sft_parameters.warmup_ratio == 0.05  # base value preserved


def test_empty_trial_config_leaves_base_unchanged() -> None:
    base = _make_training_info(rank=16, learning_rate=2e-4)
    result = apply_trial_sft_parameters(base, {})
    assert result.sft_parameters.rank == 16
    assert result.sft_parameters.learning_rate == 2e-4


def test_result_is_a_new_training_info_copy() -> None:
    base = _make_training_info(rank=16)
    result = apply_trial_sft_parameters(base, {"rank": 32})
    # original must be unchanged
    assert base.sft_parameters.rank == 16
    assert result.sft_parameters.rank == 32
