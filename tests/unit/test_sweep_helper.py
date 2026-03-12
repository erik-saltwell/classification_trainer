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


def test_sweep_run_config_overrides_base_learning_rate() -> None:
    """wandb run.config from a sweep contains the full training_info dict
    (from wandb.init) plus the sweep-chosen parameter at the top level.
    The sweep's top-level learning_rate (0.0005) must override the base
    sft_parameters learning_rate (0.0006)."""
    run_config: dict = {
        "training_info": {
            "evaluation_enabled": True,
            "training_length_type": "epoch",
            "model_name": "reddit-rpg-rules-questions-classifier",
            "evaluation_steps": 5,
            "sft_parameters": {
                "lr_scheduler_type": "linear",
                "use_loftq": False,
                "rank": 16,
                "rslora": False,
                "use_projection_modules": True,
                "learning_rate": 0.0006,
                "loftq_iter": 1,
                "weight_decay": 0.01,
                "warmup_ratio": 0.05,
                "lora_dropout": 0,
                "alpha_multiplier": 2,
                "optim": "adamw_bnb_8bit",
                "loftq_bits": 4,
            },
            "sweep_config": {
                "metric_goal": "maximize",
                "sweep_name": "learning-rate-fine-6e-4@2.5e-5",
                "description": "test learning rate around 0.0006 with 0.000025 increments",
                "run_cap": 10,
                "parameters": {
                    "learning_rate": {
                        "min": None,
                        "max": None,
                        "value": None,
                        "values": [0.0005, 0.000525, 0.00055, 0.000575, 0.0005, 0.0006],
                    }
                },
                "method": "grid",
                "metric": "f1",
            },
            "system_prompt_name": "rpg_post_classification_prompt.md",
            "publishing": "lora_merged_gguf_4_and_8",
            "train_on_outputs_only": True,
            "per_device_batch_size": 32,
            "wandb_project_name": "reddit-rpg-rules-questions-classifier",
            "model_card_description": (
                "Binary text classifier fine-tuned with LoRA on a custom dataset. "
                "This model was trained to distinguish rules questions from other kinds "
                "of posts in rpg-related subreddits.\n"
            ),
            "seed": 3414,
            "gradient_accumulation_steps": 2,
            "inference": "simple-classification",
            "load_in_4bit": True,
            "hugging_face_user_name": "eriksalt",
            "max_sequence_length": 1536,
            "base_model": "qwen2.5-7B-instruct-bnb-4bit",
            "dtype": None,
            "packing": False,
            "training_length": 3,
            "greater_is_better": False,
        },
        "learning_rate": 0.0005,
    }

    base = _make_training_info(learning_rate=0.0006)
    result = apply_trial_sft_parameters(base, run_config)
    assert result.sft_parameters.learning_rate == 0.0005
