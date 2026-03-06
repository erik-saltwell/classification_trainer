from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CommonPaths:
    """Resolves and ensures standard project directory paths for a given target mode."""

    OUTPUTS_DIR: Path = Path("outputs")
    EXPLORATION_REPORTS_DIR: Path = Path("exploration_reports")
    FRAGMENTS_DIR: Path = Path("fragments")
    DATASETS_DIR: Path = Path("dataset_info")
    BASE_MODEL_INFO_DIR: Path = Path("base_model_info")
    TRAINING_INFO_DIR: Path = Path("training_info")
    CHAT_TEMPLATE_INFO_DIR: Path = Path("chat_template_info")
    INFERENCE_INFO_DIR: Path = Path("inference_info")
    PUBLISHING_INFO_DIR: Path = Path("publishing_info")
    OUTPUT_MODELS_DIR: Path = Path("output_models")

    def __post_init__(self):
        """Create all required directories on initialization."""
        self.ensure_all_dirs_exist()

    @property
    def dataset_info(self) -> Path:
        """Return the path to the computed datasets directory under outputs."""
        return CommonPaths.DATASETS_DIR

    @property
    def base_model_info(self) -> Path:
        return CommonPaths.BASE_MODEL_INFO_DIR

    @property
    def training_info(self) -> Path:
        return CommonPaths.TRAINING_INFO_DIR

    @property
    def chat_template_info(self) -> Path:
        return CommonPaths.CHAT_TEMPLATE_INFO_DIR

    @property
    def inference_info(self) -> Path:
        return CommonPaths.INFERENCE_INFO_DIR

    @property
    def publishing_info(self) -> Path:
        return CommonPaths.PUBLISHING_INFO_DIR

    @property
    def output_models(self) -> Path:
        return CommonPaths.OUTPUT_MODELS_DIR

    def outputs(self, target_model_name: Path) -> Path:
        """Return the outputs directory path for the current target mode."""
        return CommonPaths.OUTPUTS_DIR / target_model_name

    def sweep_trial_outputs(self, model_name: str, run_id: str) -> Path:
        """Return the per-trial checkpoint directory for a sweep run.

        Not auto-created by ensure_all_dirs_exist() — runtime output only.
        """
        return CommonPaths.OUTPUTS_DIR / model_name / run_id

    @property
    def fragments(self) -> Path:
        """Return the shared fragments directory path."""
        return CommonPaths.FRAGMENTS_DIR

    def ensure_all_dirs_exist(self) -> None:
        """Create all project directories if they don't already exist."""
        self.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        self.dataset_info.mkdir(parents=True, exist_ok=True)
        self.base_model_info.mkdir(parents=True, exist_ok=True)
        self.training_info.mkdir(parents=True, exist_ok=True)
        self.chat_template_info.mkdir(parents=True, exist_ok=True)
        self.inference_info.mkdir(parents=True, exist_ok=True)
        self.publishing_info.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get() -> CommonPaths:
        """static constructor to create a CommonPaths object."""
        return CommonPaths()
