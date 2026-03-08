from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CommonPaths:
    """Resolves and ensures standard project directory paths for a given target mode."""

    EXPLORATION_REPORTS_DIR: Path = Path("exploration_reports")
    FRAGMENTS_DIR: Path = Path("fragments")
    DATASETS_DIR: Path = Path("dataset_info")
    BASE_MODEL_INFO_DIR: Path = Path("base_model_info")
    TRAINING_INFO_DIR: Path = Path("training_info")
    CHAT_TEMPLATE_INFO_DIR: Path = Path("chat_template_info")
    INFERENCE_INFO_DIR: Path = Path("inference_info")
    PUBLISHING_INFO_DIR: Path = Path("publishing_info")
    OUTPUT_MODELS_DIR: Path = Path("output_models")

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
    def fragments(self) -> Path:
        """Return the shared fragments directory path."""
        return CommonPaths.FRAGMENTS_DIR

    def get_model_checkpoint_directory(self, model_name: str) -> Path:
        return self.OUTPUT_MODELS_DIR / Path("checkpoints") / Path(model_name)

    def get_model_sweeps_directory(self, model_name: str) -> Path:
        return self.OUTPUT_MODELS_DIR / Path("sweeps") / Path(model_name)

    def get_model_save_directory(self, model_name: str, save_format: str) -> Path:
        return self.get_model_save_root_directory(model_name) / Path(save_format)

    def get_model_save_root_directory(self, model_name: str) -> Path:
        return self.OUTPUT_MODELS_DIR / Path("saves") / Path(model_name)

    def ensure_directory(self, dir: Path) -> None:
        if not dir.is_dir():
            raise ValueError("ensure_direcory called on non-directory")
        dir.mkdir(parents=True, exist_ok=True)

    def clear_cache_model_directories(self, model_name: str) -> None:
        shutil.rmtree(self.get_model_checkpoint_directory(model_name), ignore_errors=True)
        shutil.rmtree(self.get_model_sweeps_directory(model_name), ignore_errors=True)
        shutil.rmtree(self.get_model_save_root_directory(model_name), ignore_errors=True)

    @staticmethod
    def get() -> CommonPaths:
        """static constructor to create a CommonPaths object."""
        return CommonPaths()
