from dataclasses import dataclass
from pathlib import Path


@dataclass
class CommonPaths:
    """Resolves and ensures standard project directory paths for a given target mode."""

    target_mode_name: str
    INPUTS_DIR: Path = Path("inputs")
    OUTPUTS_DIR: Path = Path("outputs")
    EXPLORATION_REPORTS_DIR: Path = Path("exploration_reports")
    FRAGMENTS_DIR: Path = Path("fragments")
    DATASETS_DIR: Path = Path("dataset")

    def __post_init__(self):
        """Create all required directories on initialization."""
        self.ensure_all_dirs_exist()

    @property
    def computed_datasets(self) -> Path:
        """Return the path to the computed datasets directory under outputs."""
        return self.outputs / CommonPaths.DATASETS_DIR

    @property
    def inputs(self) -> Path:
        """Return the inputs directory path for the current target mode."""
        return CommonPaths.INPUTS_DIR / self.target_mode_name

    @property
    def outputs(self) -> Path:
        """Return the outputs directory path for the current target mode."""
        return CommonPaths.OUTPUTS_DIR / self.target_mode_name

    @property
    def exploration_reports(self) -> Path:
        """Return the exploration reports directory path under outputs."""
        return self.outputs / CommonPaths.EXPLORATION_REPORTS_DIR

    @property
    def fragments(self) -> Path:
        """Return the shared fragments directory path."""
        return CommonPaths.FRAGMENTS_DIR

    def ensure_all_dirs_exist(self) -> None:
        """Create all project directories if they don't already exist."""
        self.inputs.mkdir(parents=True, exist_ok=True)
        self.outputs.mkdir(parents=True, exist_ok=True)
        self.exploration_reports.mkdir(parents=True, exist_ok=True)
        self.computed_datasets.mkdir(parents=True, exist_ok=True)
