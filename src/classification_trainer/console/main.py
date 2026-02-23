from __future__ import annotations

import unsloth  # isort: skip  # Must precede all transformers imports

from importlib.metadata import PackageNotFoundError, metadata
from importlib.metadata import version as dist_version
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console

from classification_trainer.commands.analyze_sequence_length import AnalyzeSequenceLengthCommand
from classification_trainer.commands.compute_batch_size import ComputeBatchSizeCommand
from classification_trainer.configuration import load_base_model_info, load_dataset_info, load_training_info
from classification_trainer.utils.logging_config import configure_logging

from .console_validation import load_config_or_exit
from .rich_logging_protocol import RichConsoleLogger

load_dotenv()
configure_logging()
print(unsloth.__version__[0:0])

app = typer.Typer(
    name="classification-trainer",
    add_completion=True,
    help="CLI for classification-trainer",
)


@app.command("analyze-sequence-length")
def analyze_sequence_length(
    dataset_info: Annotated[str, typer.Option("--dataset", help="Dataset info yaml name (no extension)")],
    base_model_info: Annotated[str, typer.Option("--base-model", help="Base model info yaml name (no extension)")],
    training_info: Annotated[str, typer.Option("--training-info", help="Training info yaml name (no extension)")],
    merge_all_splits: Annotated[
        bool, typer.Option("--all-splits", help="Analyze all splits instead of just the training split")
    ] = False,
) -> None:
    """Analyze token sequence lengths for a dataset using a model's tokenizer."""

    console = Console()
    logger: RichConsoleLogger = RichConsoleLogger(console)

    ds_info = load_config_or_exit(load_dataset_info, dataset_info, "dataset info", console)
    bm_info = load_config_or_exit(load_base_model_info, base_model_info, "base model info", console)
    tr_info = load_config_or_exit(load_training_info, training_info, "training info", console)

    AnalyzeSequenceLengthCommand(
        dataset_info=ds_info,
        base_model_info=bm_info,
        training_info=tr_info,
        merge_all_splits=merge_all_splits,
    ).execute(logger=logger)


@app.command("compute-batch-size")
def compute_batch_size(
    dataset_info: Annotated[str, typer.Option("--dataset", help="Dataset info yaml name (no extension)")],
    base_model_info: Annotated[str, typer.Option("--base-model", help="Base model info yaml name (no extension)")],
    training_info: Annotated[str, typer.Option("--training-info", help="Training info yaml name (no extension)")],
    stress_set_rowcount: Annotated[
        int, typer.Option("--stress-set-rowcount", help="Number of rows (longest sequences) to use for stress testing")
    ] = 100,
) -> None:
    """Find the largest batch size that fits in GPU memory."""

    console = Console()
    logger: RichConsoleLogger = RichConsoleLogger(console)

    ds_info = load_config_or_exit(load_dataset_info, dataset_info, "dataset info", console)
    bm_info = load_config_or_exit(load_base_model_info, base_model_info, "base model info", console)
    tr_info = load_config_or_exit(load_training_info, training_info, "training info", console)

    ComputeBatchSizeCommand(
        dataset_info=ds_info,
        base_model_info=bm_info,
        training_info=tr_info,
        stress_set_rowcount=stress_set_rowcount,
    ).execute(logger=logger)


@app.command("test")
def test() -> None:
    """Simple smoke command."""
    console = Console()
    console.print("[green]Hello from test[/green]")


def _version_callback(value: bool) -> None:
    """Print version and exit."""
    if not value:
        return

    # IMPORTANT: distribution name (pyproject.toml [project].name), often hyphenated.
    # Example: "my-tool" even if your import package is "my_tool".
    DIST_NAME = "classification-trainer"

    console = Console()

    try:
        pkg_version = dist_version(DIST_NAME)
        md = metadata(DIST_NAME)
        try:
            pkg_name = md["Name"]
        except KeyError:
            pkg_name = DIST_NAME

        console.print(f"{pkg_name} {pkg_version}")
    except PackageNotFoundError:
        # Running from source without an installed distribution
        console.print(f"{DIST_NAME} 0.0.0+unknown")

    raise typer.Exit()


@app.callback()
def _callback(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Root command group for reddit_rpg_miner."""
    # Intentionally empty: this forces Typer to keep subcommands like `test`.
    pass


if __name__ == "__main__":
    app()
