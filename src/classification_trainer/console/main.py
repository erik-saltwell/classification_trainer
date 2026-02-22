# src/reddit_rpg_miner/cli/main.py

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, metadata
from importlib.metadata import version as dist_version
from typing import Annotated

import typer
from dotenv import load_dotenv
from pydantic import ValidationError
from rich.console import Console

from classification_trainer.utils.logging_config import configure_logging

load_dotenv()
configure_logging()

app = typer.Typer(
    name="classification-trainer",
    add_completion=True,
    help="CLI for classification-trainer",
)


@app.command("analyze-sequence-length")
def analyze_sequence_length(
    dataset_info: Annotated[str, typer.Argument(help="Dataset info yaml name (no extension)")],
    base_model_info: Annotated[str, typer.Argument(help="Base model info yaml name (no extension)")],
) -> None:
    """Analyze token sequence lengths for a dataset using a model's tokenizer."""
    from classification_trainer.configuration import load_base_model_info, load_dataset_info

    console = Console()

    try:
        ds_info = load_dataset_info(dataset_info)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1) from None
    except ValidationError as e:
        console.print(f"[red]Invalid dataset info '{dataset_info}':[/red]\n{e}")
        raise typer.Exit(code=1) from None

    try:
        bm_info = load_base_model_info(base_model_info)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1) from None
    except ValidationError as e:
        console.print(f"[red]Invalid base model info '{base_model_info}':[/red]\n{e}")
        raise typer.Exit(code=1) from None

    console.print(f"[green]dataset_info:[/green] {ds_info}")
    console.print(f"[green]base_model_info:[/green] {bm_info}")


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
