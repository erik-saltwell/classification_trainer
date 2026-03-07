from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table, box
from rich.traceback import Traceback

from classification_trainer.protocols.logging_protocol import (
    LoggingProtocol,
    ProgressTask,
    StatusHandle,
    _NullProgress,
    _NullStatus,
)


class FileLogger(LoggingProtocol):
    """LoggingProtocol implementation that writes plain text to a file.

    Uses Rich Console targeting a file handle to get the same table formatting
    and markup stripping as the console logger, minus colors and interactive elements.
    """

    def __init__(self, filename: str | Path) -> None:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w")
        self._console = Console(
            file=self._file,
            no_color=True,
            force_terminal=False,
            width=120,
        )

    # ----- messages -----

    def report_message(self, message: str) -> None:
        self._console.print(message)

    def report_warning(self, message: str) -> None:
        self._console.print(f"[yellow]WARNING[/yellow] {message}")

    def report_error(self, message: str) -> None:
        self._console.print(f"[red]ERROR[/red] {message}")

    def report_exception(self, context: str, exc: BaseException) -> None:
        self._console.print(f"[red]EXCEPTION[/red] {context}")
        tb = Traceback.from_exception(type(exc), exc, exc.__traceback__)
        self._console.print(tb)

    def report_table_message(self, row_data: dict[str, Any]) -> None:
        table = Table(
            show_header=True,
            show_lines=True,
            box=box.SQUARE,
        )
        table.add_column("Key")
        table.add_column("Value")
        for key, value in row_data.items():
            table.add_row(str(key), str(value))
        self._console.print(table)

    def report_multicolumn_table(self, headers: list[str], rows: list[list[str]]) -> None:
        table = Table(
            show_header=True,
            show_lines=True,
            box=box.SQUARE,
        )
        for header in headers:
            table.add_column(header)
        for row in rows:
            table.add_row(*row)
        self._console.print(table)

    def add_break(self, break_count: int = 1) -> None:
        for _ in range(break_count):
            self._console.print("")

    # ----- status/progress (no-ops for file output) -----

    @contextmanager
    def status(self, message: str) -> Iterator[StatusHandle]:
        yield _NullStatus()

    @contextmanager
    def progress(self, description: str, total: int | None = None) -> Iterator[ProgressTask]:
        yield _NullProgress()
