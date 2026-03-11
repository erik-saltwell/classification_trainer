from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.status import Status
from rich.table import Table, box
from rich.traceback import Traceback

from classification_trainer.protocols import LoggingProtocol, ProgressTask, StatusHandle


class RichConsoleLogger(LoggingProtocol):
    def __init__(self, console: Console | None = None, verbose_training: bool = False) -> None:
        self._console = console or Console()
        self._verbose_training = verbose_training

    @property
    def verbose_training(self) -> bool:
        return self._verbose_training

    # ----- messages -----

    def report_message(self, message: str) -> None:
        self._console.print(message)

    def report_warning(self, message: str) -> None:
        self._console.print(f"[yellow]WARNING[/yellow] {message}")

    def report_error(self, message: str) -> None:
        self._console.print(f"[red]ERROR[/red] {message}")

    def report_exception(self, context: str, exc: BaseException) -> None:
        # Report the *passed* exception with full traceback, regardless of whether we're inside an except block.
        self._console.print(f"[red]EXCEPTION[/red] {context}")
        tb = Traceback.from_exception(type(exc), exc, exc.__traceback__)
        self._console.print(tb)
        # Do NOT raise here; let the caller decide (and preserve traceback with `raise` at call site).

    def report_table_message(self, row_data: dict[str, Any]) -> None:
        table = Table(
            show_header=True,
            show_lines=True,  # <-- row separators
            box=box.SQUARE,  # optional: pick a box style that looks good with row lines
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
            self.report_message("")

    # ----- status/progress -----

    @contextmanager
    def status(self, message: str) -> Iterator[StatusHandle]:
        st = Status(message, console=self._console)
        st.start()

        class _Handle:
            _closed = False

            def update(self, message: str) -> None:
                st.update(message)

            def close(self) -> None:
                if not self._closed:
                    st.stop()
                    self._closed = True

        handle: StatusHandle = _Handle()
        try:
            yield handle
        finally:
            handle.close()

    @contextmanager
    def progress(self, description: str, total: int | None = None) -> Iterator[ProgressTask]:
        prog = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self._console,
            transient=True,
        )
        prog.start()

        try:
            task_id = prog.add_task(description, total=total)

            class _Task:
                def advance(self, n: int = 1) -> None:
                    prog.advance(task_id, n)

                def set_total(self, total: int | None) -> None:
                    prog.update(task_id, total=total)

                def set_completed(self, completed: int) -> None:
                    prog.update(task_id, completed=completed)

                def set_description(self, description: str) -> None:
                    prog.update(task_id, description=description)

                def close(self) -> None:
                    # Context manager owns lifecycle.
                    pass

            task: ProgressTask = _Task()
            yield task
        finally:
            prog.stop()
