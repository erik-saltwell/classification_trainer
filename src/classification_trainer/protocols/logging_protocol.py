from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Protocol, runtime_checkable


class StatusHandle(Protocol):
    def update(self, message: str) -> None: ...
    def close(self) -> None: ...


class ProgressTask(Protocol):
    def advance(self, n: int = 1) -> None: ...
    def set_total(self, total: int | None) -> None: ...
    def set_completed(self, completed: int) -> None: ...
    def set_description(self, description: str) -> None: ...
    def close(self) -> None: ...


@runtime_checkable
class LoggingProtocol(Protocol):
    def report_message(self, message: str) -> None: ...
    def report_warning(self, message: str) -> None: ...
    def report_error(self, message: str) -> None: ...
    def report_exception(self, context: str, exc: BaseException) -> None: ...
    def report_table_message(self, row_data: dict[str, Any]) -> None: ...
    def add_break(self, break_count: int = 1) -> None: ...

    def status(self, message: str) -> AbstractContextManager[StatusHandle]: ...
    def progress(self, description: str, total: int | None = None) -> AbstractContextManager[ProgressTask]: ...


class _NullStatus(StatusHandle):
    def update(self, message: str) -> None:
        pass

    def close(self) -> None:
        pass


class _NullProgress(ProgressTask):
    def advance(self, n: int = 1) -> None:
        pass

    def set_total(self, total: int | None) -> None:
        pass

    def set_completed(self, completed: int) -> None:
        pass

    def set_description(self, description: str) -> None:
        pass

    def close(self) -> None:
        pass


class NullLogger(LoggingProtocol):
    """
    Full no-op implementation of LoggingProtocol.

    Safe default for:
      - unit tests
      - CI/non-interactive runs
      - disabling all output
    """

    def report_message(self, message: str) -> None:
        pass

    def report_warning(self, message: str) -> None:
        pass

    def report_error(self, message: str) -> None:
        pass

    def report_exception(self, context: str, exc: BaseException) -> None:
        pass

    def report_table_message(self, row_data: dict[str, Any]) -> None:
        pass

    @contextmanager
    def status(self, message: str) -> Iterator[StatusHandle]:
        yield _NullStatus()

    @contextmanager
    def progress(self, description: str, total: int | None = None) -> Iterator[ProgressTask]:
        yield _NullProgress()
