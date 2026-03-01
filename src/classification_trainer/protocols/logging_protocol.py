from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Protocol, runtime_checkable


class StatusHandle(Protocol):
    """Protocol for a handle to an active status spinner or indicator."""

    def update(self, message: str) -> None:
        """Update the displayed status message."""
        ...

    def close(self) -> None:
        """Close and remove the status indicator."""
        ...


class ProgressTask(Protocol):
    """Protocol for a handle to an active progress bar or tracker."""

    def advance(self, n: int = 1) -> None:
        """Advance the progress by n steps."""
        ...

    def set_total(self, total: int | None) -> None:
        """Set or clear the total number of expected steps."""
        ...

    def set_completed(self, completed: int) -> None:
        """Set the number of completed steps to an absolute value."""
        ...

    def set_description(self, description: str) -> None:
        """Update the description text shown alongside the progress bar."""
        ...

    def close(self) -> None:
        """Close and remove the progress indicator."""
        ...


@runtime_checkable
class LoggingProtocol(Protocol):
    """Protocol defining the structured logging interface for the application."""

    def report_message(self, message: str) -> None:
        """Log an informational message."""
        ...

    def report_warning(self, message: str) -> None:
        """Log a warning message."""
        ...

    def report_error(self, message: str) -> None:
        """Log an error message."""
        ...

    def report_exception(self, context: str, exc: BaseException) -> None:
        """Log an exception with a description of the context in which it occurred."""
        ...

    def report_table_message(self, row_data: dict[str, Any]) -> None:
        """Log a row of key-value data, typically rendered as a table."""
        ...

    def report_multicolumn_table(self, headers: list[str], rows: list[list[str]]) -> None:
        """Log a columnar table with the given headers and rows."""
        ...

    def add_break(self, break_count: int = 1) -> None:
        """Insert visual line breaks in the output."""
        ...

    def status(self, message: str) -> AbstractContextManager[StatusHandle]:
        """Return a context manager providing an active status indicator."""
        ...

    def progress(self, description: str, total: int | None = None) -> AbstractContextManager[ProgressTask]:
        """Return a context manager providing an active progress tracker."""
        ...


class _NullStatus(StatusHandle):
    """No-op implementation of StatusHandle."""

    def update(self, message: str) -> None:
        """No-op: ignore status updates."""
        pass

    def close(self) -> None:
        """No-op: nothing to close."""
        pass


class _NullProgress(ProgressTask):
    """No-op implementation of ProgressTask."""

    def advance(self, n: int = 1) -> None:
        """No-op: ignore progress advances."""
        pass

    def set_total(self, total: int | None) -> None:
        """No-op: ignore total updates."""
        pass

    def set_completed(self, completed: int) -> None:
        """No-op: ignore completed updates."""
        pass

    def set_description(self, description: str) -> None:
        """No-op: ignore description updates."""
        pass

    def close(self) -> None:
        """No-op: nothing to close."""
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
        """No-op: discard informational messages."""
        pass

    def report_warning(self, message: str) -> None:
        """No-op: discard warnings."""
        pass

    def report_error(self, message: str) -> None:
        """No-op: discard errors."""
        pass

    def report_exception(self, context: str, exc: BaseException) -> None:
        """No-op: discard exceptions."""
        pass

    def report_table_message(self, row_data: dict[str, Any]) -> None:
        """No-op: discard table data."""
        pass

    def report_multicolumn_table(self, headers: list[str], rows: list[list[str]]) -> None:
        """No-op: discard table data."""
        pass

    @contextmanager
    def status(self, message: str) -> Iterator[StatusHandle]:
        """Yield a no-op status handle."""
        yield _NullStatus()

    @contextmanager
    def progress(self, description: str, total: int | None = None) -> Iterator[ProgressTask]:
        """Yield a no-op progress handle."""
        yield _NullProgress()
