from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass
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
        return

    def report_warning(self, message: str) -> None:
        return

    def report_error(self, message: str) -> None:
        return

    def report_exception(self, context: str, exc: BaseException) -> None:
        return

    def report_table_message(self, row_data: dict[str, Any]) -> None:
        return

    def report_multicolumn_table(self, headers: list[str], rows: list[list[str]]) -> None:
        return

    def add_break(self, break_count: int = 1) -> None:
        return

    @contextmanager
    def status(self, message: str) -> Iterator[StatusHandle]:
        """Yield a no-op status handle."""
        yield _NullStatus()

    @contextmanager
    def progress(self, description: str, total: int | None = None) -> Iterator[ProgressTask]:
        """Yield a no-op progress handle."""
        yield _NullProgress()


@dataclass
class CompositeStatusHandle(StatusHandle):
    members: list[StatusHandle]

    def update(self, message: str) -> None:
        for member in self.members:
            member.update(message)

    def close(self) -> None:
        for member in self.members:
            member.close()


@dataclass
class CompositeProgressTask(ProgressTask):
    members: list[ProgressTask]

    def advance(self, n: int = 1) -> None:
        for member in self.members:
            member.advance(n)

    def set_total(self, total: int | None) -> None:
        for member in self.members:
            member.set_total(total)

    def set_completed(self, completed: int) -> None:
        for member in self.members:
            member.set_completed(completed)

    def set_description(self, description: str) -> None:
        for member in self.members:
            member.set_description(description)

    def close(self) -> None:
        for member in self.members:
            member.close()


@dataclass
class CompositeLogger(LoggingProtocol):
    members: list[LoggingProtocol]

    """Protocol defining the structured logging interface for the application."""

    def report_message(self, message: str) -> None:
        for member in self.members:
            member.report_message(message)

    def report_warning(self, message: str) -> None:
        for member in self.members:
            member.report_warning(message)

    def report_error(self, message: str) -> None:
        for member in self.members:
            member.report_error(message)

    def report_exception(self, context: str, exc: BaseException) -> None:
        for member in self.members:
            member.report_exception(context, exc)

    def report_table_message(self, row_data: dict[str, Any]) -> None:
        for member in self.members:
            member.report_table_message(row_data)

    def report_multicolumn_table(self, headers: list[str], rows: list[list[str]]) -> None:
        for member in self.members:
            member.report_multicolumn_table(headers, rows)

    def add_break(self, break_count: int = 1) -> None:
        for member in self.members:
            member.add_break(break_count)

    @contextmanager
    def status(self, message: str) -> Iterator[StatusHandle]:
        with ExitStack() as stack:
            items: list[StatusHandle] = []
            for member in self.members:
                items.append(stack.enter_context(member.status(message)))
            yield CompositeStatusHandle(items)

    @contextmanager
    def progress(self, description: str, total: int | None = None) -> Iterator[ProgressTask]:
        with ExitStack() as stack:
            items: list[ProgressTask] = []
            for member in self.members:
                items.append(stack.enter_context(member.progress(description, total)))
            yield CompositeProgressTask(items)
