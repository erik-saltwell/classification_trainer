from __future__ import annotations

from typing import Protocol

from .logging_protocol import LoggingProtocol


class CommmandProtocol(Protocol):
    def execute(self, logger: LoggingProtocol) -> None: ...
