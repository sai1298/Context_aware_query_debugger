from __future__ import annotations

from abc import ABC, abstractmethod

from chell.core.types import ClarificationQuery, DebugSession


class QueryGenerator(ABC):
    """Generate a clarification question given the current debug session."""

    @abstractmethod
    def generate(self, session: DebugSession) -> ClarificationQuery:
        ...
