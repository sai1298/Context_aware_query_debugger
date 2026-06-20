from __future__ import annotations

from abc import ABC, abstractmethod

from chell.core.types import BugReport, ErrorDiagnosis


class ErrorDetector(ABC):
    """Detect logical errors in a BugReport and return a structured diagnosis."""

    @abstractmethod
    def detect(self, report: BugReport) -> ErrorDiagnosis:
        ...
