from __future__ import annotations

from abc import ABC, abstractmethod

from chell.core.types import Correction, ValidationResult


class Validator(ABC):
    """Validate a proposed correction."""

    @abstractmethod
    def validate(self, correction: Correction, reference_output: str = "") -> ValidationResult:
        ...
