from __future__ import annotations

from abc import ABC, abstractmethod

from chell.core.types import Correction, DebugSession


class Refiner(ABC):
    """Produce a corrected version of the buggy code given the full debug session."""

    @abstractmethod
    def refine(self, session: DebugSession) -> Correction:
        ...
