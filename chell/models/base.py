from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn


class ModelBackend(ABC):
    """Pluggable backend: local HuggingFace model or hosted API."""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        """Return generated text for the given prompt."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Return a fixed-length embedding vector for the given text."""

    @property
    def torch_model(self) -> Optional["nn.Module"]:
        """Return the underlying nn.Module so CAAM can patch its attention layers.

        Returns None for API backends (CAAM is only applied to local HF models).
        """
        return None
