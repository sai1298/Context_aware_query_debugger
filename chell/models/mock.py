from __future__ import annotations

import math
import hashlib

from chell.models.base import ModelBackend


class MockModelBackend(ModelBackend):
    """Deterministic stub — no network, no GPU, safe for unit tests."""

    def __init__(self, embed_dim: int = 64) -> None:
        self._embed_dim = embed_dim

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        return f"[mock response for prompt length={len(prompt)}]"

    def embed(self, text: str) -> list[float]:
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16)
        floats = []
        for i in range(self._embed_dim):
            val = math.sin(seed + i) * 0.5
            floats.append(val)
        norm = math.sqrt(sum(v * v for v in floats)) or 1.0
        return [v / norm for v in floats]

    @property
    def torch_model(self):
        return None
