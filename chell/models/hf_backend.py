from __future__ import annotations

from chell.models.base import ModelBackend


class HFBackend(ModelBackend):
    """Stubbed HuggingFace backend.

    A full implementation requires a local GPU and HuggingFace model weights.
    Use APIBackend for development and CI environments.
    """

    def __init__(self, model: str = "bigcode/starcoder2-7b", **kwargs) -> None:  # noqa: ARG002
        self._model_name = model
        raise NotImplementedError(
            "HFBackend requires a local GPU — use APIBackend for development. "
            f"Attempted to load model: {model!r}"
        )

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:  # noqa: ARG002
        raise NotImplementedError(
            "HFBackend requires a local GPU — use APIBackend for development."
        )

    def embed(self, text: str) -> list[float]:  # noqa: ARG002
        raise NotImplementedError(
            "HFBackend requires a local GPU — use APIBackend for development."
        )

    @property
    def torch_model(self):
        raise NotImplementedError(
            "HFBackend requires a local GPU — use APIBackend for development."
        )
