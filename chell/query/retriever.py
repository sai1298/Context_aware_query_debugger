from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer as _ST


class DPRRetriever:
    """Thin wrapper around a SentenceTransformer model for dense retrieval.

    The underlying model is lazy-loaded on first use so that importing this
    module does not trigger a large download or slow initialisation.
    """

    def __init__(
        self,
        encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self._encoder_model = encoder_model
        self._model: _ST | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[float]:
        """Return the embedding vector for a single text string."""
        return self._get_model().encode(text, show_progress_bar=False).tolist()

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for a batch of text strings."""
        embeddings = self._get_model().encode(texts, show_progress_bar=False)
        return [e.tolist() for e in embeddings]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_model(self) -> _ST:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]

            self._model = SentenceTransformer(self._encoder_model)
        return self._model
