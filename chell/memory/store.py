from __future__ import annotations

from typing import Callable

from chell.memory.faiss_index import FaissIndex


class MemoryStore:
    """Vector-backed case store for retrieval-augmented debugging.

    Parameters
    ----------
    embed_fn:
        Callable that maps a text string to a list of floats (embedding).
        Compatible with ``ModelBackend.embed`` and ``DPRRetriever.encode``.
    index:
        An existing :class:`FaissIndex` instance. If *None*, a new
        ``FaissIndex(dim=dim)`` is created automatically.
    dim:
        Embedding dimension, used only when *index* is *None*.
    """

    def __init__(
        self,
        embed_fn: Callable[[str], list[float]],
        index: FaissIndex | None = None,
        dim: int = 64,
    ) -> None:
        self._embed_fn = embed_fn
        self._index: FaissIndex = index if index is not None else FaissIndex(dim=dim)

        # Ordered list of string case IDs; position == FAISS integer id.
        self._ids: list[str] = []
        # Full text lookup by case ID.
        self._texts: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_case(self, case_id: str, text: str) -> None:
        """Embed *text*, store it in the index, and record the id→text mapping."""
        embedding = self._embed_fn(text)
        faiss_id = len(self._ids)          # next integer slot
        self._ids.append(case_id)
        self._texts[case_id] = text
        self._index.add([embedding], [faiss_id])

    def retrieve(self, query_text: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Return the top-k most similar cases as (case_id, score) pairs.

        Scores are inner-product similarities (higher is better), matching
        the metric used by :class:`FaissIndex` with ``index_type="flat"``.
        """
        if self._index.size() == 0:
            return []
        query_embedding = self._embed_fn(query_text)
        raw = self._index.search(query_embedding, top_k=top_k)
        results: list[tuple[str, float]] = []
        for faiss_id, score in raw:
            if 0 <= faiss_id < len(self._ids):
                results.append((self._ids[faiss_id], score))
        return results

    def size(self) -> int:
        """Number of cases currently stored."""
        return len(self._ids)
