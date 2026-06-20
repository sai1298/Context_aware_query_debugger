from __future__ import annotations

import numpy as np

try:
    import faiss
except ImportError as _faiss_err:
    raise ImportError(
        "faiss is required for FaissIndex. "
        "Install it with: pip install faiss-cpu   (CPU-only)\n"
        "                 pip install faiss-gpu   (GPU)\n"
        f"Original error: {_faiss_err}"
    ) from _faiss_err


class FaissIndex:
    """Thin wrapper around a FAISS index with integer-id tracking.

    Parameters
    ----------
    dim:
        Embedding dimensionality.
    index_type:
        "flat" → IndexFlatIP (inner-product, assumes unit-normalised vectors).
        "ivf"  → IndexIVFFlat with nlist=100 and a flat IP quantizer.
    """

    def __init__(self, dim: int, index_type: str = "flat") -> None:
        self._dim = dim
        self._index_type = index_type
        self._index = self._build(dim, index_type)
        self._ntotal: int = 0  # shadow counter (faiss .ntotal is reliable but handy)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build(dim: int, index_type: str) -> faiss.Index:
        if index_type == "flat":
            return faiss.IndexFlatIP(dim)
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, 100, faiss.METRIC_INNER_PRODUCT)
            return index
        else:
            raise ValueError(f"Unknown index_type {index_type!r}. Choose 'flat' or 'ivf'.")

    @staticmethod
    def _to_float32(vectors: list[list[float]]) -> np.ndarray:
        arr = np.array(vectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return np.ascontiguousarray(arr)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, vectors: list[list[float]], ids: list[int]) -> None:
        """Add *vectors* to the index, each identified by the corresponding *ids* entry."""
        arr = self._to_float32(vectors)
        ids_arr = np.array(ids, dtype=np.int64)

        # IVF indexes must be trained before adding; train on the incoming batch if needed.
        if self._index_type == "ivf" and not self._index.is_trained:
            self._index.train(arr)

        if self._index_type == "ivf":
            self._index.add_with_ids(arr, ids_arr)
        else:
            # IndexFlatIP assigns sequential IDs automatically (0, 1, 2...).
            # Callers (MemoryStore) always pass sequential ids, so this matches.
            self._index.add(arr)
        self._ntotal = self._index.ntotal

    def search(self, query: list[float], top_k: int = 5) -> list[tuple[int, float]]:
        """Return up to *top_k* (id, score) pairs sorted by descending score."""
        if self._index.ntotal == 0:
            return []
        arr = self._to_float32([query])
        k = min(top_k, self._index.ntotal)
        scores, ids = self._index.search(arr, k)
        results = [
            (int(i), float(s))
            for i, s in zip(ids[0], scores[0])
            if i != -1  # FAISS returns -1 for empty slots
        ]
        # Already sorted descending by FAISS for IP metric, but enforce it explicitly.
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def size(self) -> int:
        """Number of vectors currently in the index."""
        return self._index.ntotal

    def save(self, path: str) -> None:
        """Persist the FAISS index to *path*."""
        faiss.write_index(self._index, path)

    def load(self, path: str) -> None:
        """Replace the current index by reading from *path*."""
        self._index = faiss.read_index(path)
        self._ntotal = self._index.ntotal
