from __future__ import annotations

import math

from chell.core.types import ClarificationQuery, DebugSession


class QueryRanker:
    """Rank and deduplicate candidate clarification questions.

    Deduplication is done via cosine similarity against questions that have
    already been asked in the current session.  Ranking favours questions that
    carry multiple-choice options (easier for the user to answer) and questions
    that are longer / more specific.
    """

    def __init__(
        self,
        dedup_threshold: float = 0.9,
        max_candidates: int = 10,
    ) -> None:
        self._dedup_threshold = dedup_threshold
        self._max_candidates = max_candidates

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rank(
        self,
        candidates: list[ClarificationQuery],
        session: DebugSession,
    ) -> list[ClarificationQuery]:
        """Filter duplicates and return the top-ranked candidates.

        Parameters
        ----------
        candidates:
            Pool of questions to rank.
        session:
            Current debug session; used to check already-asked questions and
            retrieved case descriptions (anti-fatigue).

        Returns
        -------
        list[ClarificationQuery]
            Filtered, ranked list capped at *max_candidates*.
        """
        asked_embeddings = self._embed_asked_questions(session)
        retrieved_embeddings = self._embed_retrieved_cases(session)
        all_reference_embeddings = asked_embeddings + retrieved_embeddings
        filtered = self._deduplicate(candidates, all_reference_embeddings)
        ranked = sorted(filtered, key=self._score, reverse=True)
        return ranked[: self._max_candidates]

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors using only stdlib."""
        if len(a) != len(b):
            raise ValueError(
                f"Vector length mismatch: {len(a)} vs {len(b)}"
            )
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _embed_asked_questions(self, session: DebugSession) -> list[list[float]]:
        """Return character-frequency bag-of-words embeddings for already-asked questions.

        We avoid a heavy dependency here by using a lightweight character-level
        representation that is sufficient for near-duplicate detection.
        """
        return [
            self._lightweight_embed(turn.query.question)
            for turn in session.turns
        ]

    def _embed_retrieved_cases(self, session: DebugSession) -> list[list[float]]:
        """Return embeddings for retrieved case IDs to enable anti-fatigue filtering.

        Treats each retrieved case ID string as raw text and embeds it with the
        same lightweight character-frequency method used for asked questions.
        This prevents generating clarification questions that are too similar to
        already-retrieved case descriptions.
        """
        return [
            self._lightweight_embed(case_id)
            for case_id in session.retrieved_cases
        ]

    def _deduplicate(
        self,
        candidates: list[ClarificationQuery],
        asked_embeddings: list[list[float]],
    ) -> list[ClarificationQuery]:
        """Drop candidates that are too similar to already-asked questions."""
        if not asked_embeddings:
            return list(candidates)

        kept: list[ClarificationQuery] = []
        for query in candidates:
            emb = self._lightweight_embed(query.question)
            if not any(
                self.cosine_similarity(emb, asked) >= self._dedup_threshold
                for asked in asked_embeddings
            ):
                kept.append(query)
        return kept

    @staticmethod
    def _score(query: ClarificationQuery) -> float:
        """Heuristic relevance score for ranking.

        Higher is better.  Two factors:
        1. Having multiple-choice options (bonus +1.0 per option, capped at 4).
        2. Question specificity proxied by word count.
        """
        options_score = min(len(query.options), 4) * 1.0
        word_count = len(query.question.split())
        specificity_score = min(word_count / 20.0, 1.0)  # normalise to [0, 1]
        return options_score + specificity_score

    @staticmethod
    def _lightweight_embed(text: str) -> list[float]:
        """Character-frequency vector (256-dim) normalised to unit length.

        This is intentionally minimal — just enough for cosine-similarity-based
        near-duplicate detection without any external dependencies.
        """
        vec = [0.0] * 256
        for ch in text.lower():
            idx = ord(ch) % 256
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]
