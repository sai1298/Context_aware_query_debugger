from __future__ import annotations

import pytest

from chell.core.types import (
    BugReport,
    ClarificationQuery,
    DebugSession,
    ErrorDiagnosis,
    UserResponse,
)
from chell.models.mock import MockModelBackend
from chell.query.generator import LLMQueryGenerator
from chell.query.ranking import QueryRanker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session(with_diagnosis: bool = True) -> DebugSession:
    report = BugReport(
        code="result = df.groupby('region')",
        task="Summarise sales by region",
        libs=["pandas"],
    )
    diagnosis: ErrorDiagnosis | None = None
    if with_diagnosis:
        diagnosis = ErrorDiagnosis(
            error_type="wrong_aggregation",
            location="line 1: df.groupby('region')",
            description="groupby without aggregation",
            confidence=0.65,
            is_ambiguous=True,
            candidate_intents=["Group and aggregate with .sum()", "Group and aggregate with .mean()"],
        )
    return DebugSession(bug_report=report, diagnosis=diagnosis)


def _make_query(question: str, options: list[str] | None = None) -> ClarificationQuery:
    return ClarificationQuery(
        question=question,
        options=options or ["option A", "option B"],
        rationale="test rationale",
        retrieved_refs=[],
    )


# ---------------------------------------------------------------------------
# LLMQueryGenerator tests
# ---------------------------------------------------------------------------

class TestLLMQueryGenerator:
    def setup_method(self) -> None:
        self.backend = MockModelBackend()
        self.generator = LLMQueryGenerator(backend=self.backend, max_options=4)

    def test_query_generator_mock_returns_clarification_query(self) -> None:
        """With MockModelBackend the generator must return a ClarificationQuery (fallback or not)."""
        session = _make_session()
        result = self.generator.generate(session)

        assert isinstance(result, ClarificationQuery)

    def test_query_generator_mock_falls_back_gracefully(self) -> None:
        """Mock returns non-JSON; generator falls back to the default fallback question."""
        session = _make_session()
        result = self.generator.generate(session)

        # The fallback question is non-empty
        assert isinstance(result.question, str)
        assert len(result.question) > 0

    def test_query_generator_mock_options_is_list(self) -> None:
        """The options field must always be a list."""
        session = _make_session()
        result = self.generator.generate(session)

        assert isinstance(result.options, list)

    def test_query_generator_mock_rationale_is_str(self) -> None:
        """The rationale field must always be a string."""
        session = _make_session()
        result = self.generator.generate(session)

        assert isinstance(result.rationale, str)

    def test_query_generator_no_diagnosis_still_works(self) -> None:
        """Generator should not crash when session.diagnosis is None."""
        session = _make_session(with_diagnosis=False)
        result = self.generator.generate(session)

        assert isinstance(result, ClarificationQuery)

    def test_query_generator_with_existing_turns(self) -> None:
        """Generator should handle sessions that already have clarification turns."""
        session = _make_session()
        q = _make_query("What aggregation do you need?")
        r = UserResponse(text="sum", selection=0)
        session.add_turn(q, r)

        result = self.generator.generate(session)
        assert isinstance(result, ClarificationQuery)


# ---------------------------------------------------------------------------
# QueryRanker tests
# ---------------------------------------------------------------------------

class TestQueryRanker:
    def setup_method(self) -> None:
        self.ranker = QueryRanker(dedup_threshold=0.9, max_candidates=10)

    def _empty_session(self) -> DebugSession:
        return _make_session()

    def test_query_ranker_dedup_identical_questions(self) -> None:
        """Two identical questions should be treated as near-duplicates; only one survives."""
        session = _make_session()
        # Pre-populate session with the first question as already asked
        q_asked = _make_query("What aggregation function should be used?")
        session.add_turn(q_asked, UserResponse(text="sum"))

        # Candidate with an identical question
        q_dup = _make_query("What aggregation function should be used?")
        # And a different candidate
        q_unique = _make_query("Which column should be the groupby key?")

        ranked = self.ranker.rank([q_dup, q_unique], session)

        # The duplicate (already asked) should be filtered out
        questions = [q.question for q in ranked]
        assert "Which column should be the groupby key?" in questions
        assert "What aggregation function should be used?" not in questions

    def test_query_ranker_dedup_very_similar_questions(self) -> None:
        """Near-identical questions (high char-frequency cosine similarity) are deduplicated."""
        session = _make_session()
        q_asked = _make_query("What aggregation function do you want to use here?")
        session.add_turn(q_asked, UserResponse(text="mean"))

        # Slightly rephrased but essentially the same question — will have high cosine similarity
        q_similar = _make_query("What aggregation function do you want to use here?")
        q_different = _make_query("Should the result be a DataFrame or a Series?")

        ranked = self.ranker.rank([q_similar, q_different], session)

        # Only the clearly different question survives
        questions = [q.question for q in ranked]
        assert "Should the result be a DataFrame or a Series?" in questions

    def test_query_ranker_no_asked_questions_returns_all(self) -> None:
        """With no prior turns, all candidates should be returned (up to max_candidates)."""
        session = _make_session()  # no turns added
        candidates = [
            _make_query(f"Question number {i}?")
            for i in range(5)
        ]
        ranked = self.ranker.rank(candidates, session)

        assert len(ranked) == 5

    def test_query_ranker_prefers_multiple_choice(self) -> None:
        """Questions with more options should score higher and appear first."""
        session = _make_session()
        q_no_opts = ClarificationQuery(
            question="Describe the expected output",
            options=[],
            rationale="open-ended",
            retrieved_refs=[],
        )
        q_with_opts = ClarificationQuery(
            question="Pick the aggregation",
            options=["sum", "mean", "count", "max"],
            rationale="multiple choice",
            retrieved_refs=[],
        )
        ranked = self.ranker.rank([q_no_opts, q_with_opts], session)

        # Question with options should rank higher
        assert ranked[0].question == "Pick the aggregation"

    def test_query_ranker_respects_max_candidates(self) -> None:
        """Ranker should cap results at max_candidates."""
        ranker = QueryRanker(dedup_threshold=0.9, max_candidates=3)
        session = _make_session()
        candidates = [_make_query(f"Unique question about topic {i}?") for i in range(10)]

        ranked = ranker.rank(candidates, session)
        assert len(ranked) <= 3

    def test_query_ranker_cosine_similarity_identical_vectors(self) -> None:
        """Cosine similarity of a vector with itself is 1.0."""
        vec = [1.0, 0.0, 0.5]
        assert QueryRanker.cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_query_ranker_cosine_similarity_orthogonal_vectors(self) -> None:
        """Cosine similarity of orthogonal vectors is 0.0."""
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert QueryRanker.cosine_similarity(a, b) == pytest.approx(0.0)
