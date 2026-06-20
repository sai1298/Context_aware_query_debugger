from __future__ import annotations

import pytest

from chell.core.types import (
    BugReport,
    ClarificationQuery,
    ConversationTurn,
    Correction,
    DebugSession,
    ErrorDiagnosis,
    UserResponse,
    ValidationResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session() -> DebugSession:
    report = BugReport(
        code="df.groupby('col')",
        task="Summarise sales by region",
        libs=["pandas"],
    )
    diagnosis = ErrorDiagnosis(
        error_type="wrong_aggregation",
        location="line 1: df.groupby('col')",
        description="Missing aggregation",
        confidence=0.8,
        is_ambiguous=True,
        candidate_intents=["sum", "mean"],
    )
    return DebugSession(bug_report=report, diagnosis=diagnosis)


def _make_query(question: str = "What aggregation?") -> ClarificationQuery:
    return ClarificationQuery(
        question=question,
        options=["sum", "mean", "count"],
        rationale="Need to resolve aggregation ambiguity",
        retrieved_refs=[],
    )


def _make_response(text: str = "sum", selection: int | None = 0) -> UserResponse:
    return UserResponse(text=text, selection=selection)


# ---------------------------------------------------------------------------
# test_debug_session_add_turn
# ---------------------------------------------------------------------------

def test_debug_session_add_turn() -> None:
    session = _make_session()
    assert session.num_turns == 0

    q1 = _make_query("What aggregation do you want?")
    r1 = _make_response("sum", selection=0)
    session.add_turn(q1, r1)

    assert session.num_turns == 1
    assert session.turns[0].query is q1
    assert session.turns[0].response is r1

    q2 = _make_query("Which column should be grouped?")
    r2 = _make_response("region", selection=None)
    session.add_turn(q2, r2)

    assert session.num_turns == 2
    assert session.turns[1].query is q2
    assert session.turns[1].response is r2


# ---------------------------------------------------------------------------
# test_debug_session_num_turns
# ---------------------------------------------------------------------------

def test_debug_session_num_turns() -> None:
    session = _make_session()
    assert session.num_turns == 0

    for i in range(5):
        session.add_turn(_make_query(f"Question {i}?"), _make_response(f"Answer {i}"))

    assert session.num_turns == 5


# ---------------------------------------------------------------------------
# test_debug_session_clarification_summary
# ---------------------------------------------------------------------------

def test_debug_session_clarification_summary_empty() -> None:
    session = _make_session()
    assert session.clarification_summary() == ""


def test_debug_session_clarification_summary_with_selection() -> None:
    session = _make_session()
    q = ClarificationQuery(
        question="Which aggregation?",
        options=["sum", "mean", "count"],
        rationale="Need to resolve",
        retrieved_refs=[],
    )
    r = UserResponse(text="I want the total", selection=0)
    session.add_turn(q, r)

    summary = session.clarification_summary()
    assert "Q1: Which aggregation?" in summary
    # When selection is set the chosen option is interpolated
    assert "sum" in summary
    assert "I want the total" in summary


def test_debug_session_clarification_summary_open_ended() -> None:
    session = _make_session()
    q = ClarificationQuery(
        question="What should the output look like?",
        options=[],
        rationale="Open-ended",
        retrieved_refs=[],
    )
    r = UserResponse(text="A single value per group", selection=None)
    session.add_turn(q, r)

    summary = session.clarification_summary()
    assert "Q1: What should the output look like?" in summary
    assert "A1: A single value per group" in summary


def test_debug_session_clarification_summary_multiple_turns() -> None:
    session = _make_session()
    for i in range(3):
        session.add_turn(
            _make_query(f"Question {i + 1}?"),
            _make_response(f"answer {i + 1}", selection=None),
        )

    summary = session.clarification_summary()
    lines = summary.splitlines()
    # 3 turns → 6 lines (Q + A each)
    assert len(lines) == 6
    assert lines[0].startswith("Q1:")
    assert lines[2].startswith("Q2:")
    assert lines[4].startswith("Q3:")
