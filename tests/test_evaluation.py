"""tests/test_evaluation.py

M6 tests: metrics, Evaluator, and baseline runners.
"""
from __future__ import annotations

import pytest

from chell.core.types import (
    BugReport,
    ClarificationQuery,
    Correction,
    DebugSession,
    ErrorDiagnosis,
    UserResponse,
    ValidationResult,
)
from chell.data.schema import DebugCase
from chell.evaluation.metrics import (
    ast_similarity_score,
    compute_all_metrics,
    error_resolution_accuracy,
    execution_correctness,
    interaction_efficiency,
)


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

def _r(
    resolved: bool = True,
    num_turns: int = 1,
    executed_ok: bool = True,
    output_matches: bool = True,
    ast_sim: float = 0.9,
    case_id: str = "c0",
) -> dict:
    return {
        "case_id": case_id,
        "resolved": resolved,
        "num_turns": num_turns,
        "executed_ok": executed_ok,
        "output_matches": output_matches,
        "ast_similarity": ast_sim,
    }


def _make_case(idx: int = 0) -> DebugCase:
    return DebugCase(
        id=f"case_{idx:03d}",
        buggy_code=f"x = {idx}",
        task="compute x",
        libs=[],
        error_type="off_by_one",
        error_location="line 1",
        clarification_query="Which value?",
        clarification_options=["a", "b"],
        expected_user_response="b",
        corrected_code=f"x = {idx + 1}",
        explanation="off by one",
    )


class _MockPipeline:
    """Pipeline that returns a pre-canned DebugSession for any case."""

    def __init__(self, executed_ok: bool = True, passed: bool = True) -> None:
        self._executed_ok = executed_ok
        self._passed = passed

    def debug(
        self,
        bug_report: BugReport,
        responder,
        max_turns: int = 5,
    ) -> DebugSession:
        session = DebugSession(bug_report=bug_report)
        session.diagnosis = ErrorDiagnosis(
            error_type="off_by_one",
            location="line 1",
            description="test",
            confidence=0.9,
            is_ambiguous=True,
            candidate_intents=[],
        )
        query = ClarificationQuery(
            question="Which value?", options=["a", "b"], rationale="r"
        )
        response = responder.answer(query)
        session.add_turn(query, response)
        session.correction = Correction(
            code=bug_report.code + "\n# fixed",
            explanation="fixed",
            diff="",
        )
        session.validation = ValidationResult(
            executed_ok=self._executed_ok,
            output="",
            ast_similarity=0.85,
            passed=self._passed,
        )
        return session


# ---------------------------------------------------------------------------
# error_resolution_accuracy
# ---------------------------------------------------------------------------

class TestErrorResolutionAccuracy:

    def test_all_resolved(self) -> None:
        results = [_r(resolved=True), _r(resolved=True)]
        assert error_resolution_accuracy(results) == 1.0

    def test_none_resolved(self) -> None:
        results = [_r(resolved=False), _r(resolved=False)]
        assert error_resolution_accuracy(results) == 0.0

    def test_half_resolved(self) -> None:
        results = [_r(resolved=True), _r(resolved=False)]
        assert error_resolution_accuracy(results) == pytest.approx(0.5)

    def test_empty(self) -> None:
        assert error_resolution_accuracy([]) == 0.0


# ---------------------------------------------------------------------------
# interaction_efficiency
# ---------------------------------------------------------------------------

class TestInteractionEfficiency:

    def test_basic(self) -> None:
        results = [_r(resolved=True, num_turns=2), _r(resolved=True, num_turns=4)]
        assert interaction_efficiency(results) == pytest.approx(3.0)

    def test_only_resolved_counted(self) -> None:
        results = [
            _r(resolved=True, num_turns=2),
            _r(resolved=False, num_turns=10),
        ]
        assert interaction_efficiency(results) == pytest.approx(2.0)

    def test_no_resolved(self) -> None:
        results = [_r(resolved=False, num_turns=5)]
        assert interaction_efficiency(results) == 0.0

    def test_empty(self) -> None:
        assert interaction_efficiency([]) == 0.0


# ---------------------------------------------------------------------------
# execution_correctness
# ---------------------------------------------------------------------------

class TestExecutionCorrectness:

    def test_all_correct(self) -> None:
        results = [_r(executed_ok=True, output_matches=True)] * 3
        assert execution_correctness(results) == pytest.approx(1.0)

    def test_exec_fail(self) -> None:
        results = [_r(executed_ok=False, output_matches=True)]
        assert execution_correctness(results) == 0.0

    def test_output_mismatch(self) -> None:
        results = [_r(executed_ok=True, output_matches=False)]
        assert execution_correctness(results) == 0.0

    def test_empty(self) -> None:
        assert execution_correctness([]) == 0.0


# ---------------------------------------------------------------------------
# ast_similarity_score
# ---------------------------------------------------------------------------

class TestASTSimilarityScore:

    def test_basic(self) -> None:
        results = [_r(ast_sim=0.8), _r(ast_sim=0.6)]
        assert ast_similarity_score(results) == pytest.approx(0.7)

    def test_empty(self) -> None:
        assert ast_similarity_score([]) == 0.0


# ---------------------------------------------------------------------------
# compute_all_metrics
# ---------------------------------------------------------------------------

class TestComputeAllMetrics:

    def test_keys_present(self) -> None:
        results = [_r()]
        metrics = compute_all_metrics(results)
        assert set(metrics) == {
            "error_resolution_accuracy",
            "interaction_efficiency",
            "execution_correctness",
            "ast_similarity_score",
        }

    def test_values_are_floats(self) -> None:
        results = [_r(), _r(resolved=False, executed_ok=False)]
        metrics = compute_all_metrics(results)
        for v in metrics.values():
            assert isinstance(v, float)

    def test_empty_input(self) -> None:
        metrics = compute_all_metrics([])
        assert all(v == 0.0 for v in metrics.values())


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class TestEvaluator:

    def test_run_single_resolved(self) -> None:
        from chell.core.responders import SimulatedResponder
        from chell.evaluation.evaluator import Evaluator

        case = _make_case(0)
        pipeline = _MockPipeline(executed_ok=True, passed=True)
        evaluator = Evaluator(pipeline=pipeline, dataset=None, max_turns=3)

        result = evaluator.run_single(case)

        assert result["case_id"] == case.id
        assert result["executed_ok"] is True
        assert isinstance(result["ast_similarity"], float)
        assert isinstance(result["num_turns"], int)

    def test_run_single_unresolved(self) -> None:
        from chell.evaluation.evaluator import Evaluator

        case = _make_case(1)
        pipeline = _MockPipeline(executed_ok=False, passed=False)
        evaluator = Evaluator(pipeline=pipeline, dataset=None, max_turns=3)

        result = evaluator.run_single(case)
        assert result["executed_ok"] is False
        assert result["resolved"] is False

    def test_run_returns_all_metric_keys(self) -> None:
        from chell.data.dataset import ChellDataset
        from chell.evaluation.evaluator import Evaluator

        cases = [_make_case(i) for i in range(3)]
        pipeline = _MockPipeline(executed_ok=True, passed=True)

        class _MockDataset:
            def load_splits(self):
                return {"test": cases}

            def __iter__(self):
                return iter(cases)

        evaluator = Evaluator(pipeline=pipeline, dataset=_MockDataset(), max_turns=3)
        metrics = evaluator.run(split="test")

        assert set(metrics) == {
            "error_resolution_accuracy",
            "interaction_efficiency",
            "execution_correctness",
            "ast_similarity_score",
        }


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class TestBaselines:

    def test_pylint_baseline_returns_result_dict(self) -> None:
        from chell.evaluation.baselines import run_pylint_baseline

        case = _make_case(0)
        result = run_pylint_baseline(case)

        for key in ("case_id", "resolved", "num_turns", "executed_ok", "ast_similarity"):
            assert key in result, f"Missing key: {key}"
        assert result["case_id"] == case.id
        assert isinstance(result["resolved"], bool)
        assert result["num_turns"] == 0

    def test_oneshot_slm_baseline_mock(self) -> None:
        from chell.evaluation.baselines import run_oneshot_slm_baseline
        from chell.models.mock import MockModelBackend

        case = _make_case(0)
        backend = MockModelBackend()
        result = run_oneshot_slm_baseline(case, backend)

        for key in ("case_id", "resolved", "num_turns", "executed_ok", "ast_similarity"):
            assert key in result
        assert result["case_id"] == case.id
        assert isinstance(result["ast_similarity"], float)
