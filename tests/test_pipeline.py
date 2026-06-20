from __future__ import annotations

import pytest

from chell.core.types import (
    BugReport,
    Correction,
    DebugSession,
    ErrorDiagnosis,
    ValidationResult,
)
from chell.core.pipeline import ChellPipeline
from chell.core.responders import SimulatedResponder
from chell.detection.base import ErrorDetector
from chell.detection.static_detector import StaticDetector
from chell.detection.llm_detector import LLMDetector
from chell.models.mock import MockModelBackend
from chell.query.generator import LLMQueryGenerator
from chell.refinement.base import Refiner
from chell.refinement.refiner import LLMRefiner
from chell.validation.base import Validator
from chell.validation.executor import SandboxExecutor
from chell.validation.validators import ExecutionValidator


# ---------------------------------------------------------------------------
# Stub implementations for controlled pipeline testing
# ---------------------------------------------------------------------------

class _AlwaysPassValidator(Validator):
    """Validator stub that always reports the correction as passing."""

    def validate(self, correction: Correction, reference_output: str = "") -> ValidationResult:
        return ValidationResult(
            executed_ok=True,
            output="",
            ast_similarity=1.0,
            passed=True,
            error_message=None,
        )


class _AlwaysFailValidator(Validator):
    """Validator stub that always reports the correction as failing."""

    def validate(self, correction: Correction, reference_output: str = "") -> ValidationResult:
        return ValidationResult(
            executed_ok=False,
            output="",
            ast_similarity=0.0,
            passed=False,
            error_message="stubbed failure",
        )


class _AmbiguousDetector(ErrorDetector):
    """Detector stub that always reports an ambiguous diagnosis."""

    def detect(self, report: BugReport) -> ErrorDiagnosis:
        return ErrorDiagnosis(
            error_type="ambiguous_intent",
            location="line 1",
            description="Ambiguous intent detected by stub",
            confidence=0.5,
            is_ambiguous=True,
            candidate_intents=["interpretation A", "interpretation B"],
        )


class _UnambiguousDetector(ErrorDetector):
    """Detector stub that always reports an unambiguous diagnosis."""

    def detect(self, report: BugReport) -> ErrorDiagnosis:
        return ErrorDiagnosis(
            error_type="unknown",
            location="",
            description="No issue detected",
            confidence=0.9,
            is_ambiguous=False,
            candidate_intents=[],
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bug_report(code: str = "x = 1") -> BugReport:
    return BugReport(code=code, task="compute something", libs=[])


def _build_pipeline(
    *,
    max_turns: int = 5,
    detector: ErrorDetector | None = None,
    validator: Validator | None = None,
) -> ChellPipeline:
    backend = MockModelBackend()
    return ChellPipeline(
        detector=detector or _AmbiguousDetector(),
        query_gen=LLMQueryGenerator(backend=backend),
        refiner=LLMRefiner(backend=backend),
        validator=validator or _AlwaysPassValidator(),
        memory=None,
        max_turns=max_turns,
    )


# ---------------------------------------------------------------------------
# test_full_pipeline_mock
# ---------------------------------------------------------------------------

class TestFullPipelineMock:
    def test_full_pipeline_mock_returns_debug_session(self) -> None:
        """Pipeline should return a DebugSession when run with mocked components."""
        pipeline = _build_pipeline()
        responder = SimulatedResponder(expected_response="sum", selection=0)
        bug_report = _make_bug_report()

        session = pipeline.debug(bug_report, responder)

        assert isinstance(session, DebugSession)

    def test_full_pipeline_mock_has_diagnosis(self) -> None:
        """Session should have a populated diagnosis after pipeline runs."""
        pipeline = _build_pipeline()
        responder = SimulatedResponder(expected_response="sum")
        session = pipeline.debug(_make_bug_report(), responder)

        assert session.diagnosis is not None
        assert isinstance(session.diagnosis, ErrorDiagnosis)

    def test_full_pipeline_mock_has_correction(self) -> None:
        """Session should have a Correction after pipeline runs."""
        pipeline = _build_pipeline()
        responder = SimulatedResponder(expected_response="sum")
        session = pipeline.debug(_make_bug_report(), responder)

        assert session.correction is not None
        assert isinstance(session.correction, Correction)
        assert isinstance(session.correction.code, str)

    def test_full_pipeline_mock_has_validation(self) -> None:
        """Session should have a ValidationResult after pipeline runs."""
        pipeline = _build_pipeline()
        responder = SimulatedResponder(expected_response="sum")
        session = pipeline.debug(_make_bug_report(), responder)

        assert session.validation is not None
        assert isinstance(session.validation, ValidationResult)

    def test_full_pipeline_mock_simulated_responder_recorded(self) -> None:
        """The SimulatedResponder's answer should appear in the session turns."""
        pipeline = _build_pipeline(max_turns=3)
        responder = SimulatedResponder(expected_response="use .sum()", selection=None)
        session = pipeline.debug(_make_bug_report(), responder)

        # At least one turn should have been recorded (ambiguous detector fires)
        assert session.num_turns >= 1
        # The simulated response should appear in one of the turns
        responses = [t.response.text for t in session.turns]
        assert "use .sum()" in responses

    def test_full_pipeline_mock_with_static_detector(self) -> None:
        """End-to-end test using StaticDetector + MockModelBackend for all LLM components."""
        backend = MockModelBackend()
        pipeline = ChellPipeline(
            detector=StaticDetector(),
            query_gen=LLMQueryGenerator(backend=backend),
            refiner=LLMRefiner(backend=backend),
            validator=_AlwaysPassValidator(),
            memory=None,
            max_turns=3,
        )
        responder = SimulatedResponder(expected_response="aggregate with sum")

        # Groupby without aggregation will be flagged as ambiguous by StaticDetector
        bug_report = BugReport(
            code="result = df.groupby('region')",
            task="Summarise sales by region",
            libs=["pandas"],
        )
        session = pipeline.debug(bug_report, responder)

        assert isinstance(session, DebugSession)
        assert session.correction is not None
        assert session.validation is not None
        assert session.validation.passed is True

    def test_full_pipeline_mock_with_execution_validator(self) -> None:
        """Integration test: the refiner fallback code is valid Python and passes execution."""
        backend = MockModelBackend()
        executor = SandboxExecutor(timeout=5)
        pipeline = ChellPipeline(
            detector=_UnambiguousDetector(),
            query_gen=LLMQueryGenerator(backend=backend),
            refiner=LLMRefiner(backend=backend),
            validator=ExecutionValidator(executor=executor),
            memory=None,
            max_turns=1,
        )
        responder = SimulatedResponder(expected_response="ok")

        # Simple code that will execute cleanly when the refiner falls back to it
        bug_report = BugReport(
            code='print("hello from chell")',
            task="print a greeting",
            libs=[],
        )
        session = pipeline.debug(bug_report, responder)

        # Refiner falls back to original code; executor runs it and it passes
        assert session.correction is not None
        assert session.validation is not None
        assert session.validation.executed_ok is True


# ---------------------------------------------------------------------------
# test_pipeline_max_turns
# ---------------------------------------------------------------------------

class TestPipelineMaxTurns:
    def test_pipeline_max_turns_respected(self) -> None:
        """Pipeline should never exceed max_turns clarification rounds."""
        max_turns = 3
        pipeline = _build_pipeline(
            max_turns=max_turns,
            detector=_AmbiguousDetector(),
            validator=_AlwaysFailValidator(),  # always fails → loop tries to keep going
        )
        responder = SimulatedResponder(expected_response="keep going")

        session = pipeline.debug(_make_bug_report(), responder)

        assert session.num_turns <= max_turns

    def test_pipeline_stops_early_when_passed(self) -> None:
        """Pipeline should stop before max_turns if validation passes."""
        pipeline = _build_pipeline(
            max_turns=10,
            detector=_AmbiguousDetector(),
            validator=_AlwaysPassValidator(),
        )
        responder = SimulatedResponder(expected_response="ok")

        session = pipeline.debug(_make_bug_report(), responder)

        # Should have stopped after one turn (initial clarification) rather than 10
        assert session.num_turns < 10
        assert session.validation is not None
        assert session.validation.passed is True

    def test_pipeline_zero_max_turns(self) -> None:
        """With max_turns=0 no clarification questions should be asked."""
        pipeline = _build_pipeline(
            max_turns=0,
            detector=_AmbiguousDetector(),
            validator=_AlwaysPassValidator(),
        )
        responder = SimulatedResponder(expected_response="anything")

        session = pipeline.debug(_make_bug_report(), responder)

        assert session.num_turns == 0

    def test_pipeline_unambiguous_diagnosis_skips_clarification(self) -> None:
        """With an unambiguous diagnosis the clarification loop is skipped."""
        pipeline = _build_pipeline(
            max_turns=5,
            detector=_UnambiguousDetector(),
            validator=_AlwaysPassValidator(),
        )
        responder = SimulatedResponder(expected_response="no question needed")

        session = pipeline.debug(_make_bug_report(), responder)

        # Unambiguous + passing validation → 0 turns
        assert session.num_turns == 0
        assert session.diagnosis is not None
        assert session.diagnosis.is_ambiguous is False

    def test_pipeline_max_turns_one(self) -> None:
        """With max_turns=1 and a failing validator, exactly 1 turn is taken."""
        pipeline = _build_pipeline(
            max_turns=1,
            detector=_AmbiguousDetector(),
            validator=_AlwaysFailValidator(),
        )
        responder = SimulatedResponder(expected_response="try once")

        session = pipeline.debug(_make_bug_report(), responder)

        assert session.num_turns == 1
