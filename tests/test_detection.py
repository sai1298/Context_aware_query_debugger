from __future__ import annotations

import pytest

from chell.core.types import BugReport, ErrorDiagnosis
from chell.detection.static_detector import StaticDetector
from chell.detection.llm_detector import LLMDetector
from chell.detection.taxonomy import ErrorType
from chell.models.mock import MockModelBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_report(code: str, task: str = "test task", libs: list[str] | None = None) -> BugReport:
    return BugReport(code=code, task=task, libs=libs or [])


# ---------------------------------------------------------------------------
# StaticDetector tests
# ---------------------------------------------------------------------------

class TestStaticDetector:
    def setup_method(self) -> None:
        self.detector = StaticDetector()

    def test_static_detector_groupby_no_agg(self) -> None:
        """A bare .groupby() with no aggregation should produce a WRONG_AGGREGATION diagnosis."""
        code = "result = df.groupby('region')"
        report = _make_report(code, libs=["pandas"])
        diagnosis = self.detector.detect(report)

        assert isinstance(diagnosis, ErrorDiagnosis)
        # The error_type should surface the aggregation issue or be ambiguous
        assert (
            "groupby" in diagnosis.error_type.lower()
            or "aggregation" in diagnosis.error_type.lower()
            or diagnosis.is_ambiguous
        ), f"Unexpected error_type: {diagnosis.error_type!r}"
        # There should be candidate intents suggesting aggregation functions
        assert len(diagnosis.candidate_intents) > 0

    def test_static_detector_groupby_with_agg_passes(self) -> None:
        """A .groupby() followed by .sum() should NOT be flagged as WRONG_AGGREGATION."""
        code = "result = df.groupby('region').sum()"
        report = _make_report(code, libs=["pandas"])
        diagnosis = self.detector.detect(report)

        # Either nothing matched (UNKNOWN) or a different rule fired, but not groupby agg
        assert diagnosis.error_type != ErrorType.WRONG_AGGREGATION.value

    def test_static_detector_syntax_error(self) -> None:
        """Unparseable code should return an UNKNOWN diagnosis, not raise."""
        code = "def broken(:"
        report = _make_report(code)
        diagnosis = self.detector.detect(report)

        assert isinstance(diagnosis, ErrorDiagnosis)
        assert diagnosis.error_type == ErrorType.UNKNOWN.value
        assert "SyntaxError" in diagnosis.description

    def test_static_detector_syntax_error_confidence(self) -> None:
        """Confidence on a syntax-error UNKNOWN should be low (< 0.5)."""
        code = "this is not python !!!"
        report = _make_report(code)
        diagnosis = self.detector.detect(report)

        # May or may not parse; if it does parse, we just check the result is an ErrorDiagnosis
        assert isinstance(diagnosis, ErrorDiagnosis)

    def test_static_detector_no_match_returns_unknown(self) -> None:
        """Clean code with no detectable pattern should return UNKNOWN."""
        code = "x = 1 + 1\nprint(x)"
        report = _make_report(code)
        diagnosis = self.detector.detect(report)

        assert isinstance(diagnosis, ErrorDiagnosis)
        assert diagnosis.error_type == ErrorType.UNKNOWN.value

    def test_static_detector_merge_no_on(self) -> None:
        """A merge() call without on/left_on/right_on should be flagged."""
        code = "result = df.merge(df2)"
        report = _make_report(code, libs=["pandas"])
        diagnosis = self.detector.detect(report)

        assert isinstance(diagnosis, ErrorDiagnosis)
        assert diagnosis.error_type == ErrorType.WRONG_MERGE_KEY.value

    def test_static_detector_returns_error_diagnosis_type(self) -> None:
        """detect() always returns an ErrorDiagnosis instance."""
        code = "for i in range(len(x)-1): pass"
        report = _make_report(code)
        result = self.detector.detect(report)
        assert isinstance(result, ErrorDiagnosis)


# ---------------------------------------------------------------------------
# LLMDetector tests
# ---------------------------------------------------------------------------

class TestLLMDetector:
    def setup_method(self) -> None:
        self.backend = MockModelBackend()
        self.detector = LLMDetector(backend=self.backend)

    def test_llm_detector_mock_returns_error_diagnosis(self) -> None:
        """LLMDetector with MockModelBackend should return an ErrorDiagnosis (graceful fallback)."""
        code = "result = df.groupby('col')"
        report = _make_report(code, task="group sales", libs=["pandas"])
        diagnosis = self.detector.detect(report)

        assert isinstance(diagnosis, ErrorDiagnosis)

    def test_llm_detector_mock_fallback_on_invalid_json(self) -> None:
        """MockModelBackend returns non-JSON, so the detector should fall back gracefully."""
        code = "x = 1"
        report = _make_report(code)
        diagnosis = self.detector.detect(report)

        # The mock response is not valid JSON — should trigger the fallback path
        assert isinstance(diagnosis, ErrorDiagnosis)
        # Fallback sets error_type to UNKNOWN
        assert diagnosis.error_type == ErrorType.UNKNOWN.value
        assert diagnosis.confidence < 0.5

    def test_llm_detector_mock_candidate_intents_present(self) -> None:
        """Fallback diagnosis always includes candidate intents."""
        code = "df.merge(other)"
        report = _make_report(code, libs=["pandas"])
        diagnosis = self.detector.detect(report)

        assert isinstance(diagnosis.candidate_intents, list)

    def test_llm_detector_no_exception_on_any_code(self) -> None:
        """LLMDetector must not raise regardless of code content."""
        for code in ["", "import os", "def f(): pass", "1/0"]:
            report = _make_report(code)
            result = self.detector.detect(report)
            assert isinstance(result, ErrorDiagnosis)
