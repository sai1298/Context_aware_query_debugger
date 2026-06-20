from __future__ import annotations

import pytest

from chell.core.types import Correction, ValidationResult
from chell.validation.executor import SandboxExecutor
from chell.validation.validators import ASTValidator, ExecutionValidator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_correction(code: str, explanation: str = "test fix", diff: str = "") -> Correction:
    return Correction(code=code, explanation=explanation, diff=diff)


# ---------------------------------------------------------------------------
# SandboxExecutor tests
# ---------------------------------------------------------------------------

class TestSandboxExecutor:
    def setup_method(self) -> None:
        self.executor = SandboxExecutor(timeout=5)

    def test_sandbox_executor_success(self) -> None:
        """A simple print() should execute successfully and capture stdout."""
        success, stdout, stderr = self.executor.execute('print("hello")')

        assert success is True
        assert "hello" in stdout
        assert stderr == "" or stderr is not None  # stderr may be empty string

    def test_sandbox_executor_prints_multiple_lines(self) -> None:
        """Multiple print statements should all appear in stdout."""
        code = 'print("line1")\nprint("line2")'
        success, stdout, _stderr = self.executor.execute(code)

        assert success is True
        assert "line1" in stdout
        assert "line2" in stdout

    def test_sandbox_executor_runtime_error(self) -> None:
        """Code that raises an exception at runtime should report failure."""
        code = "raise ValueError('intentional error')"
        success, _stdout, stderr = self.executor.execute(code)

        assert success is False
        # stderr should contain the error message
        assert "ValueError" in stderr or "intentional error" in stderr

    def test_sandbox_executor_syntax_error(self) -> None:
        """Code with a syntax error should fail without raising in the caller."""
        code = "def broken(:"
        success, _stdout, _stderr = self.executor.execute(code)

        assert success is False

    def test_sandbox_executor_returns_tuple(self) -> None:
        """execute() must always return a 3-tuple (bool, str, str)."""
        result = self.executor.execute("x = 1")
        assert isinstance(result, tuple)
        assert len(result) == 3
        ok, out, err = result
        assert isinstance(ok, bool)
        assert isinstance(out, str)
        assert isinstance(err, str)

    def test_sandbox_executor_timeout(self) -> None:
        """An infinite loop should be stopped by the timeout."""
        executor = SandboxExecutor(timeout=1)
        code = "while True: pass"
        success, _stdout, stderr = executor.execute(code)

        assert success is False
        assert "Timeout" in stderr or "timeout" in stderr.lower()

    def test_sandbox_executor_empty_code(self) -> None:
        """Empty code string should succeed with empty stdout."""
        success, stdout, _stderr = self.executor.execute("")

        assert success is True
        assert stdout == ""


# ---------------------------------------------------------------------------
# ASTValidator tests
# ---------------------------------------------------------------------------

class TestASTValidator:
    def test_ast_validator_identical_code(self) -> None:
        """Identical code compared to itself must yield similarity == 1.0."""
        code = "x = df.groupby('col').sum()"
        validator = ASTValidator(reference_code=code)
        correction = _make_correction(code)

        result = validator.validate(correction)

        assert isinstance(result, ValidationResult)
        assert result.ast_similarity == pytest.approx(1.0)

    def test_ast_validator_very_different_code(self) -> None:
        """Structurally very different code should yield similarity < 1.0."""
        reference = "import pandas as pd\ndf = pd.DataFrame({'a': [1, 2, 3]})\nresult = df.groupby('a').sum()"
        different = "print('hello world')"
        validator = ASTValidator(reference_code=reference)
        correction = _make_correction(different)

        result = validator.validate(correction)

        assert isinstance(result, ValidationResult)
        assert result.ast_similarity < 1.0

    def test_ast_validator_passed_threshold(self) -> None:
        """Identical code should always pass the similarity threshold."""
        code = "result = x + y"
        validator = ASTValidator(reference_code=code)
        result = validator.validate(_make_correction(code))

        assert result.passed is True

    def test_ast_validator_syntax_error_returns_zero_similarity(self) -> None:
        """If either snippet has a syntax error, similarity should be 0.0."""
        reference = "x = 1"
        broken = "def bad(:"
        validator = ASTValidator(reference_code=reference)
        result = validator.validate(_make_correction(broken))

        assert result.ast_similarity == pytest.approx(0.0)
        assert result.passed is False

    def test_ast_validator_executed_ok_is_always_true(self) -> None:
        """ASTValidator doesn't execute code so executed_ok is always True."""
        code = "x = 1"
        validator = ASTValidator(reference_code=code)
        result = validator.validate(_make_correction(code))

        assert result.executed_ok is True

    def test_ast_validator_structurally_similar_code(self) -> None:
        """Code that shares AST node types should have non-zero similarity."""
        reference = "result = df.groupby('col').sum()"
        similar = "out = df.groupby('region').mean()"
        validator = ASTValidator(reference_code=reference)
        result = validator.validate(_make_correction(similar))

        # Both use Attribute, Call, Assign, etc. — should be > 0
        assert result.ast_similarity > 0.0


# ---------------------------------------------------------------------------
# ExecutionValidator integration tests
# ---------------------------------------------------------------------------

class TestExecutionValidator:
    def setup_method(self) -> None:
        executor = SandboxExecutor(timeout=5)
        self.validator = ExecutionValidator(executor=executor)

    def test_execution_validator_success(self) -> None:
        code = "x = 1 + 1\nprint(x)"
        correction = _make_correction(code)
        result = self.validator.validate(correction)

        assert isinstance(result, ValidationResult)
        assert result.executed_ok is True
        assert result.passed is True

    def test_execution_validator_failure(self) -> None:
        code = "raise RuntimeError('fail')"
        correction = _make_correction(code)
        result = self.validator.validate(correction)

        assert result.executed_ok is False
        assert result.passed is False
        assert result.error_message is not None
