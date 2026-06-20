from __future__ import annotations

import ast

from chell.core.types import Correction, ValidationResult
from chell.validation.base import Validator
from chell.validation.executor import SandboxExecutor


def _ast_jaccard(code_a: str, code_b: str) -> float:
    """Compute Jaccard similarity on the multiset of AST node-type names.

    Uses the *set* of unique node-type names as a simple structural proxy.
    Returns 0.0 if either snippet fails to parse.
    """
    try:
        dump_a = ast.dump(ast.parse(code_a))
        dump_b = ast.dump(ast.parse(code_b))
    except SyntaxError:
        return 0.0

    # Extract node type names (words that start with an uppercase letter and
    # are followed by an opening paren in the dump string — a cheap proxy).
    def node_types(dump: str) -> set[str]:
        import re
        return set(re.findall(r"([A-Z][A-Za-z]+)\(", dump))

    types_a = node_types(dump_a)
    types_b = node_types(dump_b)

    if not types_a and not types_b:
        return 1.0
    if not types_a or not types_b:
        return 0.0

    intersection = types_a & types_b
    union = types_a | types_b
    return len(intersection) / len(union)


class ExecutionValidator(Validator):
    """Validate a correction by actually running it in a sandbox."""

    def __init__(self, executor: SandboxExecutor) -> None:
        self.executor = executor

    def validate(self, correction: Correction, reference_output: str = "") -> ValidationResult:
        executed_ok, stdout, stderr = self.executor.execute(correction.code)

        # Output match check (only when a reference is supplied).
        if reference_output and executed_ok:
            passed = stdout.strip() == reference_output.strip()
        else:
            passed = executed_ok

        error_message: str | None = stderr if not executed_ok else None

        return ValidationResult(
            executed_ok=executed_ok,
            output=stdout,
            ast_similarity=0.0,
            passed=passed,
            error_message=error_message,
        )


class ASTValidator(Validator):
    """Validate a correction by comparing its AST to a reference implementation."""

    SIMILARITY_THRESHOLD: float = 0.7

    def __init__(self, reference_code: str) -> None:
        self.reference_code = reference_code

    def validate(self, correction: Correction, reference_output: str = "") -> ValidationResult:
        score = _ast_jaccard(correction.code, self.reference_code)
        passed = score >= self.SIMILARITY_THRESHOLD

        return ValidationResult(
            executed_ok=True,
            output="",
            ast_similarity=score,
            passed=passed,
            error_message=None,
        )
