from __future__ import annotations

import ast
from dataclasses import dataclass, field

from chell.core.types import BugReport, ErrorDiagnosis
from chell.detection.base import ErrorDetector
from chell.detection.taxonomy import ErrorType


@dataclass
class _Finding:
    """Intermediate result produced by a single detection rule."""

    error_type: ErrorType
    location: str
    description: str
    confidence: float
    is_ambiguous: bool
    candidate_intents: list[str] = field(default_factory=list)


def _source_line(source: str, lineno: int) -> str:
    """Return the stripped source line at lineno (1-based)."""
    lines = source.splitlines()
    if 1 <= lineno <= len(lines):
        return lines[lineno - 1].strip()
    return ""


# ---------------------------------------------------------------------------
# Rule helpers — each returns a _Finding or None
# ---------------------------------------------------------------------------


def _rule_groupby_no_agg(tree: ast.AST, source: str) -> _Finding | None:
    """Detect .groupby() calls whose result is used directly without aggregation."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "groupby"):
            continue
        # The groupby call is wrapped in another Call — check parent usage
        # We look for assignment: target = df.groupby(...) with no chained call
        # Heuristic: the groupby node's parent in the AST is *not* an Attribute
        # (meaning no method is chained after it).
        lineno = getattr(node, "lineno", 0)
        line_text = _source_line(source, lineno)
        # If the line contains .groupby( but no subsequent .sum/.mean/.agg etc.
        agg_keywords = {"sum", "mean", "count", "min", "max", "agg", "aggregate",
                        "size", "first", "last", "std", "var", "median", "apply"}
        remainder = line_text[line_text.find(".groupby") + len(".groupby"):]
        has_agg = any(f".{kw}" in remainder or f".{kw}(" in remainder
                      for kw in agg_keywords)
        if not has_agg:
            return _Finding(
                error_type=ErrorType.WRONG_AGGREGATION,
                location=f"line {lineno}: {line_text}",
                description=(
                    ".groupby() is called but no aggregation function is chained "
                    "(e.g. .sum(), .mean(), .agg()). The result will be a "
                    "DataFrameGroupBy object, not a summarised DataFrame."
                ),
                confidence=0.65,
                is_ambiguous=True,
                candidate_intents=[
                    "Group and aggregate with .sum()",
                    "Group and aggregate with .mean()",
                    "Group and aggregate with a custom .agg()",
                ],
            )
    return None


def _rule_merge_no_on(tree: ast.AST, source: str) -> _Finding | None:
    """Detect pd.merge() / df.merge() calls missing an explicit 'on' or 'left_on' argument."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_merge = (
            (isinstance(func, ast.Attribute) and func.attr == "merge")
            or (
                isinstance(func, ast.Name) and func.id == "merge"
            )
        )
        if not is_merge:
            continue
        kwarg_names = {kw.arg for kw in node.keywords}
        has_key = kwarg_names & {"on", "left_on", "right_on"}
        lineno = getattr(node, "lineno", 0)
        line_text = _source_line(source, lineno)
        if not has_key:
            return _Finding(
                error_type=ErrorType.WRONG_MERGE_KEY,
                location=f"line {lineno}: {line_text}",
                description=(
                    ".merge() is called without an explicit 'on', 'left_on', or "
                    "'right_on' key. Pandas will attempt to join on all common "
                    "column names, which is usually unintentional."
                ),
                confidence=0.62,
                is_ambiguous=True,
                candidate_intents=[
                    "Merge on a specific shared key column",
                    "Merge on different left/right key columns",
                ],
            )
    return None


def _rule_boolean_inversion(tree: ast.AST, source: str) -> _Finding | None:
    """Detect 'not x == y' patterns that should probably be 'x != y', or vice-versa."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.UnaryOp):
            continue
        if not isinstance(node.op, ast.Not):
            continue
        operand = node.operand
        if isinstance(operand, ast.Compare):
            lineno = getattr(node, "lineno", 0)
            line_text = _source_line(source, lineno)
            return _Finding(
                error_type=ErrorType.LOGIC_INVERSION,
                location=f"line {lineno}: {line_text}",
                description=(
                    "A 'not <comparison>' expression was detected. "
                    "This is often a logic inversion bug — consider using the "
                    "complementary operator directly (e.g. != instead of not ==)."
                ),
                confidence=0.55,
                is_ambiguous=True,
                candidate_intents=[
                    "Negate the entire comparison intentionally",
                    "Use the complementary comparison operator instead",
                ],
            )
    return None


def _rule_numpy_axis(tree: ast.AST, source: str, libs: list[str]) -> _Finding | None:
    """Detect numpy reduction calls where axis=0 vs axis=1 may be swapped."""
    if not any(lib in libs for lib in ("numpy", "np")):
        return None
    numpy_reductions = {"sum", "mean", "std", "var", "min", "max", "argmin", "argmax",
                        "cumsum", "cumprod", "prod", "all", "any"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr not in numpy_reductions:
            continue
        # Look for an axis keyword argument
        for kw in node.keywords:
            if kw.arg == "axis" and isinstance(kw.value, ast.Constant):
                axis_val = kw.value.value
                lineno = getattr(node, "lineno", 0)
                line_text = _source_line(source, lineno)
                # We flag this as a potential axis confusion (can't be certain)
                return _Finding(
                    error_type=ErrorType.WRONG_AXIS,
                    location=f"line {lineno}: {line_text}",
                    description=(
                        f"NumPy reduction '{func.attr}' is called with axis={axis_val}. "
                        "Verify that the axis direction is correct: "
                        "axis=0 reduces along rows (per-column result), "
                        "axis=1 reduces along columns (per-row result)."
                    ),
                    confidence=0.52,
                    is_ambiguous=True,
                    candidate_intents=[
                        f"Reduce along axis={axis_val} (current)",
                        f"Reduce along axis={1 - axis_val} (opposite direction)",
                    ],
                )
    return None


def _rule_matplotlib_missing_labels(tree: ast.AST, source: str,
                                    libs: list[str]) -> _Finding | None:
    """Detect matplotlib plots that are missing xlabel, ylabel, or title calls."""
    if not any(lib in libs for lib in ("matplotlib", "plt")):
        return None
    plot_methods = {"plot", "bar", "barh", "scatter", "hist", "pie", "boxplot",
                    "violinplot", "imshow", "contour", "contourf"}
    label_methods = {"xlabel", "ylabel", "title", "set_xlabel", "set_ylabel",
                     "set_title", "legend"}
    found_plot: int | None = None
    found_labels: set[str] = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            if func.attr in plot_methods and found_plot is None:
                found_plot = getattr(node, "lineno", 0)
            if func.attr in label_methods:
                found_labels.add(func.attr)

    if found_plot is not None:
        missing = {"xlabel", "ylabel", "title"} - found_labels
        if missing:
            line_text = _source_line(source, found_plot)
            missing_str = ", ".join(sorted(missing))
            return _Finding(
                error_type=ErrorType.MISSING_LABELS,
                location=f"line {found_plot}: {line_text}",
                description=(
                    f"A matplotlib plot is created but the following labels appear to "
                    f"be missing: {missing_str}. Charts should have clear axis labels "
                    f"and a descriptive title."
                ),
                confidence=0.60,
                is_ambiguous=False,
                candidate_intents=[
                    "Add missing axis labels and title for clarity",
                ],
            )
    return None


def _rule_off_by_one(tree: ast.AST, source: str) -> _Finding | None:
    """Detect slice or range patterns with suspicious literal boundaries."""
    for node in ast.walk(tree):
        # range(len(x)) used in a slice is common; range(len(x)-1) often off-by-one
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "range":
                args = node.args
                if args:
                    arg = args[-1]  # last positional arg (stop)
                    if (
                        isinstance(arg, ast.BinOp)
                        and isinstance(arg.op, ast.Sub)
                        and isinstance(arg.right, ast.Constant)
                        and arg.right.value == 1
                    ):
                        lineno = getattr(node, "lineno", 0)
                        line_text = _source_line(source, lineno)
                        return _Finding(
                            error_type=ErrorType.OFF_BY_ONE,
                            location=f"line {lineno}: {line_text}",
                            description=(
                                "range(..., n-1) detected. This excludes the last "
                                "element; if you meant to include it, use range(..., n)."
                            ),
                            confidence=0.55,
                            is_ambiguous=True,
                            candidate_intents=[
                                "Intentionally stop one element before the end",
                                "Include the final element — use range(n) instead",
                            ],
                        )
        # Slice with hard-coded 0 start and explicit stop-1
        if isinstance(node, ast.Subscript):
            sl = node.slice
            if isinstance(sl, ast.Slice):
                if (
                    sl.upper is not None
                    and isinstance(sl.upper, ast.BinOp)
                    and isinstance(sl.upper.op, ast.Sub)
                    and isinstance(sl.upper.right, ast.Constant)
                    and sl.upper.right.value == 1
                ):
                    lineno = getattr(node, "lineno", 0)
                    line_text = _source_line(source, lineno)
                    return _Finding(
                        error_type=ErrorType.OFF_BY_ONE,
                        location=f"line {lineno}: {line_text}",
                        description=(
                            "Slice upper bound is 'n-1', which excludes the last element. "
                            "If you need to include it, use [:n] or [:]."
                        ),
                        confidence=0.55,
                        is_ambiguous=True,
                        candidate_intents=[
                            "Intentionally exclude the final element",
                            "Include the final element — remove the '-1'",
                        ],
                    )
    return None


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

_RULES = [
    _rule_groupby_no_agg,
    _rule_merge_no_on,
    _rule_boolean_inversion,
    _rule_off_by_one,
]

_RULES_WITH_LIBS = [
    _rule_numpy_axis,
    _rule_matplotlib_missing_labels,
]


class StaticDetector(ErrorDetector):
    """AST-based heuristic detector. No external dependencies beyond the stdlib."""

    def detect(self, report: BugReport) -> ErrorDiagnosis:
        try:
            tree = ast.parse(report.code)
        except SyntaxError as exc:
            return ErrorDiagnosis(
                error_type=ErrorType.UNKNOWN,
                location=f"line {exc.lineno}: {exc.text or ''}".strip(),
                description=f"SyntaxError while parsing code: {exc.msg}",
                confidence=0.3,
                is_ambiguous=False,
                candidate_intents=[],
            )

        findings: list[_Finding] = []

        for rule in _RULES:
            result = rule(tree, report.code)
            if result is not None:
                findings.append(result)

        for rule in _RULES_WITH_LIBS:
            result = rule(tree, report.code, report.libs)
            if result is not None:
                findings.append(result)

        if not findings:
            return ErrorDiagnosis(
                error_type=ErrorType.UNKNOWN,
                location="",
                description=(
                    "No heuristic pattern matched. The error may be a subtle logic "
                    "issue that requires semantic analysis."
                ),
                confidence=0.2,
                is_ambiguous=True,
                candidate_intents=[
                    "Run the LLM-based detector for deeper analysis",
                ],
            )

        # Return the finding with the highest confidence
        best = max(findings, key=lambda f: f.confidence)
        return ErrorDiagnosis(
            error_type=best.error_type,
            location=best.location,
            description=best.description,
            confidence=best.confidence,
            is_ambiguous=best.is_ambiguous,
            candidate_intents=best.candidate_intents,
        )
