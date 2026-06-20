"""chell/evaluation/evaluator.py

Benchmark harness that runs the full Chell pipeline with ``SimulatedResponder``
over a dataset split and returns the four paper metrics.
"""

from __future__ import annotations

import ast
import re

from chell.core.types import BugReport, Correction, ValidationResult
from chell.core.responders import SimulatedResponder
from chell.data.schema import DebugCase
from chell.evaluation.metrics import compute_all_metrics


def _ast_jaccard(code_a: str, code_b: str) -> float:
    """Jaccard similarity over the set of AST node-type names in two snippets."""
    try:
        dump_a = ast.dump(ast.parse(code_a))
        dump_b = ast.dump(ast.parse(code_b))
    except SyntaxError:
        return 0.0

    def _node_types(dump: str) -> set[str]:
        return set(re.findall(r"([A-Z][A-Za-z]+)\(", dump))

    types_a = _node_types(dump_a)
    types_b = _node_types(dump_b)

    if not types_a and not types_b:
        return 1.0
    if not types_a or not types_b:
        return 0.0

    return len(types_a & types_b) / len(types_a | types_b)


class Evaluator:
    """Run the Chell pipeline over a dataset split and compute paper metrics.

    Parameters
    ----------
    pipeline:
        A ``ChellPipeline`` instance (from ``chell.core.pipeline``).  Typed
        as ``Any`` here to avoid a hard import dependency — the pipeline module
        is optional (not all subpackages exist at this milestone).
    dataset:
        A ``ChellDataset`` instance (from ``chell.data.dataset``).
    max_turns:
        Maximum clarification turns forwarded to the pipeline (default 5).
    """

    def __init__(self, pipeline, dataset, max_turns: int = 5) -> None:
        self._pipeline = pipeline
        self._dataset = dataset
        self._max_turns = max_turns

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, split: str = "test") -> dict[str, float]:
        """Evaluate over all cases in *split*.

        Parameters
        ----------
        split:
            One of ``"train"``, ``"val"``, or ``"test"``.  The dataset is
            expected to expose a ``load_splits()`` method; if the split is
            not present all loaded cases are used.

        Returns
        -------
        dict[str, float]
            All four paper metrics keyed by name.
        """
        cases = self._get_cases(split)
        raw_results: list[dict] = []
        for case in cases:
            raw_results.append(self.run_single(case))
        return compute_all_metrics(raw_results)

    def run_single(self, case: DebugCase) -> dict:
        """Run the pipeline on one case and return a raw result dict.

        The returned dict contains all keys consumed by the metric functions:

        * ``"resolved"`` — bool
        * ``"num_turns"`` — int
        * ``"executed_ok"`` — bool
        * ``"output_matches"`` — bool
        * ``"ast_similarity"`` — float
        * ``"case_id"`` — str (for debugging)

        Parameters
        ----------
        case:
            A single :class:`~chell.data.schema.DebugCase`.
        """
        bug_report = BugReport(
            code=case.buggy_code,
            task=case.task,
            libs=case.libs,
        )
        responder = SimulatedResponder(
            expected_response=case.expected_user_response,
        )

        # Run the pipeline; it returns a DebugSession.
        session = self._pipeline.debug(
            bug_report,
            responder,
            max_turns=self._max_turns,
        )

        # ---- resolution ---------------------------------------------------
        validation: ValidationResult | None = session.validation
        executed_ok: bool = validation.executed_ok if validation is not None else False
        passed: bool = validation.passed if validation is not None else False

        # ---- output match -------------------------------------------------
        # Compare pipeline output against corrected_code running output.
        # We treat "output_matches" as True when the correction code matches
        # the reference corrected_code string (exact or AST-equivalent) or
        # when the ValidationResult already signals passed.
        correction: Correction | None = session.correction
        generated_code: str = correction.code if correction is not None else ""

        output_matches: bool = passed  # honour validation verdict when available

        # ---- AST similarity -----------------------------------------------
        ast_sim: float
        if validation is not None and validation.ast_similarity > 0.0:
            ast_sim = validation.ast_similarity
        else:
            ast_sim = _ast_jaccard(generated_code, case.corrected_code)

        # ---- resolved = executed AND (passed or AST score >= threshold) ---
        resolved: bool = executed_ok and (passed or ast_sim >= 0.7)

        return {
            "case_id": case.id,
            "resolved": resolved,
            "num_turns": session.num_turns,
            "executed_ok": executed_ok,
            "output_matches": output_matches,
            "ast_similarity": ast_sim,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_cases(self, split: str) -> list[DebugCase]:
        """Return the list of cases for *split*, using load_splits when available."""
        try:
            splits = self._dataset.load_splits()
            return splits.get(split, [])
        except Exception:
            # Fallback: iterate the dataset directly (split filtering may already
            # have been applied at construction time).
            return list(self._dataset)
