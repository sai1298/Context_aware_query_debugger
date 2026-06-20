"""chell/evaluation/metrics.py

Four paper metrics for the Chell evaluation harness.

Each function accepts a list of result dicts — the shape produced by
``Evaluator.run_single`` — and returns a single float.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------


def error_resolution_accuracy(results: list[dict]) -> float:
    """Fraction of cases where the error was successfully resolved.

    Parameters
    ----------
    results:
        Each dict must contain ``"resolved": bool``.

    Returns
    -------
    float
        Value in [0, 1].  Returns 0.0 for an empty list.
    """
    if not results:
        return 0.0
    resolved = sum(1 for r in results if r.get("resolved", False))
    return resolved / len(results)


def interaction_efficiency(results: list[dict]) -> float:
    """Mean number of clarification turns for *resolved* cases (lower is better).

    Parameters
    ----------
    results:
        Each dict must contain ``"num_turns": int`` and ``"resolved": bool``.

    Returns
    -------
    float
        Mean turns over resolved cases.  Returns 0.0 when no cases were
        resolved (avoids division-by-zero and signals perfect efficiency
        vacuously).
    """
    resolved = [r for r in results if r.get("resolved", False)]
    if not resolved:
        return 0.0
    return sum(r.get("num_turns", 0) for r in resolved) / len(resolved)


def execution_correctness(results: list[dict]) -> float:
    """Fraction of cases where the corrected code both executed and matched output.

    Parameters
    ----------
    results:
        Each dict must contain ``"executed_ok": bool`` and
        ``"output_matches": bool``.

    Returns
    -------
    float
        Value in [0, 1].  Returns 0.0 for an empty list.
    """
    if not results:
        return 0.0
    correct = sum(
        1 for r in results
        if r.get("executed_ok", False) and r.get("output_matches", False)
    )
    return correct / len(results)


def ast_similarity_score(results: list[dict]) -> float:
    """Mean AST similarity between generated and reference corrected code.

    Parameters
    ----------
    results:
        Each dict must contain ``"ast_similarity": float`` in [0, 1].

    Returns
    -------
    float
        Mean value in [0, 1].  Returns 0.0 for an empty list.
    """
    if not results:
        return 0.0
    return sum(r.get("ast_similarity", 0.0) for r in results) / len(results)


# ---------------------------------------------------------------------------
# Convenience aggregate
# ---------------------------------------------------------------------------


def compute_all_metrics(results: list[dict]) -> dict[str, float]:
    """Compute all four paper metrics in one call.

    Parameters
    ----------
    results:
        List of result dicts; each must contain the union of keys expected by
        the individual metric functions.

    Returns
    -------
    dict[str, float]
        Keys: ``"error_resolution_accuracy"``, ``"interaction_efficiency"``,
        ``"execution_correctness"``, ``"ast_similarity_score"``.
    """
    return {
        "error_resolution_accuracy": error_resolution_accuracy(results),
        "interaction_efficiency": interaction_efficiency(results),
        "execution_correctness": execution_correctness(results),
        "ast_similarity_score": ast_similarity_score(results),
    }
