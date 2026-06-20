"""chell/evaluation — evaluation harness for the Chell paper.

Public API
----------
compute_all_metrics
    Compute all four paper metrics from a list of result dicts.

Evaluator
    Run the Chell pipeline over a dataset split and return metric scores.

run_all_baselines
    Run all baseline comparators (PyLint, one-shot SLM, GPT-4, Gemini).
"""

from __future__ import annotations

from chell.evaluation.metrics import compute_all_metrics
from chell.evaluation.evaluator import Evaluator
from chell.evaluation.baselines import run_all_baselines

__all__ = [
    "compute_all_metrics",
    "Evaluator",
    "run_all_baselines",
]
