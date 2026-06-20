#!/usr/bin/env python3
"""scripts/evaluate.py

Standalone evaluation script for the Chell project.

Loads the test split, builds the Evaluator and baselines, runs them all,
and writes a timestamped JSON results file to results/.

Usage
-----
    python scripts/evaluate.py
    python scripts/evaluate.py --config configs/eval.yaml --split test
    python scripts/evaluate.py --baselines-only
    python scripts/evaluate.py --chell-only
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_yaml(path: str) -> dict:
    try:
        import yaml
    except ImportError:
        print("ERROR: pyyaml not installed. Run: pip install pyyaml", file=sys.stderr)
        sys.exit(1)
    p = Path(path)
    if not p.exists():
        print(f"WARNING: config file not found: {path} — using defaults.")
        return {}
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _print_metrics(name: str, metrics: dict) -> None:
    print(f"\n  [{name}]")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")
        else:
            print(f"    {k}: {v}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Chell evaluation harness against all baselines."
    )
    parser.add_argument(
        "--config",
        default="configs/eval.yaml",
        metavar="PATH",
        help="Path to evaluation config YAML (default: configs/eval.yaml).",
    )
    parser.add_argument(
        "--split",
        default=None,
        metavar="SPLIT",
        help="Dataset split to evaluate on. Overrides the config file value.",
    )
    parser.add_argument(
        "--cases-dir",
        default="data/curated/",
        metavar="DIR",
        help="Directory with curated JSON case files (default: data/curated/).",
    )
    parser.add_argument(
        "--model-config",
        default="configs/model.yaml",
        metavar="PATH",
        help="Path to model config YAML (default: configs/model.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Directory to write results JSON. Overrides the config file value.",
    )
    parser.add_argument(
        "--chell-only",
        action="store_true",
        help="Run only the Chell pipeline evaluation (skip baselines).",
    )
    parser.add_argument(
        "--baselines-only",
        action="store_true",
        help="Run only the baseline evaluations (skip Chell pipeline).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load configs
    # ------------------------------------------------------------------
    eval_config = _load_yaml(args.config)
    model_config = _load_yaml(args.model_config)

    split: str = args.split or eval_config.get("split", "test")
    output_dir = Path(args.output_dir or eval_config.get("output_dir", "results/"))
    max_turns: int = eval_config.get("max_turns", 5)

    if not model_config:
        model_config = {
            "backend": "api",
            "api": {"provider": "anthropic", "model": "claude-sonnet-4-6"},
        }

    # ------------------------------------------------------------------
    # Imports
    # ------------------------------------------------------------------
    try:
        from chell.models.registry import build_backend
        from chell.data.dataset import ChellDataset
        from chell.evaluation.evaluator import Evaluator
        from chell.evaluation.baselines import run_all_baselines
        from chell.evaluation.metrics import compute_all_metrics
    except ImportError as exc:
        print(f"Import error: {exc}", file=sys.stderr)
        print("Ensure all dependencies are installed: pip install -e .", file=sys.stderr)
        sys.exit(1)

    # Conditionally import pipeline
    pipeline = None
    if not args.baselines_only:
        try:
            from chell.pipeline import ChellPipeline
        except ImportError:
            print(
                "WARNING: chell.pipeline not found — skipping Chell pipeline evaluation.",
                file=sys.stderr,
            )

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    print(f"Loading '{split}' split from {args.cases_dir} ...")
    dataset = ChellDataset(cases_dir=args.cases_dir, split=split)
    splits = dataset.load_splits()
    cases = splits.get(split, list(dataset))
    print(f"  {len(cases)} cases in split '{split}'.")

    # ------------------------------------------------------------------
    # Build backend
    # ------------------------------------------------------------------
    print("Building backend ...")
    try:
        backend = build_backend(model_config)
    except Exception as exc:
        print(f"Failed to build backend: {exc}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Collect results
    # ------------------------------------------------------------------
    all_results: dict = {"split": split, "timestamp": datetime.now().isoformat()}

    # 1. Chell pipeline ---------------------------------------------------
    if not args.baselines_only:
        try:
            from chell.pipeline import ChellPipeline

            print("\nRunning Chell pipeline evaluation ...")
            pipeline_obj = ChellPipeline.from_backend(backend, max_turns=max_turns)
            evaluator = Evaluator(
                pipeline=pipeline_obj, dataset=dataset, max_turns=max_turns
            )
            chell_metrics = evaluator.run(split=split)
            all_results["chell"] = chell_metrics
            _print_metrics("Chell", chell_metrics)
        except Exception as exc:
            print(f"Chell evaluation failed: {exc}", file=sys.stderr)
            all_results["chell"] = {"error": str(exc)}

    # 2. Baselines --------------------------------------------------------
    if not args.chell_only:
        print("\nRunning baselines ...")
        baseline_names = eval_config.get(
            "baselines", ["pylint", "one_shot_slm", "gpt4", "gemini"]
        )

        try:
            raw_baselines = run_all_baselines(cases=cases, backend=backend)
        except Exception as exc:
            print(f"Baselines failed: {exc}", file=sys.stderr)
            raw_baselines = {}

        all_results["baselines"] = {}
        for bname, bname_key in [
            ("pylint", "pylint"),
            ("one_shot_slm", "oneshot_slm"),
            ("gpt4", "gpt4"),
            ("gemini", "gemini"),
        ]:
            if bname not in baseline_names and bname_key not in baseline_names:
                continue
            key = bname_key if bname_key in raw_baselines else bname
            if key not in raw_baselines:
                continue
            try:
                bm = compute_all_metrics(raw_baselines[key])
            except Exception as exc:
                bm = {"error": str(exc)}
            all_results["baselines"][key] = bm
            _print_metrics(key, bm)

    # ------------------------------------------------------------------
    # Write results
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"eval_{split}_{ts}.json"
    with result_file.open("w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2)

    print(f"\nResults written to: {result_file}")


if __name__ == "__main__":
    main()
