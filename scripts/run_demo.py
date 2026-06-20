#!/usr/bin/env python3
"""scripts/run_demo.py

Standalone demo script that runs a Chell debugging session without the CLI.

Usage
-----
    python scripts/run_demo.py --file examples/buggy_pandas.py
    python scripts/run_demo.py --code "import pandas as pd; df.groupby('x').mean()"
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

# Make the project root importable when run directly
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _detect_libs(source_code: str) -> list[str]:
    """Return a sorted list of top-level library names imported in *source_code*."""
    libs: list[str] = []
    try:
        tree = ast.parse(source_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    libs.append(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    libs.append(node.module.split(".")[0])
    except SyntaxError:
        pass
    return sorted(set(libs))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chell standalone demo — runs a debugging session using APIBackend (Claude)."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--file",
        metavar="PATH",
        help="Path to a Python source file to debug.",
    )
    source_group.add_argument(
        "--code",
        metavar="SNIPPET",
        help="Inline Python code snippet to debug.",
    )
    parser.add_argument(
        "--task",
        metavar="DESCRIPTION",
        default=None,
        help="Natural-language description of what the code should do. "
             "If omitted, you will be prompted interactively.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Read source code
    # ------------------------------------------------------------------
    if args.file:
        source_path = Path(args.file)
        if not source_path.exists():
            print(f"Error: file not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        source_code = source_path.read_text(encoding="utf-8")
        print(f"Debugging file: {args.file}")
    else:
        source_code = args.code
        print("Debugging inline code snippet.")

    print("\n--- Source Code ---")
    print(source_code)
    print("-------------------\n")

    # ------------------------------------------------------------------
    # Task description
    # ------------------------------------------------------------------
    task = args.task
    if task is None:
        task = input("Describe what this code is supposed to do: ").strip()
        if not task:
            print("Error: task description cannot be empty.", file=sys.stderr)
            sys.exit(1)

    # ------------------------------------------------------------------
    # Build pipeline with APIBackend (Claude)
    # ------------------------------------------------------------------
    try:
        from chell.models.api_backend import APIBackend
        from chell.pipeline import ChellPipeline
        from chell.core.types import BugReport
        from chell.core.responders import InteractiveResponder
    except ImportError as exc:
        print(f"Import error: {exc}", file=sys.stderr)
        print(
            "Ensure all dependencies are installed: pip install -e .",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Building backend (APIBackend with Anthropic Claude)...")
    try:
        backend = APIBackend(provider="anthropic")
    except Exception as exc:
        print(f"Failed to build backend: {exc}", file=sys.stderr)
        sys.exit(1)

    pipeline = ChellPipeline(backend=backend)

    # ------------------------------------------------------------------
    # Run the pipeline
    # ------------------------------------------------------------------
    libs = _detect_libs(source_code)
    bug_report = BugReport(code=source_code, task=task, libs=libs)
    responder = InteractiveResponder()

    print("\n--- Starting Chell Debug Session ---\n")
    try:
        session = pipeline.debug(bug_report, responder)
    except Exception as exc:
        print(f"Pipeline error: {exc}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print(f"\n--- Session complete ({session.num_turns} clarification turn(s)) ---\n")

    if session.correction is not None:
        c = session.correction
        print("=== Corrected Code ===")
        print(c.code)
        print("\n=== Explanation ===")
        print(c.explanation)
        if c.diff:
            print("\n=== Diff ===")
            print(c.diff)
    else:
        print("No correction was produced.")

    if session.validation is not None:
        v = session.validation
        print(f"\n=== Validation ===")
        print(f"  Passed:         {v.passed}")
        print(f"  Executed OK:    {v.executed_ok}")
        print(f"  AST Similarity: {v.ast_similarity:.2%}")
        if v.output:
            print(f"  Output:         {v.output[:200]}")
        if v.error_message:
            print(f"  Error:          {v.error_message}")


if __name__ == "__main__":
    main()
