#!/usr/bin/env python3
"""scripts/curate_dataset.py

Print dataset curation instructions and validate all existing cases
in data/curated/.

Usage
-----
    python scripts/curate_dataset.py
    python scripts/curate_dataset.py --cases-dir data/curated/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


_INSTRUCTIONS = """
Chell Dataset Curation Instructions
=====================================

Each curated case is a JSON file placed in the cases directory.
File naming convention: case_NNN.json (e.g. case_042.json)

Required JSON fields
--------------------
  id                     : unique string, e.g. "case_042"
  buggy_code             : Python source containing the logical error
  task                   : natural-language description of the intended behaviour
  libs                   : list of library names used, e.g. ["pandas", "numpy"]
  error_type             : one of the ErrorType values (see taxonomy below)
  error_location         : e.g. "line 3: df.groupby('col')"
  clarification_query    : the question posed to the user to resolve ambiguity
  clarification_options  : list of >= 2 multiple-choice strings
  expected_user_response : ground-truth answer (should match one of the options)
  corrected_code         : fixed Python source
  explanation            : prose explanation of the bug and the fix

Optional JSON fields
--------------------
  difficulty  : "easy" | "medium" | "hard"   (default: "medium")
  category    : "pandas" | "numpy" | "matplotlib" | "misc"  (default: "pandas")

Valid error_type values
-----------------------
  wrong_groupby_key, wrong_aggregation, wrong_merge_key,
  wrong_filter_condition, wrong_column_selection, missing_reset_index,
  wrong_axis, broadcasting_error, wrong_operation, wrong_plot_type,
  missing_labels, wrong_data_mapping, off_by_one, wrong_variable,
  logic_inversion, ambiguous_intent, unknown

Quality guidelines
------------------
  1. The buggy code must be syntactically valid Python.
  2. The error must be a LOGICAL error, not a syntax or runtime error.
  3. The clarification query must be answerable from the task description alone.
  4. The corrected code must run without error on a minimal fixture dataset.
  5. Prefer real-world pandas/numpy patterns over contrived examples.
  6. Aim for a balanced distribution across error types and difficulty levels.
"""


def validate_all(cases_dir: Path) -> tuple[int, int]:
    """Validate all JSON files in *cases_dir*.  Return (ok_count, fail_count)."""
    try:
        from chell.data.schema import validate_case
    except ImportError:
        print("ERROR: chell package not importable. Run: pip install -e .", file=sys.stderr)
        sys.exit(1)

    json_files = sorted(cases_dir.glob("*.json"))
    if not json_files:
        print(f"No .json files found in {cases_dir}")
        return 0, 0

    ok_count = 0
    fail_count = 0

    print(f"\nValidating {len(json_files)} case file(s) in {cases_dir}...\n")
    print(f"{'File':<25} {'ID':<15} {'Status'}")
    print("-" * 60)

    for json_file in json_files:
        try:
            with json_file.open() as fh:
                data = json.load(fh)
            validate_case(data)
            case_id = data.get("id", "?")
            print(f"{json_file.name:<25} {case_id:<15} OK")
            ok_count += 1
        except json.JSONDecodeError as exc:
            print(f"{json_file.name:<25} {'?':<15} FAIL (JSON parse error: {exc})")
            fail_count += 1
        except Exception as exc:
            case_id = data.get("id", "?") if isinstance(data, dict) else "?"
            print(f"{json_file.name:<25} {case_id:<15} FAIL ({exc})")
            fail_count += 1

    print("-" * 60)
    print(f"\nSummary: {ok_count} valid, {fail_count} invalid (total {ok_count + fail_count})")
    return ok_count, fail_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print Chell dataset curation instructions and validate existing cases."
    )
    parser.add_argument(
        "--cases-dir",
        default="data/curated/",
        metavar="DIR",
        help="Directory containing curated JSON case files (default: data/curated/).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Skip instructions; only run validation.",
    )
    args = parser.parse_args()

    if not args.validate_only:
        print(_INSTRUCTIONS)

    cases_dir = Path(args.cases_dir)
    if not cases_dir.exists():
        print(f"Cases directory does not exist: {cases_dir}")
        print(f"Create it with: mkdir -p {cases_dir}")
        sys.exit(0)

    ok, fail = validate_all(cases_dir)
    sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()
