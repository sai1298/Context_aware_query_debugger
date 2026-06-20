"""chell/evaluation/baselines.py

Baseline comparators for the Chell paper evaluation:

* ``run_pylint_baseline``  — static analysis (no model)
* ``run_oneshot_slm_baseline`` — one-shot fix with any ModelBackend (no clarification)
* ``run_gpt4_baseline``    — GPT-4o one-shot via APIBackend
* ``run_gemini_baseline``  — Gemini 1.5 Pro one-shot via APIBackend
* ``run_all_baselines``    — convenience wrapper that runs all four
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import tempfile
import textwrap
from typing import Optional

from chell.data.schema import DebugCase
from chell.models.base import ModelBackend


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ast_jaccard(code_a: str, code_b: str) -> float:
    """Jaccard similarity over the set of AST node-type names."""
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


def _run_code(code: str, timeout: int = 10) -> tuple[bool, str]:
    """Execute *code* in a subprocess; return (success, stdout_or_stderr)."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return True, result.stdout
        return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, f"TimeoutExpired after {timeout}s"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def _build_result(
    case: DebugCase,
    generated_code: str,
    executed_ok: bool,
    stdout: str,
    num_turns: int = 0,
) -> dict:
    """Build a result dict compatible with the metric functions."""
    ast_sim = _ast_jaccard(generated_code, case.corrected_code)
    resolved = executed_ok and ast_sim >= 0.7
    return {
        "case_id": case.id,
        "resolved": resolved,
        "num_turns": num_turns,
        "executed_ok": executed_ok,
        "output_matches": executed_ok,  # baselines have no reference output to compare
        "ast_similarity": ast_sim,
    }


_ONESHOT_PROMPT_TEMPLATE = textwrap.dedent("""\
    You are an expert Python debugging assistant.
    Fix the logical error in the code below. Return ONLY the corrected Python code
    with no explanation, no markdown fences, and no prose before or after.

    ## Task
    {task}

    ## Libraries
    {libs}

    ## Buggy code
    {code}
""")


def _oneshot_fix(case: DebugCase, backend: ModelBackend) -> str:
    """Ask *backend* to fix *case.buggy_code* in one shot (no clarification)."""
    prompt = _ONESHOT_PROMPT_TEMPLATE.format(
        task=case.task,
        libs=", ".join(case.libs) if case.libs else "not specified",
        code=case.buggy_code,
    )
    raw = backend.generate(prompt, max_tokens=1024, temperature=0.2)
    # Strip markdown fences if the model added them.
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text or case.buggy_code


# ---------------------------------------------------------------------------
# Baseline implementations
# ---------------------------------------------------------------------------


def run_pylint_baseline(case: DebugCase) -> dict:
    """Run pylint on the buggy code; treat a matching error code as 'resolved'.

    PyLint cannot *fix* code — it only detects issues.  We record
    ``resolved=True`` when pylint exits with a non-fatal return code and
    reports at least one message (indicating it found the problem), and
    ``executed_ok=False`` always (because no code generation happens).

    The AST similarity is computed between the *buggy* code and the
    *corrected* reference to represent the 0-fix baseline.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as tmp:
        tmp.write(case.buggy_code)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pylint",
                "--output-format=text",
                "--score=no",
                tmp_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout + result.stderr
        # pylint return code: 0 = no issues, 4 = warning, 8 = refactor,
        # 16 = convention, 32 = error.  Any non-zero means it found something.
        found_issue = result.returncode != 0 and bool(output.strip())
    except FileNotFoundError:
        # pylint not installed
        found_issue = False
        output = "pylint not installed"
    except subprocess.TimeoutExpired:
        found_issue = False
        output = "pylint timed out"
    except Exception as exc:  # noqa: BLE001
        found_issue = False
        output = str(exc)
    finally:
        import os
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    ast_sim = _ast_jaccard(case.buggy_code, case.corrected_code)

    return {
        "case_id": case.id,
        # pylint detection counts as resolution for comparison purposes
        "resolved": found_issue,
        "num_turns": 0,
        "executed_ok": False,   # pylint does not produce runnable code
        "output_matches": False,
        "ast_similarity": ast_sim,
        "pylint_output": output[:500],
    }


def run_oneshot_slm_baseline(case: DebugCase, backend: ModelBackend) -> dict:
    """Prompt *backend* to fix the code in a single shot, without clarification.

    Parameters
    ----------
    case:
        The debug case to fix.
    backend:
        Any :class:`~chell.models.base.ModelBackend` (local HF or API).
    """
    generated_code = _oneshot_fix(case, backend)
    executed_ok, stdout = _run_code(generated_code)
    return _build_result(case, generated_code, executed_ok, stdout, num_turns=0)


def run_gpt4_baseline(
    case: DebugCase,
    api_key: Optional[str] = None,
) -> dict:
    """One-shot fix using ``APIBackend(provider="openai", model="gpt-4o")``.

    Parameters
    ----------
    case:
        The debug case to fix.
    api_key:
        OpenAI API key.  Falls back to the ``OPENAI_API_KEY`` environment
        variable when ``None``.
    """
    from chell.models.api_backend import APIBackend  # local import avoids hard dep

    backend = APIBackend(provider="openai", model="gpt-4o", api_key=api_key)
    return run_oneshot_slm_baseline(case, backend)


def run_gemini_baseline(
    case: DebugCase,
    api_key: Optional[str] = None,
) -> dict:
    """One-shot fix using ``APIBackend(provider="gemini")``.

    Parameters
    ----------
    case:
        The debug case to fix.
    api_key:
        Google API key.  Falls back to the ``GOOGLE_API_KEY`` environment
        variable when ``None``.
    """
    from chell.models.api_backend import APIBackend  # local import avoids hard dep

    backend = APIBackend(provider="gemini", api_key=api_key)
    return run_oneshot_slm_baseline(case, backend)


# ---------------------------------------------------------------------------
# Aggregate runner
# ---------------------------------------------------------------------------


def run_all_baselines(
    cases: list[DebugCase],
    backend: ModelBackend,
) -> dict[str, list[dict]]:
    """Run every baseline over *cases* and return results keyed by baseline name.

    Parameters
    ----------
    cases:
        List of :class:`~chell.data.schema.DebugCase` objects (e.g. the test
        split from ``ChellDataset.load_splits()["test"]``).
    backend:
        A :class:`~chell.models.base.ModelBackend` used for the one-shot SLM
        baseline.  Not used for PyLint, GPT-4, or Gemini baselines (those have
        their own backends / no model).

    Returns
    -------
    dict[str, list[dict]]
        Keys: ``"pylint"``, ``"oneshot_slm"``, ``"gpt4"``, ``"gemini"``.
        Each value is a list of per-case result dicts.
    """
    pylint_results: list[dict] = []
    oneshot_results: list[dict] = []
    gpt4_results: list[dict] = []
    gemini_results: list[dict] = []

    for case in cases:
        pylint_results.append(run_pylint_baseline(case))
        oneshot_results.append(run_oneshot_slm_baseline(case, backend))

        try:
            gpt4_results.append(run_gpt4_baseline(case))
        except Exception as exc:  # noqa: BLE001
            gpt4_results.append(
                {
                    "case_id": case.id,
                    "resolved": False,
                    "num_turns": 0,
                    "executed_ok": False,
                    "output_matches": False,
                    "ast_similarity": 0.0,
                    "error": str(exc),
                }
            )

        try:
            gemini_results.append(run_gemini_baseline(case))
        except Exception as exc:  # noqa: BLE001
            gemini_results.append(
                {
                    "case_id": case.id,
                    "resolved": False,
                    "num_turns": 0,
                    "executed_ok": False,
                    "output_matches": False,
                    "ast_similarity": 0.0,
                    "error": str(exc),
                }
            )

    return {
        "pylint": pylint_results,
        "oneshot_slm": oneshot_results,
        "gpt4": gpt4_results,
        "gemini": gemini_results,
    }
