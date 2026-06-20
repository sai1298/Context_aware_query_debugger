from __future__ import annotations

import difflib
import json

from chell.core.types import Correction, DebugSession
from chell.models.base import ModelBackend
from chell.refinement.base import Refiner

_PROMPT_TEMPLATE = """\
You are an expert Python debugging assistant specialising in data-science code.
Your sole task is to produce a corrected version of the buggy code below by fixing
ONLY the specific logical error that has been identified. Do not refactor, rename, or
alter any code that is unrelated to the diagnosed bug.

## Original buggy code
```python
{code}
```

## Task description
{task}

## Libraries in use
{libs}

## Diagnosed error
Type: {error_type}
Location: {location}
Description: {description}
Confidence: {confidence}

## Clarification Q&A
{clarification}

Based on the information above, produce the corrected code with a concise explanation
of what was changed and why.

Respond with **only** a JSON object using these exact keys:

{{
  "corrected_code": "<full corrected Python code as a string>",
  "explanation": "<concise explanation of the fix — what was wrong and what was changed>",
  "diff": ""
}}

The "diff" field should be left as an empty string; it will be computed automatically.
Return ONLY the JSON object — no prose, no markdown fences.
"""


class LLMRefiner(Refiner):
    """Produce a corrected version of the buggy code via a pluggable LLM backend."""

    def __init__(self, backend: ModelBackend) -> None:
        self._backend = backend

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refine(self, session: DebugSession) -> Correction:
        prompt = self._build_prompt(session)
        raw = self._backend.generate(prompt, max_tokens=1024, temperature=0.2)
        return self._parse_response(raw, original_code=session.bug_report.code)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_prompt(self, session: DebugSession) -> str:
        report = session.bug_report
        diag = session.diagnosis

        if diag is not None:
            error_type = diag.error_type
            location = diag.location
            description = diag.description
            confidence = f"{diag.confidence:.2f}"
        else:
            error_type = "unknown"
            location = "unknown"
            description = "No diagnosis available."
            confidence = "N/A"

        clarification = session.clarification_summary() or "No clarification questions were asked."

        return _PROMPT_TEMPLATE.format(
            code=report.code,
            task=report.task,
            libs=", ".join(report.libs) if report.libs else "not specified",
            error_type=error_type,
            location=location,
            description=description,
            confidence=confidence,
            clarification=clarification,
        )

    def _parse_response(self, raw: str, original_code: str) -> Correction:
        try:
            data = self._parse_json(raw)
            corrected_code = str(data["corrected_code"])
            explanation = str(data.get("explanation", ""))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            # Fall back: return the original code with an error note.
            corrected_code = original_code
            explanation = (
                "LLM response could not be parsed; returning original code unchanged. "
                f"Raw response (first 200 chars): {raw[:200]}"
            )

        diff_str = self._compute_diff(original_code, corrected_code)
        return Correction(code=corrected_code, explanation=explanation, diff=diff_str)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Extract and parse a JSON object from the model output.

        The model may occasionally wrap the JSON in markdown fences; this
        method strips them before parsing.
        """
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()
        return json.loads(stripped)

    @staticmethod
    def _compute_diff(original: str, corrected: str) -> str:
        """Return a unified diff string between *original* and *corrected* code."""
        original_lines = original.splitlines(keepends=True)
        corrected_lines = corrected.splitlines(keepends=True)
        diff_lines = list(
            difflib.unified_diff(
                original_lines,
                corrected_lines,
                fromfile="original.py",
                tofile="corrected.py",
            )
        )
        return "".join(diff_lines)
