from __future__ import annotations

import json

from chell.core.types import BugReport, ErrorDiagnosis
from chell.detection.base import ErrorDetector
from chell.detection.taxonomy import ErrorType
from chell.models.base import ModelBackend

_PROMPT_TEMPLATE = """\
You are an expert Python debugging assistant specialising in data-science code.
Analyse the following code for logical errors (not syntax errors).

## Task description
{task}

## Libraries in use
{libs}

## Code
```python
{code}
```

Identify the single most likely logical error and respond with **only** a JSON object
using these exact keys:

{{
  "error_type": "<one of the ErrorType enum values listed below>",
  "location": "<human-readable location, e.g. 'line 5: df.groupby(\\'col\\')'>",
  "description": "<clear explanation of what is wrong and why>",
  "confidence": <float between 0.0 and 1.0>,
  "is_ambiguous": <true or false>,
  "candidate_intents": ["<interpretation 1>", "<interpretation 2>"]
}}

Valid error_type values:
  wrong_groupby_key, wrong_aggregation, wrong_merge_key, wrong_filter_condition,
  wrong_column_selection, missing_reset_index,
  wrong_axis, broadcasting_error, wrong_operation,
  wrong_plot_type, missing_labels, wrong_data_mapping,
  off_by_one, wrong_variable, logic_inversion, ambiguous_intent, unknown

Return ONLY the JSON object — no prose, no markdown fences.
"""


class LLMDetector(ErrorDetector):
    """Detector that delegates to a language model for semantic error analysis."""

    def __init__(self, backend: ModelBackend) -> None:
        self._backend = backend

    def detect(self, report: BugReport) -> ErrorDiagnosis:
        prompt = _PROMPT_TEMPLATE.format(
            task=report.task,
            libs=", ".join(report.libs) if report.libs else "not specified",
            code=report.code,
        )

        raw = self._backend.generate(prompt, max_tokens=512, temperature=0.2)

        try:
            data = self._parse_json(raw)
            return ErrorDiagnosis(
                error_type=self._coerce_error_type(data.get("error_type", "")),
                location=str(data.get("location", "")),
                description=str(data.get("description", "")),
                confidence=float(data.get("confidence", 0.5)),
                is_ambiguous=bool(data.get("is_ambiguous", False)),
                candidate_intents=list(data.get("candidate_intents", [])),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return ErrorDiagnosis(
                error_type=ErrorType.UNKNOWN,
                location="",
                description=(
                    "LLM response could not be parsed into a structured diagnosis. "
                    f"Raw response: {raw[:200]}"
                ),
                confidence=0.1,
                is_ambiguous=True,
                candidate_intents=[
                    "Retry with a different model backend",
                    "Fall back to static analysis",
                ],
            )

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
        # Remove optional ```json ... ``` fencing
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            # drop opening fence
            lines = lines[1:]
            # drop closing fence if present
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()
        return json.loads(stripped)

    @staticmethod
    def _coerce_error_type(value: str) -> str:
        """Return value if it matches a known ErrorType, otherwise UNKNOWN."""
        try:
            return ErrorType(value).value
        except ValueError:
            return ErrorType.UNKNOWN.value
