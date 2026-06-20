from __future__ import annotations

import json

from chell.core.types import ClarificationQuery, DebugSession
from chell.models.base import ModelBackend
from chell.query.base import QueryGenerator

_FALLBACK_QUESTION = "Can you describe what you expected the code to do differently?"


class LLMQueryGenerator(QueryGenerator):
    """Generate a clarification question using a pluggable LLM backend."""

    def __init__(self, backend: ModelBackend, max_options: int = 4) -> None:
        self._backend = backend
        self._max_options = max_options

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, session: DebugSession) -> ClarificationQuery:
        prompt = self._build_prompt(session)
        raw = self._backend.generate(prompt, max_tokens=512, temperature=0.2)
        return self._parse_response(raw)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_prompt(self, session: DebugSession) -> str:
        report = session.bug_report
        diag = session.diagnosis

        diagnosis_block = ""
        if diag is not None:
            intents = "\n".join(f"  - {i}" for i in diag.candidate_intents)
            diagnosis_block = (
                f"Error type: {diag.error_type}\n"
                f"Description: {diag.description}\n"
                f"Candidate intents:\n{intents}\n"
            )

        history_block = ""
        summary = session.clarification_summary()
        if summary:
            history_block = f"\nPrevious clarification Q&A:\n{summary}\n"

        options_note = (
            f"Produce exactly {self._max_options} multiple-choice options "
            "unless the question is naturally open-ended (then use an empty list)."
        )

        return (
            "You are a debugging assistant helping a developer fix a code bug.\n"
            "Your task is to ask ONE targeted clarification question that resolves "
            "the most important ambiguity in the bug.\n\n"
            f"### Buggy code\n```\n{report.code}\n```\n\n"
            f"### Task description\n{report.task}\n\n"
            f"### Libraries in use\n{', '.join(report.libs) or 'none'}\n\n"
            f"### Diagnosis\n{diagnosis_block}"
            f"{history_block}\n"
            f"{options_note}\n\n"
            "Respond ONLY with a JSON object — no prose before or after — "
            "with these keys:\n"
            '  "question": string\n'
            '  "options": list[string]   # empty list for open-ended\n'
            '  "rationale": string       # internal reasoning, not shown to the user\n'
        )

    def _parse_response(self, raw: str) -> ClarificationQuery:
        # Strip markdown code fences if the model wrapped the JSON.
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            # Drop first and last fence lines.
            text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

        try:
            data = json.loads(text)
            question = str(data["question"])
            options: list[str] = [str(o) for o in data.get("options", [])]
            rationale = str(data.get("rationale", ""))
            return ClarificationQuery(
                question=question,
                options=options[: self._max_options],
                rationale=rationale,
                retrieved_refs=[],
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            return ClarificationQuery(
                question=_FALLBACK_QUESTION,
                options=[],
                rationale="JSON parse failed; using fallback question.",
                retrieved_refs=[],
            )
