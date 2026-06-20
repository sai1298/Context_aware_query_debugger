from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class BugReport:
    code: str
    task: str          # natural-language description of what the code should do
    libs: list[str]    # e.g. ['pandas', 'numpy']


@dataclass(frozen=True)
class ErrorDiagnosis:
    error_type: str           # value from ErrorType enum in detection/taxonomy.py
    location: str             # human-readable location, e.g. "line 5: df.groupby('col')"
    description: str
    confidence: float         # 0.0–1.0
    is_ambiguous: bool
    candidate_intents: list[str]  # competing interpretations of user intent


@dataclass(frozen=True)
class ClarificationQuery:
    question: str
    options: list[str]        # multiple-choice options; empty → open-ended
    rationale: str            # internal reasoning (not shown to user)
    retrieved_refs: list[str] = field(default_factory=list)  # memory-store case IDs


@dataclass(frozen=True)
class UserResponse:
    text: str
    selection: Optional[int] = None  # index into ClarificationQuery.options


@dataclass
class ConversationTurn:
    query: ClarificationQuery
    response: UserResponse


@dataclass(frozen=True)
class Correction:
    code: str
    explanation: str
    diff: str  # unified-diff string


@dataclass(frozen=True)
class ValidationResult:
    executed_ok: bool
    output: str
    ast_similarity: float  # 0.0–1.0
    passed: bool
    error_message: Optional[str] = None


@dataclass
class DebugSession:
    """Accumulates all state for a single debugging interaction."""

    bug_report: BugReport
    diagnosis: Optional[ErrorDiagnosis] = None
    turns: list[ConversationTurn] = field(default_factory=list)
    retrieved_cases: list[str] = field(default_factory=list)  # case IDs from MemoryStore
    correction: Optional[Correction] = None
    validation: Optional[ValidationResult] = None

    def add_turn(self, query: ClarificationQuery, response: UserResponse) -> None:
        self.turns.append(ConversationTurn(query=query, response=response))

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    def clarification_summary(self) -> str:
        """Flat text of all Q&A turns, consumed by ContextEncoder and prompt builders."""
        lines = []
        for i, turn in enumerate(self.turns, 1):
            lines.append(f"Q{i}: {turn.query.question}")
            if turn.response.selection is not None and turn.query.options:
                chosen = turn.query.options[turn.response.selection]
                lines.append(f"A{i}: {chosen} — {turn.response.text}")
            else:
                lines.append(f"A{i}: {turn.response.text}")
        return "\n".join(lines)
