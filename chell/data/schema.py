from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import jsonschema

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class DebugCase:
    """A single annotated debugging example used for training and evaluation."""

    id: str
    buggy_code: str
    task: str
    libs: list[str]
    error_type: str           # ErrorType value from detection/taxonomy.py
    error_location: str       # human-readable location string
    clarification_query: str
    clarification_options: list[str]
    expected_user_response: str
    corrected_code: str
    explanation: str
    difficulty: str = "medium"   # "easy" | "medium" | "hard"
    category: str = "pandas"     # "pandas" | "numpy" | "matplotlib" | "misc"

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "buggy_code": self.buggy_code,
            "task": self.task,
            "libs": self.libs,
            "error_type": self.error_type,
            "error_location": self.error_location,
            "clarification_query": self.clarification_query,
            "clarification_options": self.clarification_options,
            "expected_user_response": self.expected_user_response,
            "corrected_code": self.corrected_code,
            "explanation": self.explanation,
            "difficulty": self.difficulty,
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DebugCase":
        return cls(
            id=d["id"],
            buggy_code=d["buggy_code"],
            task=d["task"],
            libs=d["libs"],
            error_type=d["error_type"],
            error_location=d["error_location"],
            clarification_query=d["clarification_query"],
            clarification_options=d["clarification_options"],
            expected_user_response=d["expected_user_response"],
            corrected_code=d["corrected_code"],
            explanation=d["explanation"],
            difficulty=d.get("difficulty", "medium"),
            category=d.get("category", "pandas"),
        )


# ---------------------------------------------------------------------------
# JSON Schema
# ---------------------------------------------------------------------------

_VALID_ERROR_TYPES = [
    "wrong_groupby_key",
    "wrong_aggregation",
    "wrong_merge_key",
    "wrong_filter_condition",
    "wrong_column_selection",
    "missing_reset_index",
    "wrong_axis",
    "broadcasting_error",
    "wrong_operation",
    "wrong_plot_type",
    "missing_labels",
    "wrong_data_mapping",
    "off_by_one",
    "wrong_variable",
    "logic_inversion",
    "ambiguous_intent",
    "unknown",
]

DEBUG_CASE_SCHEMA: dict = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "DebugCase",
    "type": "object",
    "required": [
        "id",
        "buggy_code",
        "task",
        "libs",
        "error_type",
        "error_location",
        "clarification_query",
        "clarification_options",
        "expected_user_response",
        "corrected_code",
        "explanation",
    ],
    "additionalProperties": False,
    "properties": {
        "id": {
            "type": "string",
            "minLength": 1,
            "description": "Unique case identifier, e.g. 'case_001'",
        },
        "buggy_code": {
            "type": "string",
            "minLength": 1,
            "description": "Python source code containing the logical error",
        },
        "task": {
            "type": "string",
            "minLength": 1,
            "description": "Natural-language description of what the code should do",
        },
        "libs": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of library names the code uses (may be empty for pure-Python cases)",
        },
        "error_type": {
            "type": "string",
            "enum": _VALID_ERROR_TYPES,
            "description": "ErrorType enum value from detection/taxonomy.py",
        },
        "error_location": {
            "type": "string",
            "minLength": 1,
            "description": "Human-readable pointer to the error, e.g. 'line 3: df.merge(...)'",
        },
        "clarification_query": {
            "type": "string",
            "minLength": 1,
            "description": "Question posed to the user to resolve the ambiguity",
        },
        "clarification_options": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "description": "Multiple-choice options presented with the query",
        },
        "expected_user_response": {
            "type": "string",
            "minLength": 1,
            "description": "Ground-truth answer the user should give",
        },
        "corrected_code": {
            "type": "string",
            "minLength": 1,
            "description": "Fixed Python code",
        },
        "explanation": {
            "type": "string",
            "minLength": 1,
            "description": "Human-readable explanation of what was wrong and how it was fixed",
        },
        "difficulty": {
            "type": "string",
            "enum": ["easy", "medium", "hard"],
            "default": "medium",
        },
        "category": {
            "type": "string",
            "enum": ["pandas", "numpy", "matplotlib", "misc"],
            "default": "pandas",
        },
    },
}


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def validate_case(case_dict: dict) -> None:
    """Validate *case_dict* against DEBUG_CASE_SCHEMA.

    Raises
    ------
    jsonschema.ValidationError
        When the dict does not conform to the schema.
    """
    jsonschema.validate(instance=case_dict, schema=DEBUG_CASE_SCHEMA)
