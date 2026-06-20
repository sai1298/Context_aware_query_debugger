from __future__ import annotations

"""chell.data — dataset schema, loading, and building utilities.

Public API
----------
DebugCase
    Dataclass representing one annotated debugging example.
DEBUG_CASE_SCHEMA
    jsonschema-compatible dict for validating DebugCase dicts.
validate_case
    Validate a raw dict against DEBUG_CASE_SCHEMA.
ChellDataset
    Sequence-like collection of DebugCase objects with optional split filtering.
DatasetBuilder
    Builder for accumulating, validating, deduplicating, and saving cases.
"""

from chell.data.schema import DEBUG_CASE_SCHEMA, DebugCase, validate_case
from chell.data.dataset import ChellDataset
from chell.data.builder import DatasetBuilder

__all__ = [
    "DebugCase",
    "DEBUG_CASE_SCHEMA",
    "validate_case",
    "ChellDataset",
    "DatasetBuilder",
]
