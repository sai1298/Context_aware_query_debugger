from __future__ import annotations

import math
import tempfile

import numpy as np
import pytest

from chell.memory.faiss_index import FaissIndex
from chell.memory.store import MemoryStore
from chell.core.pipeline import ChellPipeline
from chell.core.types import BugReport, Correction, ValidationResult
from chell.core.responders import SimulatedResponder
from chell.detection.base import ErrorDetector
from chell.detection.static_detector import StaticDetector
from chell.models.mock import MockModelBackend
from chell.query.generator import LLMQueryGenerator
from chell.refinement.refiner import LLMRefiner
from chell.validation.base import Validator


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _AlwaysPassValidator(Validator):
    def validate(self, correction: Correction, reference_output: str = "") -> ValidationResult:
        return ValidationResult(
            executed_ok=True,
            output="",
            ast_similarity=1.0,
            passed=True,
            error_message=None,
        )


# ---------------------------------------------------------------------------
# Helper: build unit-norm 4-dim vectors
# ---------------------------------------------------------------------------

def _unit_norm(vec: list[float]) -> list[float]:
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0.0:
        return vec
    return (arr / norm).tolist()


# Pre-defined unit-norm vectors for deterministic tests
_V0 = _unit_norm([1.0, 0.0, 0.0, 0.0])
_V1 = _unit_norm([0.0, 1.0, 0.0, 0.0])
_V2 = _unit_norm([0.0, 0.0, 1.0, 0.0])


# ---------------------------------------------------------------------------
# FaissIndex tests
# ---------------------------------------------------------------------------

class TestFaissIndex:
    def test_round_trip_rank_one(self) -> None:
        """Adding 3 unit-norm vectors and searching for one returns it at rank 1."""
        index = FaissIndex(dim=4, index_type="flat")
        index.add([_V0, _V1, _V2], [0, 1, 2])

        results = index.search(_V1, top_k=3)

        assert len(results) > 0
        top_id, top_score = results[0]
        assert top_id == 1
        assert top_score > 0.99  # should be ~1.0 for unit-norm IP

    def test_round_trip_all_three_returned(self) -> None:
        """Searching with top_k=3 returns all three added vectors."""
        index = FaissIndex(dim=4, index_type="flat")
        index.add([_V0, _V1, _V2], [0, 1, 2])

        results = index.search(_V0, top_k=3)
        ids = [r[0] for r in results]

        assert 0 in ids
        assert len(results) == 3

    def test_size_after_add(self) -> None:
        """Size reflects number of vectors added."""
        index = FaissIndex(dim=4, index_type="flat")
        assert index.size() == 0
        index.add([_V0], [0])
        assert index.size() == 1
        index.add([_V1, _V2], [1, 2])
        assert index.size() == 3

    def test_search_empty_returns_empty(self) -> None:
        """Searching an empty index returns an empty list."""
        index = FaissIndex(dim=4, index_type="flat")
        assert index.search(_V0, top_k=5) == []

    def test_save_load_round_trip(self) -> None:
        """Save and load preserves vectors; search after load still works."""
        index = FaissIndex(dim=4, index_type="flat")
        index.add([_V0, _V1, _V2], [0, 1, 2])

        with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as f:
            path = f.name

        index.save(path)

        # Load into a fresh index
        index2 = FaissIndex(dim=4, index_type="flat")
        index2.load(path)

        assert index2.size() == 3
        results = index2.search(_V2, top_k=3)
        assert results[0][0] == 2
        assert results[0][1] > 0.99


# ---------------------------------------------------------------------------
# MemoryStore tests
# ---------------------------------------------------------------------------

# Simple deterministic embed function: maps text to a fixed 4-dim unit vector
# keyed by content so similar queries can be tested.
_TEXT_TO_VEC: dict[str, list[float]] = {
    "case_alpha": _V0,
    "case_beta":  _V1,
    "case_gamma": _V2,
    "query_alpha": _V0,   # will match case_alpha
    "query_beta":  _V1,   # will match case_beta
}


def _mock_embed(text: str) -> list[float]:
    return _TEXT_TO_VEC.get(text, _unit_norm([1.0, 1.0, 1.0, 1.0]))


class TestMemoryStore:
    def test_add_and_retrieve_returns_case_id(self) -> None:
        """After adding 3 cases, retrieve returns the correct case_id at rank 1."""
        store = MemoryStore(embed_fn=_mock_embed, dim=4)
        store.add_case("case_alpha", "case_alpha")
        store.add_case("case_beta",  "case_beta")
        store.add_case("case_gamma", "case_gamma")

        results = store.retrieve("query_alpha", top_k=3)
        case_ids = [cid for cid, _ in results]

        assert "case_alpha" in case_ids
        assert results[0][0] == "case_alpha"

    def test_size_increments(self) -> None:
        """MemoryStore.size() increments correctly with each add_case call."""
        store = MemoryStore(embed_fn=_mock_embed, dim=4)
        assert store.size() == 0
        store.add_case("c1", "case_alpha")
        assert store.size() == 1
        store.add_case("c2", "case_beta")
        assert store.size() == 2
        store.add_case("c3", "case_gamma")
        assert store.size() == 3

    def test_retrieve_empty_returns_empty_list(self) -> None:
        """Retrieving from an empty MemoryStore returns an empty list."""
        store = MemoryStore(embed_fn=_mock_embed, dim=4)
        assert store.retrieve("query_alpha") == []

    def test_retrieve_top_k_respected(self) -> None:
        """retrieve(top_k=1) returns at most 1 result."""
        store = MemoryStore(embed_fn=_mock_embed, dim=4)
        store.add_case("case_alpha", "case_alpha")
        store.add_case("case_beta",  "case_beta")
        store.add_case("case_gamma", "case_gamma")

        results = store.retrieve("query_alpha", top_k=1)
        assert len(results) == 1

    def test_retrieve_score_high_for_exact_match(self) -> None:
        """IP score for a vector matched exactly should be close to 1.0."""
        store = MemoryStore(embed_fn=_mock_embed, dim=4)
        store.add_case("case_beta", "case_beta")

        results = store.retrieve("query_beta", top_k=1)
        assert len(results) == 1
        _cid, score = results[0]
        assert score > 0.99


# ---------------------------------------------------------------------------
# Pipeline + memory integration test
# ---------------------------------------------------------------------------

class TestPipelineMemoryIntegration:
    def test_pipeline_with_memory_populates_retrieved_cases(self) -> None:
        """Pipeline with a wired MemoryStore should populate session.retrieved_cases."""
        # Build a simple embed_fn that returns a fixed dim-4 vector
        fixed_vec = _unit_norm([0.5, 0.5, 0.5, 0.5])

        def _embed(text: str) -> list[float]:
            return fixed_vec

        memory = MemoryStore(embed_fn=_embed, dim=4)
        memory.add_case("BUG-001", "NullPointerException in login flow")
        memory.add_case("BUG-002", "IndexError when processing empty list")

        backend = MockModelBackend()
        pipeline = ChellPipeline(
            detector=StaticDetector(),
            query_gen=LLMQueryGenerator(backend=backend),
            refiner=LLMRefiner(backend=backend),
            validator=_AlwaysPassValidator(),
            memory=memory,
            max_turns=3,
        )
        responder = SimulatedResponder(expected_response="aggregate with sum")

        bug_report = BugReport(
            code="result = df.groupby('region')",
            task="Summarise sales by region",
            libs=["pandas"],
        )
        session = pipeline.debug(bug_report, responder)

        assert isinstance(session.retrieved_cases, list)
        assert len(session.retrieved_cases) > 0
        # Returned ids should be strings matching what we added
        for cid in session.retrieved_cases:
            assert isinstance(cid, str)

    def test_pipeline_without_memory_has_empty_retrieved_cases(self) -> None:
        """Pipeline without memory should leave retrieved_cases empty."""
        backend = MockModelBackend()
        pipeline = ChellPipeline(
            detector=StaticDetector(),
            query_gen=LLMQueryGenerator(backend=backend),
            refiner=LLMRefiner(backend=backend),
            validator=_AlwaysPassValidator(),
            memory=None,
            max_turns=3,
        )
        responder = SimulatedResponder(expected_response="aggregate with sum")

        bug_report = BugReport(
            code="result = df.groupby('region')",
            task="Summarise sales by region",
            libs=["pandas"],
        )
        session = pipeline.debug(bug_report, responder)

        assert session.retrieved_cases == []
