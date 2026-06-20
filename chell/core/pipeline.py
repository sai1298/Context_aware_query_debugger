from __future__ import annotations

from typing import TYPE_CHECKING

from chell.core.types import BugReport, DebugSession
from chell.core.responders import UserResponder
from chell.detection.base import ErrorDetector
from chell.query.base import QueryGenerator
from chell.refinement.base import Refiner
from chell.validation.base import Validator

if TYPE_CHECKING:
    from chell.config import ChellConfig
    from chell.memory.store import MemoryStore


class ChellPipeline:
    """Orchestrate the full Chell debugging loop.

    The pipeline drives interaction between the error detector, query
    generator, refiner, and validator, optionally augmenting with
    retrieved cases from a :class:`~chell.memory.store.MemoryStore`.

    Parameters
    ----------
    detector:
        An :class:`~chell.detection.base.ErrorDetector` that analyses a
        :class:`~chell.core.types.BugReport` and returns an
        :class:`~chell.core.types.ErrorDiagnosis`.
    query_gen:
        A :class:`~chell.query.base.QueryGenerator` that produces a
        :class:`~chell.core.types.ClarificationQuery` from the current
        :class:`~chell.core.types.DebugSession`.
    refiner:
        A :class:`~chell.refinement.base.Refiner` that generates a
        :class:`~chell.core.types.Correction` given the full session.
    validator:
        A :class:`~chell.validation.base.Validator` that checks whether
        a :class:`~chell.core.types.Correction` is acceptable.
    memory:
        Optional :class:`~chell.memory.store.MemoryStore` for
        retrieval-augmented debugging. When provided, similar historical
        cases are retrieved before refinement.
    max_turns:
        Maximum number of clarification turns before forcing refinement.
    """

    def __init__(
        self,
        detector: ErrorDetector,
        query_gen: QueryGenerator,
        refiner: Refiner,
        validator: Validator,
        memory: MemoryStore | None = None,
        max_turns: int = 5,
    ) -> None:
        self._detector = detector
        self._query_gen = query_gen
        self._refiner = refiner
        self._validator = validator
        self._memory = memory
        self._max_turns = max_turns

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def debug(
        self,
        bug_report: BugReport,
        responder: UserResponder,
        max_turns: int | None = None,
    ) -> DebugSession:
        """Run the interactive debugging loop for *bug_report*.

        Parameters
        ----------
        bug_report:
            The :class:`~chell.core.types.BugReport` submitted by the user.
        responder:
            A :class:`~chell.core.responders.UserResponder` that supplies
            answers to clarification questions (interactive or simulated).
        max_turns:
            Override for the max clarification turns set in the constructor.
            Useful when the evaluator needs per-run control.

        Returns
        -------
        DebugSession
            The fully populated session including diagnosis, retrieved
            cases, clarification turns, correction, and validation result.

        Algorithm
        ---------
        1. Create a :class:`~chell.core.types.DebugSession`.
        2. Run the detector to populate ``session.diagnosis``.
        3. If a memory store is present, retrieve similar cases.
        4. While the diagnosis is ambiguous AND turn budget remains:
           a. Generate a clarification query.
           b. Ask the responder; record the turn.
        5. Refine to produce a correction.
        6. Validate the correction.
        7. If validation failed AND turn budget remains: go to 4a.
        8. Return the session.
        """
        turn_budget = max_turns if max_turns is not None else self._max_turns

        # Step 1: create session
        session = DebugSession(bug_report=bug_report)

        # Step 2: detect the error
        session.diagnosis = self._detector.detect(bug_report)

        # Step 3: retrieve similar cases from memory (if available)
        if self._memory is not None:
            query_text = f"{bug_report.task}\n{bug_report.code}"
            hits = self._memory.retrieve(query_text, top_k=5)
            session.retrieved_cases = [case_id for case_id, _score in hits]

        # Step 4: clarification loop — only if ambiguous and budget allows
        if session.diagnosis.is_ambiguous and session.num_turns < turn_budget:
            query = self._query_gen.generate(session)
            response = responder.answer(query)
            session.add_turn(query, response)

        # Step 5: produce a correction
        session.correction = self._refiner.refine(session)

        # Step 6: validate the correction
        session.validation = self._validator.validate(session.correction)

        # Step 7: re-clarify and re-refine if validation failed and budget remains
        while (
            not session.validation.passed
            and session.num_turns < turn_budget
        ):
            query = self._query_gen.generate(session)
            response = responder.answer(query)
            session.add_turn(query, response)

            session.correction = self._refiner.refine(session)
            session.validation = self._validator.validate(session.correction)

        # Step 8: return the completed session
        return session

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_backend(cls, backend, max_turns: int = 5) -> "ChellPipeline":
        """Build a :class:`ChellPipeline` from a pre-built backend.

        A convenience factory for evaluation scripts that have already
        constructed a :class:`~chell.models.base.ModelBackend`.

        Parameters
        ----------
        backend:
            Any :class:`~chell.models.base.ModelBackend` (API or HF).
        max_turns:
            Maximum clarification turns (default 5).

        Returns
        -------
        ChellPipeline
        """
        from chell.detection.llm_detector import LLMDetector
        from chell.query.generator import LLMQueryGenerator
        from chell.refinement.refiner import LLMRefiner
        from chell.validation.validators import ExecutionValidator
        from chell.validation.executor import SandboxExecutor
        from chell.query.retriever import DPRRetriever
        from chell.memory.store import MemoryStore

        detector = LLMDetector(backend=backend)
        query_gen = LLMQueryGenerator(backend=backend)
        refiner = LLMRefiner(backend=backend)
        executor = SandboxExecutor()
        validator = ExecutionValidator(executor=executor)
        retriever = DPRRetriever()
        memory_store = MemoryStore(embed_fn=retriever.encode)

        return cls(
            detector=detector,
            query_gen=query_gen,
            refiner=refiner,
            validator=validator,
            memory=memory_store,
            max_turns=max_turns,
        )

    @classmethod
    def from_config(cls, config: ChellConfig) -> "ChellPipeline":
        """Build a :class:`ChellPipeline` from a :class:`~chell.config.ChellConfig`.

        Constructs concrete implementations of each pipeline component
        using the config values and the model registry.

        Parameters
        ----------
        config:
            A :class:`~chell.config.ChellConfig` (typically produced by
            :func:`~chell.config.load_config`).

        Returns
        -------
        ChellPipeline
            A fully assembled pipeline ready for use.
        """
        from chell.models.registry import build_backend
        from chell.detection.llm_detector import LLMDetector
        from chell.query.generator import LLMQueryGenerator
        from chell.refinement.refiner import LLMRefiner
        from chell.validation.validators import ExecutionValidator
        from chell.validation.executor import SandboxExecutor
        from chell.query.retriever import DPRRetriever
        from chell.memory.store import MemoryStore

        # Build the model backend from the model section of the config.
        model_cfg = config.model
        backend_dict: dict = {"backend": model_cfg.backend}
        if model_cfg.api:
            backend_dict["api"] = model_cfg.api
        if model_cfg.hf:
            backend_dict["hf"] = model_cfg.hf

        backend = build_backend(backend_dict)

        # Construct the default component implementations.
        detector = LLMDetector(backend=backend)
        query_gen = LLMQueryGenerator(backend=backend)
        refiner = LLMRefiner(backend=backend)
        executor = SandboxExecutor()
        validator = ExecutionValidator(executor=executor)

        # Wire up retrieval-augmented memory using DPRRetriever as the embed_fn.
        retriever = DPRRetriever()
        memory_store = MemoryStore(embed_fn=retriever.encode)

        return cls(
            detector=detector,
            query_gen=query_gen,
            refiner=refiner,
            validator=validator,
            memory=memory_store,
            max_turns=config.max_turns,
        )
