from __future__ import annotations

import ast
import difflib
import json
from collections import defaultdict
from pathlib import Path

import jsonschema

from chell.data.schema import DebugCase, validate_case


class DatasetBuilder:
    """Accumulate, validate, deduplicate and persist :class:`~chell.data.schema.DebugCase` objects.

    Typical usage::

        builder = DatasetBuilder()
        builder.add_case(case_a)
        builder.add_case(case_b)
        errors = builder.validate_all()   # check all cases
        removed = builder.dedup()         # remove near-duplicates
        builder.save("data/")             # write JSON files + split manifests
        print(builder.stats())
    """

    def __init__(self) -> None:
        self._cases: list[DebugCase] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_case(self, case: DebugCase) -> None:
        """Append *case* to the builder's internal collection."""
        self._cases.append(case)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_all(self) -> list[str]:
        """Validate every case against the JSON Schema.

        Returns
        -------
        list[str]
            One human-readable error message per failing case (empty list
            means all cases are valid).
        """
        errors: list[str] = []
        for case in self._cases:
            try:
                validate_case(case.to_dict())
            except jsonschema.ValidationError as exc:
                errors.append(f"[{case.id}] {exc.message}")
        return errors

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    @staticmethod
    def _token_similarity(a: str, b: str) -> float:
        """SequenceMatcher ratio on whitespace-normalised source tokens."""
        a_tokens = a.split()
        b_tokens = b.split()
        return difflib.SequenceMatcher(None, a_tokens, b_tokens).ratio()

    @staticmethod
    def _ast_similarity(a: str, b: str) -> float:
        """Structural similarity via AST dump string comparison.

        Falls back to token similarity when either snippet fails to parse.
        """
        try:
            a_dump = ast.dump(ast.parse(a))
            b_dump = ast.dump(ast.parse(b))
            return difflib.SequenceMatcher(None, a_dump.split(), b_dump.split()).ratio()
        except SyntaxError:
            return DatasetBuilder._token_similarity(a, b)

    def dedup(self, threshold: float = 0.9) -> int:
        """Remove near-duplicate cases by comparing buggy-code similarity.

        Two cases are considered near-duplicates when their AST similarity
        score exceeds *threshold*.  The first occurrence is kept; subsequent
        near-duplicates are removed.

        Parameters
        ----------
        threshold:
            Similarity score (0–1) above which two cases are deemed
            duplicates.  Default is 0.9.

        Returns
        -------
        int
            Number of cases removed.
        """
        if not (0.0 < threshold <= 1.0):
            raise ValueError("threshold must be in (0, 1]")

        kept: list[DebugCase] = []
        removed = 0

        for candidate in self._cases:
            is_dup = False
            for existing in kept:
                sim = self._ast_similarity(candidate.buggy_code, existing.buggy_code)
                if sim >= threshold:
                    is_dup = True
                    break
            if is_dup:
                removed += 1
            else:
                kept.append(candidate)

        self._cases = kept
        return removed

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, output_dir: str) -> None:
        """Persist cases to disk.

        Writes:
        - ``{output_dir}/curated/{case.id}.json`` — one file per case
        - ``{output_dir}/splits/train.json``
        - ``{output_dir}/splits/val.json``
        - ``{output_dir}/splits/test.json``

        The split assignment reuses the deterministic hash logic from
        :meth:`~chell.data.dataset.ChellDataset.load_splits`.

        Parameters
        ----------
        output_dir:
            Root directory under which ``curated/`` and ``splits/``
            sub-directories will be created.
        """
        import hashlib

        root = Path(output_dir)
        curated_dir = root / "curated"
        splits_dir = root / "splits"
        curated_dir.mkdir(parents=True, exist_ok=True)
        splits_dir.mkdir(parents=True, exist_ok=True)

        split_ids: dict[str, list[str]] = {"train": [], "val": [], "test": []}

        for case in self._cases:
            # Write individual file
            out_file = curated_dir / f"{case.id}.json"
            with out_file.open("w") as fh:
                json.dump(case.to_dict(), fh, indent=2)

            # Assign to split (same algorithm as ChellDataset.load_splits)
            digest = hashlib.sha256(case.id.encode()).hexdigest()[-8:]
            bucket = int(digest, 16) / 0xFFFF_FFFF
            if bucket < 0.70:
                split_ids["train"].append(case.id)
            elif bucket < 0.85:
                split_ids["val"].append(case.id)
            else:
                split_ids["test"].append(case.id)

        for split_name, ids in split_ids.items():
            split_file = splits_dir / f"{split_name}.json"
            with split_file.open("w") as fh:
                json.dump(sorted(ids), fh, indent=2)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return counts broken down by category and difficulty.

        Returns
        -------
        dict
            ``{"total": int, "by_category": {...}, "by_difficulty": {...}}``
        """
        by_category: dict[str, int] = defaultdict(int)
        by_difficulty: dict[str, int] = defaultdict(int)

        for case in self._cases:
            by_category[case.category] += 1
            by_difficulty[case.difficulty] += 1

        return {
            "total": len(self._cases),
            "by_category": dict(by_category),
            "by_difficulty": dict(by_difficulty),
        }
