from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterator

from chell.data.schema import DebugCase, validate_case

# Optional PyTorch Dataset base class — not required at runtime
try:
    from torch.utils.data import Dataset as _TorchDataset  # type: ignore

    _TORCH_AVAILABLE = True
except ImportError:
    _TorchDataset = object  # fallback so the class statement below still works
    _TORCH_AVAILABLE = False


class ChellDataset(_TorchDataset):  # type: ignore[misc]
    """Iterable collection of :class:`~chell.data.schema.DebugCase` objects.

    Parameters
    ----------
    cases_dir:
        Path to the directory containing individual ``*.json`` case files
        (typically ``data/curated/``).
    split:
        One of ``"train"``, ``"val"``, or ``"test"``.  When a matching
        ``data/splits/{split}.json`` file exists (written by
        :class:`~chell.data.builder.DatasetBuilder`) it is used to filter
        which case IDs belong to this split.  Otherwise *all* cases in
        *cases_dir* are loaded (useful during initial data collection before
        splits are generated).
    """

    def __init__(self, cases_dir: str, split: str = "train") -> None:
        self._cases_dir = Path(cases_dir)
        self._split = split
        self._cases: list[DebugCase] = []
        self._load()

    # ------------------------------------------------------------------
    # Internal loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load cases from disk, optionally filtered by split file."""
        all_cases = self._load_all_from_dir(self._cases_dir)

        split_file = self._cases_dir.parent / "splits" / f"{self._split}.json"
        if split_file.exists():
            with split_file.open() as fh:
                ids_in_split: set[str] = set(json.load(fh))
            self._cases = [c for c in all_cases if c.id in ids_in_split]
        else:
            # No split file — return everything (dev / first-run scenario)
            self._cases = all_cases

    @staticmethod
    def _load_all_from_dir(directory: Path) -> list[DebugCase]:
        cases: list[DebugCase] = []
        for json_file in sorted(directory.glob("*.json")):
            with json_file.open() as fh:
                data = json.load(fh)
            try:
                validate_case(data)
            except Exception as exc:
                raise ValueError(
                    f"Case file {json_file} failed schema validation: {exc}"
                ) from exc
            cases.append(DebugCase.from_dict(data))
        return cases

    # ------------------------------------------------------------------
    # Sequence protocol (works with or without PyTorch)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._cases)

    def __getitem__(self, idx: int) -> DebugCase:
        return self._cases[idx]

    def __iter__(self) -> Iterator[DebugCase]:
        return iter(self._cases)

    # ------------------------------------------------------------------
    # Split generation
    # ------------------------------------------------------------------

    def load_splits(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
    ) -> dict[str, list[DebugCase]]:
        """Deterministically partition cases into train / val / test.

        The split is derived from a hash of each case's ``id`` so that adding
        new cases does not shuffle existing assignments.

        Parameters
        ----------
        train_ratio:
            Fraction of cases assigned to ``"train"`` (default 0.70).
        val_ratio:
            Fraction of cases assigned to ``"val"`` (default 0.15).
            The remaining ``1 - train_ratio - val_ratio`` goes to ``"test"``.

        Returns
        -------
        dict[str, list[DebugCase]]
            Keys are ``"train"``, ``"val"``, ``"test"``.
        """
        if not (0.0 < train_ratio < 1.0):
            raise ValueError("train_ratio must be in (0, 1)")
        if not (0.0 < val_ratio < 1.0):
            raise ValueError("val_ratio must be in (0, 1)")
        if train_ratio + val_ratio >= 1.0:
            raise ValueError("train_ratio + val_ratio must be < 1.0")

        # Load all cases regardless of the current split filter
        all_cases = self._load_all_from_dir(self._cases_dir)

        splits: dict[str, list[DebugCase]] = {"train": [], "val": [], "test": []}
        for case in all_cases:
            # Stable hash: take last 8 hex digits of SHA-256 → 0..0xFFFFFFFF
            digest = hashlib.sha256(case.id.encode()).hexdigest()[-8:]
            bucket = int(digest, 16) / 0xFFFF_FFFF  # float in [0, 1]
            if bucket < train_ratio:
                splits["train"].append(case)
            elif bucket < train_ratio + val_ratio:
                splits["val"].append(case)
            else:
                splits["test"].append(case)

        return splits
