from __future__ import annotations

import yaml  # type: ignore[import]

from chell.models.base import ModelBackend
from chell.models.api_backend import APIBackend
from chell.models.hf_backend import HFBackend
from chell.models.mock import MockModelBackend


def build_backend(config: dict) -> ModelBackend:
    """Construct a ModelBackend from a plain configuration dictionary.

    Expected shapes
    ---------------
    ``{"backend": "api",  "api":  {"provider": "anthropic", ...}}``
    ``{"backend": "hf",   "hf":   {"model": "bigcode/starcoder2-7b"}}``
    ``{"backend": "mock"}``

    Parameters
    ----------
    config:
        Dictionary with at least a ``"backend"`` key.

    Returns
    -------
    ModelBackend
        A fully initialised backend instance.

    Raises
    ------
    ValueError
        If ``config["backend"]`` is not one of the recognised values.
    """
    backend_type = config.get("backend")

    if backend_type == "api":
        api_config: dict = config.get("api", {})
        return APIBackend(**api_config)

    if backend_type == "hf":
        hf_config: dict = config.get("hf", {})
        return HFBackend(**hf_config)

    if backend_type == "mock":
        return MockModelBackend()

    raise ValueError(
        f"Unknown backend type {backend_type!r}. "
        "Supported values: 'api', 'hf', 'mock'."
    )


def build_backend_from_yaml(path: str) -> ModelBackend:
    """Load a YAML config file and delegate to :func:`build_backend`.

    Parameters
    ----------
    path:
        Absolute or relative path to a YAML file whose top-level keys match
        the shape expected by :func:`build_backend`.

    Returns
    -------
    ModelBackend
        A fully initialised backend instance.
    """
    with open(path, "r", encoding="utf-8") as fh:
        config: dict = yaml.safe_load(fh)
    return build_backend(config)
