from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import yaml  # type: ignore[import]


@dataclass
class ModelConfig:
    """Configuration for the model backend."""

    backend: str = "api"
    api: Optional[dict] = None   # provider, model, max_tokens, temperature
    hf: Optional[dict] = None    # model_name, device, load_in_4bit, max_new_tokens

    def __post_init__(self) -> None:
        if self.api is None:
            self.api = {}
        if self.hf is None:
            self.hf = {}


@dataclass
class CAAMConfig:
    """Configuration for the Context-Aware Attention Modulation module."""

    enabled: bool = True
    context_dim: int = 256
    scale: float = 0.1
    zero_init: bool = True


@dataclass
class TrainConfig:
    """Configuration for fine-tuning / training runs."""

    batch_size: int = 8
    learning_rate: float = 5e-5
    num_epochs: int = 3
    optimizer: str = "adamw"
    warmup_steps: int = 100
    max_seq_len: int = 1024
    lora: dict = field(default_factory=lambda: {
        "enabled": True,
        "r": 16,
        "alpha": 32,
        "target_modules": ["q_proj", "v_proj"],
        "dropout": 0.05,
    })
    output_dir: str = "results/checkpoints"


@dataclass
class ChellConfig:
    """Top-level configuration object for the Chell pipeline."""

    model: ModelConfig
    caam: CAAMConfig
    train: TrainConfig
    max_turns: int = 5
    data_dir: str = "data"


def load_config(
    model_yaml: str,
    caam_yaml: Optional[str] = None,
    train_yaml: Optional[str] = None,
) -> ChellConfig:
    """Load a :class:`ChellConfig` from YAML files.

    Parameters
    ----------
    model_yaml:
        Path to the model backend configuration YAML file.
    caam_yaml:
        Optional path to the CAAM configuration YAML file.
        If *None*, defaults are used.
    train_yaml:
        Optional path to the training configuration YAML file.
        If *None*, defaults are used.

    Returns
    -------
    ChellConfig
        A fully populated configuration object.
    """
    # --- model config ---
    with open(model_yaml, "r", encoding="utf-8") as fh:
        model_data: dict = yaml.safe_load(fh) or {}

    model_cfg = ModelConfig(
        backend=model_data.get("backend", "api"),
        api=model_data.get("api") or {},
        hf=model_data.get("hf") or {},
    )

    # --- CAAM config ---
    if caam_yaml is not None:
        with open(caam_yaml, "r", encoding="utf-8") as fh:
            caam_data: dict = yaml.safe_load(fh) or {}
        caam_cfg = CAAMConfig(
            enabled=caam_data.get("enabled", True),
            context_dim=caam_data.get("context_dim", 256),
            scale=caam_data.get("scale", 0.1),
            zero_init=caam_data.get("zero_init", True),
        )
    else:
        caam_cfg = CAAMConfig()

    # --- train config ---
    if train_yaml is not None:
        with open(train_yaml, "r", encoding="utf-8") as fh:
            train_data: dict = yaml.safe_load(fh) or {}
        train_cfg = TrainConfig(
            batch_size=train_data.get("batch_size", 8),
            learning_rate=train_data.get("learning_rate", 5e-5),
            num_epochs=train_data.get("num_epochs", 3),
            optimizer=train_data.get("optimizer", "adamw"),
            warmup_steps=train_data.get("warmup_steps", 100),
            max_seq_len=train_data.get("max_seq_len", 1024),
            lora=train_data.get("lora", {
                "enabled": True,
                "r": 16,
                "alpha": 32,
                "target_modules": ["q_proj", "v_proj"],
                "dropout": 0.05,
            }),
            output_dir=train_data.get("output_dir", "results/checkpoints"),
        )
    else:
        train_cfg = TrainConfig()

    return ChellConfig(
        model=model_cfg,
        caam=caam_cfg,
        train=train_cfg,
    )
