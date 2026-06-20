from __future__ import annotations

from typing import Any

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from transformers import TrainingArguments, Trainer  # type: ignore[import]

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from peft import get_peft_model, LoraConfig, TaskType  # type: ignore[import]

    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

from chell.models.base import ModelBackend
from chell.training.collator import ChellCollator


class ChellTrainer:
    """Fine-tunes a local HuggingFace model on a :class:`~chell.data.schema.DebugCase`
    dataset using the Chell training format.

    The trainer integrates three optional capabilities:

    * **CAAM** — patches the model's attention layers via
      :class:`~chell.caam.patcher.CAAMPatcher` when *caam_config* is provided.
    * **LoRA** — wraps the model with PEFT LoRA adapters when *use_lora* is
      ``True`` (requires the ``peft`` package).
    * **HF Trainer** — delegates the actual training loop to
      :class:`transformers.Trainer`.

    Args:
        backend: A :class:`~chell.models.base.ModelBackend` instance.
            Must expose a non-``None`` :attr:`~chell.models.base.ModelBackend.torch_model`
            property (i.e. an :class:`~chell.models.hf_backend.HFBackend`);
            :class:`~chell.models.api_backend.APIBackend` instances will raise
            :class:`ValueError` at :meth:`train` time.
        dataset: A dataset of :class:`~chell.data.schema.DebugCase` objects.
            Anything that behaves like a ``torch.utils.data.Dataset`` or a plain
            list is accepted; it is passed directly to :class:`transformers.Trainer`.
        config: Training hyper-parameters.  Expected keys mirror ``configs/train.yaml``::

                batch_size: 8
                learning_rate: 5.0e-5
                num_epochs: 3
                optimizer: adamw
                warmup_steps: 100
                max_seq_len: 1024
                lora:
                  r: 16
                  alpha: 32
                  target_modules: ["q_proj", "v_proj"]
                  dropout: 0.05
                output_dir: results/checkpoints

        caam_config: Optional dict of CAAM hyper-parameters (``context_dim``,
            ``scale``, ``arch``).  When provided the model is patched before
            LoRA is applied.
        use_lora: Apply LoRA adapters via ``peft``.  Defaults to ``True``.
    """

    def __init__(
        self,
        backend: ModelBackend,
        dataset: Any,
        config: dict,
        caam_config: dict | None = None,
        use_lora: bool = True,
    ) -> None:
        self.backend = backend
        self.dataset = dataset
        self.config = config
        self.caam_config = caam_config
        self.use_lora = use_lora

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train(self) -> dict[str, float]:
        """Run fine-tuning and return final training metrics.

        Steps:

        1. Retrieve ``nn.Module`` from the backend (raises for API backends).
        2. Optionally patch attention layers with CAAM.
        3. Optionally wrap with LoRA adapters.
        4. Run :class:`transformers.Trainer`.

        Returns:
            A dict with at least ``{"loss": float, "steps": int}``.

        Raises:
            ImportError: If ``torch``, ``transformers``, or ``peft`` (when
                ``use_lora=True``) are not installed.
            ValueError: If the backend does not expose a local ``nn.Module``
                (i.e. :attr:`~chell.models.base.ModelBackend.torch_model` is ``None``).
        """
        if not HAS_TORCH:
            raise ImportError(
                "ChellTrainer requires PyTorch. "
                "Install it with: pip install torch"
            )
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "ChellTrainer requires the 'transformers' package. "
                "Install it with: pip install transformers"
            )

        # 1. Obtain the local nn.Module ----------------------------------------
        model = self.backend.torch_model
        if model is None:
            raise ValueError(
                "ChellTrainer requires a backend that exposes a local nn.Module "
                "via .torch_model (e.g. HFBackend). "
                "APIBackend does not support fine-tuning."
            )

        # 2. Apply CAAM attention patch ----------------------------------------
        if self.caam_config is not None:
            model = self._apply_caam(model)

        # 3. Apply LoRA adapters -----------------------------------------------
        if self.use_lora:
            model = self._apply_lora(model)

        # 4. Build collator and tokenizer reference ----------------------------
        tokenizer = self._get_tokenizer()
        max_seq_len: int = self.config.get("max_seq_len", 1024)
        collator = ChellCollator(tokenizer, max_seq_len=max_seq_len)

        # 5. Build HuggingFace TrainingArguments ------------------------------
        output_dir: str = self.config.get("output_dir", "results/checkpoints")
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.get("num_epochs", 3),
            per_device_train_batch_size=self.config.get("batch_size", 8),
            learning_rate=float(self.config.get("learning_rate", 5e-5)),
            warmup_steps=self.config.get("warmup_steps", 100),
            optim=self._resolve_optimizer(self.config.get("optimizer", "adamw")),
            logging_steps=10,
            save_strategy="epoch",
            report_to="none",
        )

        # 6. Run training ------------------------------------------------------
        hf_trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.dataset,
            data_collator=collator,
        )
        train_output = hf_trainer.train()

        return {
            "loss": float(train_output.training_loss),
            "steps": int(train_output.global_step),
        }

    def evaluate(self, eval_dataset: Any) -> dict[str, float]:
        """Evaluate the backend model on *eval_dataset*.

        Args:
            eval_dataset: A dataset of :class:`~chell.data.schema.DebugCase`
                objects (same contract as the training dataset).

        Returns:
            A dict of evaluation metrics as returned by
            :meth:`transformers.Trainer.evaluate`.

        Raises:
            ImportError: If ``torch`` or ``transformers`` are not installed.
            ValueError: If the backend does not expose a local ``nn.Module``.
        """
        if not HAS_TORCH:
            raise ImportError(
                "ChellTrainer requires PyTorch. "
                "Install it with: pip install torch"
            )
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "ChellTrainer requires the 'transformers' package. "
                "Install it with: pip install transformers"
            )

        model = self.backend.torch_model
        if model is None:
            raise ValueError(
                "ChellTrainer.evaluate() requires a backend that exposes a local "
                "nn.Module via .torch_model. APIBackend does not support evaluation."
            )

        tokenizer = self._get_tokenizer()
        max_seq_len: int = self.config.get("max_seq_len", 1024)
        collator = ChellCollator(tokenizer, max_seq_len=max_seq_len)

        output_dir: str = self.config.get("output_dir", "results/checkpoints")
        eval_args = TrainingArguments(
            output_dir=output_dir,
            per_device_eval_batch_size=self.config.get("batch_size", 8),
            report_to="none",
        )

        hf_trainer = Trainer(
            model=model,
            args=eval_args,
            eval_dataset=eval_dataset,
            data_collator=collator,
        )
        metrics = hf_trainer.evaluate()
        return {k: float(v) for k, v in metrics.items()}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_caam(self, model: Any) -> Any:
        """Patch *model* attention layers in-place using CAAMPatcher."""
        from chell.caam.patcher import CAAMPatcher  # local import to avoid hard dep

        context_dim: int = self.caam_config.get("context_dim", 256)  # type: ignore[union-attr]
        scale: float = float(self.caam_config.get("scale", 0.1))  # type: ignore[union-attr]
        arch: str = self.caam_config.get("arch", "")  # type: ignore[union-attr]

        if not arch:
            raise ValueError(
                "caam_config must include an 'arch' key "
                "(e.g. 'phi2', 'qwen2', 'gptbigcode', 'llama')."
            )

        patcher = CAAMPatcher(context_dim=context_dim, scale=scale)
        return patcher.patch(model, arch=arch)

    def _apply_lora(self, model: Any) -> Any:
        """Wrap *model* with LoRA adapters via peft."""
        if not HAS_PEFT:
            raise ImportError(
                "LoRA fine-tuning requires the 'peft' package. "
                "Install it with: pip install peft"
            )

        lora_cfg: dict = self.config.get("lora", {})
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("alpha", 32),
            target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            bias="none",
        )
        return get_peft_model(model, lora_config)

    def _get_tokenizer(self) -> Any:
        """Retrieve the tokenizer associated with the backend.

        HFBackend is expected to expose a ``tokenizer`` attribute.  When it
        does not (e.g. a mock or partially constructed backend), a helpful
        :class:`AttributeError` is raised.
        """
        if not hasattr(self.backend, "tokenizer"):
            raise AttributeError(
                "ChellTrainer expects backend.tokenizer to be set. "
                "Ensure the HFBackend initialises self.tokenizer before training."
            )
        return self.backend.tokenizer  # type: ignore[union-attr]

    @staticmethod
    def _resolve_optimizer(name: str) -> str:
        """Map a human-friendly optimizer name to a HF TrainingArguments optim key."""
        _MAP = {
            "adamw": "adamw_torch",
            "adamw_torch": "adamw_torch",
            "adam": "adamw_torch",
            "adafactor": "adafactor",
            "sgd": "sgd",
        }
        resolved = _MAP.get(name.lower())
        if resolved is None:
            # Fall back gracefully — HF will surface a clear error if invalid.
            return name
        return resolved
