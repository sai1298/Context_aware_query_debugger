"""tests/test_training.py

M5 tests: ChellCollator and ChellTrainer.

Smoke-tests the training machinery using a tiny in-memory GPT-2 model
(no download required) and a char-level mock tokenizer.  Also verifies
that Wc receives gradients and moves off zero when non-zero context is
provided through ContextAwareAttention.
"""
from __future__ import annotations

import pytest

# Guard all tests on torch + transformers availability.
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

from chell.data.schema import DebugCase
from chell.models.base import ModelBackend
from chell.training.collator import ChellCollator
from chell.training.trainer import ChellTrainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MockTokenizer:
    """Character-level tokenizer (char-code mod vocab_size → int ID)."""

    def __init__(self, vocab_size: int = 128) -> None:
        self.pad_token_id = 0
        self.eos_token = "<eos>"
        self._vocab = vocab_size

    def save_pretrained(self, path: str, **kwargs: object) -> None:
        pass

    def __call__(
        self,
        text: str,
        max_length: int = 32,
        truncation: bool = True,
        padding: str | None = None,
        return_tensors: str | None = None,
        add_special_tokens: bool = True,
    ) -> dict:
        ids = [ord(c) % self._vocab for c in text]
        if truncation and max_length:
            ids = ids[:max_length]
        if padding == "max_length" and max_length:
            pad_len = max_length - len(ids)
            attn = [1] * len(ids) + [0] * pad_len
            ids = ids + [0] * pad_len
        else:
            attn = [1] * len(ids)

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.tensor([attn], dtype=torch.long),
            }
        return {"input_ids": ids, "attention_mask": attn}


class _MockHFBackend(ModelBackend):
    """Minimal backend wrapping a real nn.Module + mock tokenizer."""

    def __init__(self, model: nn.Module, tokenizer: _MockTokenizer) -> None:
        self._model = model
        self.tokenizer = tokenizer

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        return "[mock]"

    def embed(self, text: str) -> list[float]:
        return [0.0] * 64

    @property
    def torch_model(self) -> nn.Module:
        return self._model


def _make_debug_cases(n: int = 3) -> list[DebugCase]:
    cases = []
    for i in range(n):
        cases.append(
            DebugCase(
                id=f"t{i}",
                buggy_code=f"x = {i}",
                task="compute value",
                libs=[],
                error_type="off_by_one",
                error_location="line 1",
                clarification_query="Is x correct?",
                clarification_options=["yes", "no"],
                expected_user_response="no",
                corrected_code=f"x = {i + 1}",
                explanation="off by one",
            )
        )
    return cases


def _make_tiny_gpt2(vocab_size: int = 128, seq_len: int = 32) -> GPT2LMHeadModel:
    cfg = GPT2Config(
        n_embd=32,
        n_layer=1,
        n_head=2,
        n_positions=seq_len * 2,
        vocab_size=vocab_size,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        summary_first_dropout=0.0,
    )
    return GPT2LMHeadModel(cfg)


# ---------------------------------------------------------------------------
# ChellCollator tests
# ---------------------------------------------------------------------------

class TestChellCollator:
    MAX_SEQ = 32

    @pytest.fixture()
    def collator(self) -> ChellCollator:
        return ChellCollator(_MockTokenizer(), max_seq_len=self.MAX_SEQ)

    @pytest.fixture()
    def batch(self) -> list[DebugCase]:
        return _make_debug_cases(2)

    def test_output_keys(self, collator: ChellCollator, batch: list[DebugCase]) -> None:
        out = collator(batch)
        assert set(out) == {"input_ids", "attention_mask", "labels"}

    def test_tensor_shapes(self, collator: ChellCollator, batch: list[DebugCase]) -> None:
        out = collator(batch)
        B = len(batch)
        for key in ("input_ids", "attention_mask", "labels"):
            assert out[key].shape == (B, self.MAX_SEQ), f"{key} shape mismatch"

    def test_label_masking_prefix(self, collator: ChellCollator) -> None:
        case = _make_debug_cases(1)[0]
        out = collator([case])
        labels = out["labels"][0]
        # The prefix up to and including [FIX] must be masked with -100.
        # There must be at least one -100 at the start.
        assert labels[0].item() == -100

    def test_label_masking_padding(self, collator: ChellCollator) -> None:
        case = _make_debug_cases(1)[0]
        out = collator([case])
        labels = out["labels"][0]
        attn = out["attention_mask"][0]
        # Wherever attention_mask == 0 (padding), label must be -100.
        pad_positions = (attn == 0).nonzero(as_tuple=True)[0]
        if len(pad_positions) > 0:
            assert (labels[pad_positions] == -100).all()

    def test_input_ids_in_vocab(self, collator: ChellCollator, batch: list[DebugCase]) -> None:
        out = collator(batch)
        assert out["input_ids"].max().item() < 128
        assert out["input_ids"].min().item() >= 0


# ---------------------------------------------------------------------------
# ChellTrainer tests
# ---------------------------------------------------------------------------

class TestChellTrainer:

    def test_rejects_api_backend(self, tmp_path: pytest.FixtureRequest) -> None:
        from chell.models.mock import MockModelBackend
        backend = MockModelBackend()
        assert backend.torch_model is None

        trainer = ChellTrainer(
            backend=backend,
            dataset=_make_debug_cases(2),
            config={"num_epochs": 1, "batch_size": 2, "output_dir": str(tmp_path)},
            use_lora=False,
        )
        with pytest.raises(ValueError, match="torch_model"):
            trainer.train()

    def test_smoke_no_lora(self, tmp_path: pytest.FixtureRequest) -> None:
        """train() with a tiny GPT-2 completes and returns loss + steps."""
        tokenizer = _MockTokenizer(vocab_size=128)
        model = _make_tiny_gpt2(vocab_size=128, seq_len=32)
        backend = _MockHFBackend(model, tokenizer)

        cases = _make_debug_cases(3)
        config = {
            "num_epochs": 1,
            "batch_size": 2,
            "learning_rate": 1e-4,
            "warmup_steps": 0,
            "max_seq_len": 32,
            "output_dir": str(tmp_path),
        }
        trainer = ChellTrainer(backend=backend, dataset=cases, config=config, use_lora=False)
        metrics = trainer.train()

        assert "loss" in metrics
        assert "steps" in metrics
        assert isinstance(metrics["loss"], float)
        assert metrics["loss"] == metrics["loss"]  # not NaN


# ---------------------------------------------------------------------------
# CAAM Wc gradient test (verifies Wc moves off zero with non-zero context)
# ---------------------------------------------------------------------------

class TestCAAMWcGradient:

    def test_wc_starts_zero(self) -> None:
        from chell.caam.attention import ContextAwareAttention
        caam = ContextAwareAttention(num_heads=2, head_dim=8, context_dim=16)
        assert torch.all(caam.Wc.weight == 0), "Wc must be zero-initialized"

    def test_wc_receives_gradient_with_nonzero_context(self) -> None:
        from chell.caam.attention import ContextAwareAttention

        context_dim, head_dim, num_heads = 16, 8, 2
        B, T = 1, 4

        caam = ContextAwareAttention(
            num_heads=num_heads, head_dim=head_dim, context_dim=context_dim, scale=0.1
        )

        q = torch.randn(B, num_heads, T, head_dim)
        k = torch.randn(B, num_heads, T, head_dim)
        v = torch.randn(B, num_heads, T, head_dim)
        context = torch.randn(1, context_dim)  # non-zero context

        out = caam(q, k, v, context)
        out.sum().backward()

        assert caam.Wc.weight.grad is not None
        assert not torch.all(caam.Wc.weight.grad == 0), "Wc should receive non-zero gradient"

    def test_wc_moves_off_zero_after_step(self) -> None:
        from chell.caam.attention import ContextAwareAttention

        context_dim, head_dim, num_heads = 16, 8, 2
        B, T = 1, 4

        caam = ContextAwareAttention(
            num_heads=num_heads, head_dim=head_dim, context_dim=context_dim, scale=0.1
        )
        assert torch.all(caam.Wc.weight == 0)

        optim = torch.optim.SGD(caam.parameters(), lr=0.1)

        q = torch.randn(B, num_heads, T, head_dim)
        k = torch.randn(B, num_heads, T, head_dim)
        v = torch.randn(B, num_heads, T, head_dim)
        context = torch.randn(1, context_dim)

        out = caam(q, k, v, context)
        out.sum().backward()
        optim.step()

        assert not torch.all(caam.Wc.weight == 0), "Wc must move off zero after optimizer step"
