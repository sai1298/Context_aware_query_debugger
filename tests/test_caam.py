from __future__ import annotations

import math

import pytest

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


# ---------------------------------------------------------------------------
# Mock attention modules — mirror the attribute structure expected by adapters
# ---------------------------------------------------------------------------

if HAS_TORCH:

    class _MockPhi2Attn(nn.Module):
        """Toy Phi-2-style attention with separate q/k/v projections and ``dense`` output."""

        def __init__(self, hidden_size: int = 32, num_heads: int = 4) -> None:
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = hidden_size // num_heads
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.dense = nn.Linear(hidden_size, hidden_size, bias=False)

        def forward(self, hidden_states: torch.Tensor, **kwargs: object) -> torch.Tensor:
            B, T, C = hidden_states.shape
            q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            weights = F.softmax(scores, dim=-1)
            out = (weights @ v).transpose(1, 2).contiguous().view(B, T, -1)
            return self.dense(out)

    class _MockQwen2Attn(nn.Module):
        """Toy Qwen2-style attention with separate q/k/v projections and ``o_proj`` output."""

        def __init__(self, hidden_size: int = 32, num_heads: int = 4) -> None:
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = hidden_size // num_heads
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        def forward(self, hidden_states: torch.Tensor, **kwargs: object) -> torch.Tensor:
            B, T, C = hidden_states.shape
            q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            weights = F.softmax(scores, dim=-1)
            out = (weights @ v).transpose(1, 2).contiguous().view(B, T, -1)
            return self.o_proj(out)

    class _MockLlamaAttn(nn.Module):
        """Toy Llama-style attention with separate q/k/v projections and ``o_proj`` output."""

        def __init__(self, hidden_size: int = 32, num_heads: int = 4) -> None:
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = hidden_size // num_heads
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        def forward(self, hidden_states: torch.Tensor, **kwargs: object) -> torch.Tensor:
            B, T, C = hidden_states.shape
            q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            weights = F.softmax(scores, dim=-1)
            out = (weights @ v).transpose(1, 2).contiguous().view(B, T, -1)
            return self.o_proj(out)

    class _MockGPTBigCodeAttn(nn.Module):
        """Toy GPTBigCode-style attention with combined ``c_attn`` QKV and ``c_proj`` output."""

        def __init__(self, hidden_size: int = 32, num_heads: int = 4) -> None:
            super().__init__()
            self.num_heads = num_heads
            self.head_size = hidden_size // num_heads  # note: head_size, not head_dim
            # Combined QKV projection: out_features = 3 * hidden_size (MHA)
            self.c_attn = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
            self.c_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        def forward(self, hidden_states: torch.Tensor, **kwargs: object) -> torch.Tensor:
            B, T, C = hidden_states.shape
            hidden_size = self.num_heads * self.head_size
            qkv = self.c_attn(hidden_states)
            q, k, v = qkv.split(hidden_size, dim=-1)
            q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
            k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
            v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)
            weights = F.softmax(scores, dim=-1)
            out = (weights @ v).transpose(1, 2).contiguous().view(B, T, -1)
            return self.c_proj(out)

    # ---------------------------------------------------------------------------
    # Mock model containers — mirror the layer/attribute nesting adapters traverse
    # ---------------------------------------------------------------------------

    def _make_layer(attn_attr: str, attn_module: nn.Module) -> nn.Module:
        """Build a bare nn.Module with a named attribute for the attention."""
        layer = nn.Module()
        setattr(layer, attn_attr, attn_module)
        return layer

    class _MockPhi2Model(nn.Module):
        """Two-layer mock model with Phi-2-style structure."""

        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([
                _make_layer("self_attn", _MockPhi2Attn())
                for _ in range(2)
            ])

    class _MockQwen2Model(nn.Module):
        """Two-layer mock model with Qwen2-style structure."""

        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([
                _make_layer("self_attn", _MockQwen2Attn())
                for _ in range(2)
            ])

    class _MockLlamaModel(nn.Module):
        """Two-layer mock model with Llama-style structure."""

        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([
                _make_layer("self_attn", _MockLlamaAttn())
                for _ in range(2)
            ])

    class _MockGPTBigCodeModel(nn.Module):
        """Two-layer mock model with GPTBigCode-style structure."""

        def __init__(self) -> None:
            super().__init__()
            self.transformer = nn.Module()
            self.transformer.h = nn.ModuleList([
                _make_layer("attn", _MockGPTBigCodeAttn())
                for _ in range(2)
            ])


# ---------------------------------------------------------------------------
# Helper: run a single-layer forward pass before and after patching
# ---------------------------------------------------------------------------

def _base_forward_phi2(model: "nn.Module", hidden: "torch.Tensor") -> "torch.Tensor":
    with torch.no_grad():
        return model.model.layers[0].self_attn(hidden)


def _patched_forward_phi2(
    model: "nn.Module",
    hidden: "torch.Tensor",
    ctx: "torch.Tensor",
) -> "torch.Tensor":
    for layer in model.model.layers:
        layer.self_attn.set_context(ctx)
    with torch.no_grad():
        return model.model.layers[0].self_attn(hidden)


# ---------------------------------------------------------------------------
# Zero-init equivalence tests
# ---------------------------------------------------------------------------

def test_phi2_zero_init_equivalence() -> None:
    """Patched Phi-2 output must equal base output when Wc=0 (zero-initialized)."""
    from chell.caam.adapters.phi2 import patch_phi2

    model = _MockPhi2Model()
    model.eval()

    hidden = torch.randn(1, 5, 32)
    base_out = _base_forward_phi2(model, hidden)

    patch_phi2(model, context_dim=16, scale=0.1)
    model.eval()

    ctx = torch.zeros(1, 16)
    patched_out = _patched_forward_phi2(model, hidden, ctx)

    assert torch.allclose(base_out, patched_out, atol=1e-5), (
        "Zero-init Wc must not change Phi-2 attention output"
    )


def test_qwen2_zero_init_equivalence() -> None:
    """Patched Qwen2 output must equal base output when Wc=0 (zero-initialized)."""
    from chell.caam.adapters.qwen2 import patch_qwen2

    model = _MockQwen2Model()
    model.eval()

    hidden = torch.randn(1, 5, 32)
    with torch.no_grad():
        base_out = model.model.layers[0].self_attn(hidden)

    patch_qwen2(model, context_dim=16, scale=0.1)
    model.eval()

    ctx = torch.zeros(1, 16)
    for layer in model.model.layers:
        layer.self_attn.set_context(ctx)

    with torch.no_grad():
        patched_out = model.model.layers[0].self_attn(hidden)

    assert torch.allclose(base_out, patched_out, atol=1e-5), (
        "Zero-init Wc must not change Qwen2 attention output"
    )


def test_llama_zero_init_equivalence() -> None:
    """Patched Llama output must equal base output when Wc=0 (zero-initialized)."""
    from chell.caam.adapters.llama import patch_llama

    model = _MockLlamaModel()
    model.eval()

    hidden = torch.randn(1, 5, 32)
    with torch.no_grad():
        base_out = model.model.layers[0].self_attn(hidden)

    patch_llama(model, context_dim=16, scale=0.1)
    model.eval()

    ctx = torch.zeros(1, 16)
    for layer in model.model.layers:
        layer.self_attn.set_context(ctx)

    with torch.no_grad():
        patched_out = model.model.layers[0].self_attn(hidden)

    assert torch.allclose(base_out, patched_out, atol=1e-5), (
        "Zero-init Wc must not change Llama attention output"
    )


def test_gptbigcode_zero_init_equivalence() -> None:
    """Patched GPTBigCode output must equal base output when Wc=0 (zero-initialized)."""
    from chell.caam.adapters.gptbigcode import patch_gptbigcode

    model = _MockGPTBigCodeModel()
    model.eval()

    hidden = torch.randn(1, 5, 32)
    with torch.no_grad():
        base_out = model.transformer.h[0].attn(hidden)

    patch_gptbigcode(model, context_dim=16, scale=0.1)
    model.eval()

    ctx = torch.zeros(1, 16)
    for layer in model.transformer.h:
        layer.attn.set_context(ctx)

    with torch.no_grad():
        patched_out = model.transformer.h[0].attn(hidden)

    assert torch.allclose(base_out, patched_out, atol=1e-5), (
        "Zero-init Wc must not change GPTBigCode attention output"
    )


# ---------------------------------------------------------------------------
# Structural tests — all layers replaced, set_context callable on each
# ---------------------------------------------------------------------------

def test_phi2_all_layers_replaced() -> None:
    """All layers in a Phi-2 model should be replaced after patching."""
    from chell.caam.adapters.phi2 import _Phi2CAAMAttn, patch_phi2

    model = _MockPhi2Model()
    patch_phi2(model, context_dim=16, scale=0.1)

    for layer in model.model.layers:
        assert isinstance(layer.self_attn, _Phi2CAAMAttn)
        assert hasattr(layer.self_attn, "set_context")


def test_qwen2_all_layers_replaced() -> None:
    """All layers in a Qwen2 model should be replaced after patching."""
    from chell.caam.adapters.qwen2 import _Qwen2CAAMAttn, patch_qwen2

    model = _MockQwen2Model()
    patch_qwen2(model, context_dim=16, scale=0.1)

    for layer in model.model.layers:
        assert isinstance(layer.self_attn, _Qwen2CAAMAttn)
        assert hasattr(layer.self_attn, "set_context")


def test_llama_all_layers_replaced() -> None:
    """All layers in a Llama model should be replaced after patching."""
    from chell.caam.adapters.llama import _LlamaCAAMAttn, patch_llama

    model = _MockLlamaModel()
    patch_llama(model, context_dim=16, scale=0.1)

    for layer in model.model.layers:
        assert isinstance(layer.self_attn, _LlamaCAAMAttn)
        assert hasattr(layer.self_attn, "set_context")


def test_gptbigcode_all_layers_replaced() -> None:
    """All layers in a GPTBigCode model should be replaced after patching."""
    from chell.caam.adapters.gptbigcode import _GPTBigCodeCAAMAttn, patch_gptbigcode

    model = _MockGPTBigCodeModel()
    patch_gptbigcode(model, context_dim=16, scale=0.1)

    for layer in model.transformer.h:
        assert isinstance(layer.attn, _GPTBigCodeCAAMAttn)
        assert hasattr(layer.attn, "set_context")


# ---------------------------------------------------------------------------
# CAAMPatcher dispatch tests
# ---------------------------------------------------------------------------

def test_caam_patcher_dispatches_phi2() -> None:
    """CAAMPatcher.patch should dispatch to patch_phi2 and return the same model object."""
    from chell.caam.patcher import CAAMPatcher

    patcher = CAAMPatcher(context_dim=16, scale=0.1)
    model = _MockPhi2Model()
    result = patcher.patch(model, arch="phi2")

    assert result is model


def test_caam_patcher_dispatches_qwen2() -> None:
    """CAAMPatcher.patch should dispatch to patch_qwen2 and return the same model object."""
    from chell.caam.patcher import CAAMPatcher

    patcher = CAAMPatcher(context_dim=16, scale=0.1)
    model = _MockQwen2Model()
    result = patcher.patch(model, arch="qwen2")

    assert result is model


def test_caam_patcher_dispatches_llama() -> None:
    """CAAMPatcher.patch should dispatch to patch_llama and return the same model object."""
    from chell.caam.patcher import CAAMPatcher

    patcher = CAAMPatcher(context_dim=16, scale=0.1)
    model = _MockLlamaModel()
    result = patcher.patch(model, arch="llama")

    assert result is model


def test_caam_patcher_dispatches_gptbigcode() -> None:
    """CAAMPatcher.patch should dispatch to patch_gptbigcode and return the same model object."""
    from chell.caam.patcher import CAAMPatcher

    patcher = CAAMPatcher(context_dim=16, scale=0.1)
    model = _MockGPTBigCodeModel()
    result = patcher.patch(model, arch="gptbigcode")

    assert result is model


def test_caam_patcher_unsupported_arch() -> None:
    """CAAMPatcher.patch should raise NotImplementedError for unknown architectures."""
    from chell.caam.patcher import CAAMPatcher

    patcher = CAAMPatcher()
    with pytest.raises(NotImplementedError):
        patcher.patch(object(), arch="unsupported_arch")


# ---------------------------------------------------------------------------
# Context injection tests — non-zero context changes output
# ---------------------------------------------------------------------------

def test_phi2_set_context_updates_stored_tensor() -> None:
    """set_context should store the provided tensor so it is used during forward."""
    from chell.caam.adapters.phi2 import patch_phi2

    model = _MockPhi2Model()
    patch_phi2(model, context_dim=16, scale=0.1)

    ctx = torch.ones(1, 16)
    layer = model.model.layers[0]
    assert layer.self_attn._context is None

    layer.self_attn.set_context(ctx)
    assert layer.self_attn._context is ctx


def test_set_context_none_uses_zeros() -> None:
    """When no context is set (None), the wrapper should use zero context (no-op)."""
    from chell.caam.adapters.phi2 import patch_phi2

    model = _MockPhi2Model()
    model.eval()

    hidden = torch.randn(1, 5, 32)
    with torch.no_grad():
        base_out = model.model.layers[0].self_attn(hidden)

    patch_phi2(model, context_dim=16, scale=0.1)
    model.eval()

    # Do NOT call set_context — should fall back to zeros internally
    with torch.no_grad():
        patched_out = model.model.layers[0].self_attn(hidden)

    assert torch.allclose(base_out, patched_out, atol=1e-5), (
        "With no context set (None fallback to zeros), output must equal base"
    )
