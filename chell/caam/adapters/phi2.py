from __future__ import annotations

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from chell.caam.attention import ContextAwareAttention

    class _Phi2CAAMAttn(nn.Module):
        """Wrapper that replaces a Phi-2 self-attention module with CAAM."""

        def __init__(
            self,
            original_attn: nn.Module,
            context_dim: int,
            scale: float,
        ) -> None:
            super().__init__()
            self.q_proj = original_attn.q_proj
            self.k_proj = original_attn.k_proj
            self.v_proj = original_attn.v_proj
            self.out_proj = original_attn.dense
            self.num_heads: int = original_attn.num_heads
            self.head_dim: int = original_attn.head_dim
            self.caam = ContextAwareAttention(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                context_dim=context_dim,
                scale=scale,
            )
            self._context: torch.Tensor | None = None

        def set_context(self, context_tensor: torch.Tensor) -> None:
            self._context = context_tensor

        def forward(self, hidden_states: torch.Tensor, **kwargs: object) -> torch.Tensor:
            B, T, C = hidden_states.shape
            q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

            ctx = self._context
            if ctx is None:
                ctx = torch.zeros(1, self.caam.Wc.weight.shape[1], device=hidden_states.device)

            out = self.caam(q, k, v, ctx)  # [B, num_heads, T, head_dim]
            out = out.transpose(1, 2).contiguous().view(B, T, -1)
            return self.out_proj(out)


def patch_phi2(model: object, context_dim: int, scale: float) -> object:
    """Patch a Phi-2 model with ContextAwareAttention.

    Iterates over ``model.model.layers`` and replaces each ``layer.self_attn``
    with a ``_Phi2CAAMAttn`` wrapper that adds a context bias to QK^T logits.

    Args:
        model: A ``transformers.AutoModelForCausalLM`` Phi-2 instance.
        context_dim: Dimension of the context tensor from ContextEncoder.
        scale: Scaling factor for the context bias C.

    Returns:
        The mutated model (same object, modified in-place).
    """
    if not HAS_TORCH:
        raise ImportError("patch_phi2 requires PyTorch. Install it with: pip install torch")

    for layer in model.model.layers:  # type: ignore[union-attr]
        original_attn = layer.self_attn
        layer.self_attn = _Phi2CAAMAttn(original_attn, context_dim=context_dim, scale=scale)

    return model
