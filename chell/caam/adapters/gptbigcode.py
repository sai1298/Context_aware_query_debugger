from __future__ import annotations

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from chell.caam.attention import ContextAwareAttention

    class _GPTBigCodeCAAMAttn(nn.Module):
        """Wrapper that replaces a GPTBigCode attention module with CAAM.

        GPTBigCode uses a combined ``c_attn`` projection (out_features = 3 * hidden)
        instead of separate q/k/v projections.  We split the combined output along
        the last dimension to recover q, k, v.
        """

        def __init__(
            self,
            original_attn: nn.Module,
            context_dim: int,
            scale: float,
        ) -> None:
            super().__init__()
            self.c_attn = original_attn.c_attn  # combined QKV projection
            self.out_proj = original_attn.c_proj
            self.num_heads: int = original_attn.num_heads
            # GPTBigCode uses ``head_size``, not ``head_dim``
            self.head_dim: int = original_attn.head_size
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
            hidden_size = self.num_heads * self.head_dim

            # Combined QKV projection: [B, T, 3 * hidden_size]
            qkv = self.c_attn(hidden_states)
            q, k, v = qkv.split(hidden_size, dim=-1)

            # Reshape to [B, num_heads, T, head_dim]
            q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

            ctx = self._context
            if ctx is None:
                ctx = torch.zeros(1, self.caam.Wc.weight.shape[1], device=hidden_states.device)

            out = self.caam(q, k, v, ctx)  # [B, num_heads, T, head_dim]
            out = out.transpose(1, 2).contiguous().view(B, T, -1)
            return self.out_proj(out)


def patch_gptbigcode(model: object, context_dim: int, scale: float) -> object:
    """Patch a GPTBigCode model with ContextAwareAttention.

    Iterates over ``model.transformer.h`` and replaces each ``layer.attn``
    with a ``_GPTBigCodeCAAMAttn`` wrapper that adds a context bias to QK^T logits.

    Args:
        model: A ``transformers.AutoModelForCausalLM`` GPTBigCode instance.
        context_dim: Dimension of the context tensor from ContextEncoder.
        scale: Scaling factor for the context bias C.

    Returns:
        The mutated model (same object, modified in-place).
    """
    if not HAS_TORCH:
        raise ImportError("patch_gptbigcode requires PyTorch. Install it with: pip install torch")

    for layer in model.transformer.h:  # type: ignore[union-attr]
        original_attn = layer.attn
        layer.attn = _GPTBigCodeCAAMAttn(original_attn, context_dim=context_dim, scale=scale)

    return model
