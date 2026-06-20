from __future__ import annotations

import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:

    class ContextAwareAttention(nn.Module):
        """Drop-in self-attention module that adds a context bias C to QKᵀ logits.

        The bias is computed as:
            C = scale · (Q @ Wc(context_tensor)ᵀ)

        Wc is zero-initialized so the patched model is identical to the base
        model at the start of fine-tuning (training stability guarantee).
        """

        def __init__(
            self,
            num_heads: int,
            head_dim: int,
            context_dim: int,
            scale: float = 0.1,
        ) -> None:
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = head_dim
            self.scale = scale

            # Projects context vector into head space; zero-initialized.
            self.Wc = nn.Linear(context_dim, head_dim, bias=False)
            nn.init.zeros_(self.Wc.weight)

        def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            context_tensor: torch.Tensor,
            attn_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """Compute context-aware attention output.

            Args:
                q: Query tensor of shape [..., seq_len, head_dim].
                k: Key tensor of shape [..., seq_len, head_dim].
                v: Value tensor of shape [..., seq_len, head_dim].
                context_tensor: Context tensor of shape [..., context_dim] or
                    [..., 1, context_dim].  Broadcast-compatible with q.
                attn_mask: Optional additive mask of shape broadcastable to
                    [..., seq_len, seq_len].  Added to raw logits before softmax.

            Returns:
                Output tensor of shape [..., seq_len, head_dim].
            """
            # Ensure context_tensor has a sequence dimension for matmul broadcast.
            # Expected: [..., 1, head_dim] after projection.
            ctx_proj = self.Wc(context_tensor)  # [..., (1,) head_dim]
            if ctx_proj.dim() == q.dim() - 1:
                ctx_proj = ctx_proj.unsqueeze(-2)  # add seq dim

            # Standard scaled dot-product scores plus context bias C.
            # raw_scores shape: [..., seq_len, seq_len]
            raw_scores = (
                q @ k.transpose(-2, -1)
                + self.scale * (q @ ctx_proj.transpose(-2, -1))
            ) / math.sqrt(self.head_dim)

            if attn_mask is not None:
                raw_scores = raw_scores + attn_mask

            weights = F.softmax(raw_scores, dim=-1)
            return weights @ v

else:

    class ContextAwareAttention:  # type: ignore[no-redef]
        """Stub used when PyTorch is not installed."""

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "ContextAwareAttention requires PyTorch. "
                "Install it with: pip install torch"
            )
