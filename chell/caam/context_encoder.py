from __future__ import annotations

from typing import Callable

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from chell.core.types import DebugSession


if HAS_TORCH:

    class ContextEncoder(nn.Module):
        """Encodes a DebugSession into a fixed-size context tensor.

        The session is condensed to a single string (clarification summary +
        retrieved case IDs), embedded via an external text embedding function,
        then projected to *output_dim* through a single linear layer.

        Args:
            text_embed_fn: Callable that maps a string to a list of floats
                (embedding vector).  The embedding dimension is inferred from
                the first call and must be consistent across calls.
            hidden_dim: Not used in the current single-layer projection but
                reserved for future MLP expansion.
            output_dim: Dimensionality of the returned context tensor.
        """

        def __init__(
            self,
            text_embed_fn: Callable[[str], list[float]],
            hidden_dim: int = 256,
            output_dim: int = 256,
        ) -> None:
            super().__init__()
            self.text_embed_fn = text_embed_fn
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim

            # Projection layer is built lazily on the first encode() call once
            # we know the embedding dimension from text_embed_fn.
            self._proj: nn.Linear | None = None

        def _ensure_proj(self, embed_dim: int) -> None:
            if self._proj is None:
                self._proj = nn.Linear(embed_dim, self.output_dim, bias=True)
                nn.init.xavier_uniform_(self._proj.weight)
                nn.init.zeros_(self._proj.bias)

        def encode(self, session: DebugSession) -> torch.Tensor:
            """Encode a DebugSession into a context tensor of shape [1, output_dim].

            Args:
                session: The active debug session to pool context from.

            Returns:
                A float32 tensor of shape [1, output_dim].
            """
            text = session.clarification_summary() + "\n" + str(session.retrieved_cases)
            raw_embed: list[float] = self.text_embed_fn(text)

            embed_tensor = torch.tensor(raw_embed, dtype=torch.float32).unsqueeze(0)
            self._ensure_proj(embed_tensor.shape[-1])

            assert self._proj is not None  # narrowing for type checker
            return self._proj(embed_tensor)  # shape [1, output_dim]

else:

    class ContextEncoder:  # type: ignore[no-redef]
        """Encodes a DebugSession into a context representation.

        Falls back to returning a plain list[float] when PyTorch is unavailable.
        """

        def __init__(
            self,
            text_embed_fn: Callable[[str], list[float]],
            hidden_dim: int = 256,
            output_dim: int = 256,
        ) -> None:
            self.text_embed_fn = text_embed_fn
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim

        def encode(self, session: DebugSession) -> list[float]:
            """Encode session to a raw embedding list (no projection without torch).

            Returns:
                list[float] — the raw embedding from text_embed_fn.
            """
            text = session.clarification_summary() + "\n" + str(session.retrieved_cases)
            return self.text_embed_fn(text)
