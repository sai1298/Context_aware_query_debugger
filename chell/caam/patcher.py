from __future__ import annotations

try:
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from chell.caam.adapters import (
    patch_gptbigcode,
    patch_llama,
    patch_phi2,
    patch_qwen2,
)

_ARCH_DISPATCH: dict[str, object] = {
    "phi2": patch_phi2,
    "qwen2": patch_qwen2,
    "gptbigcode": patch_gptbigcode,
    "llama": patch_llama,
}


class CAAMPatcher:
    """Patches a HuggingFace model in-place to use ContextAwareAttention.

    Each architecture requires a dedicated adapter that knows how to locate
    and swap out the attention forward pass.  Unsupported architectures raise
    NotImplementedError immediately so callers fail loudly rather than silently
    applying no patch.

    Args:
        context_dim: Dimension of the context tensor produced by ContextEncoder.
        scale: Scaling factor for the context bias C (see ContextAwareAttention).
    """

    def __init__(self, context_dim: int = 256, scale: float = 0.1) -> None:
        self.context_dim = context_dim
        self.scale = scale

    def patch(self, model: "nn.Module", arch: str) -> "nn.Module":
        """Swap attention layers of *model* to ContextAwareAttention.

        Args:
            model: A HuggingFace transformers model (nn.Module).
            arch: Architecture identifier — one of
                ``{"phi2", "qwen2", "gptbigcode", "llama"}``.

        Returns:
            The patched model (same object, mutated in-place, returned for
            convenience in chained calls).

        Raises:
            NotImplementedError: If *arch* is not in the supported set.
        """
        adapter_fn = _ARCH_DISPATCH.get(arch)
        if adapter_fn is None:
            supported = ", ".join(sorted(_ARCH_DISPATCH))
            raise NotImplementedError(
                f"CAAM patch not implemented for architecture '{arch}'. "
                f"Supported: {supported}. "
                "Add a new adapter under chell/caam/adapters/ to extend support."
            )
        return adapter_fn(model, self.context_dim, self.scale)  # type: ignore[operator]
