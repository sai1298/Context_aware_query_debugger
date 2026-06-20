from __future__ import annotations

from chell.caam.adapters.gptbigcode import patch_gptbigcode
from chell.caam.adapters.llama import patch_llama
from chell.caam.adapters.phi2 import patch_phi2
from chell.caam.adapters.qwen2 import patch_qwen2

__all__ = [
    "patch_phi2",
    "patch_qwen2",
    "patch_gptbigcode",
    "patch_llama",
]
