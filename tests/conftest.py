from __future__ import annotations

import os

# faiss and torch both bundle their own libomp.dylib, causing a dual-OMP crash
# when both are loaded in the same process. Workaround:
#   1. Import faiss before torch so faiss owns the first OMP initialisation.
#   2. Set torch to single-threaded so it never tries to init a parallel OMP pool.
# Both steps are needed; KMP_DUPLICATE_LIB_OK alone does not prevent the segfault.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import faiss  # noqa: F401, E402  — must precede any torch import

try:
    import torch
    torch.set_num_threads(1)
except ImportError:
    pass
