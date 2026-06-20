from __future__ import annotations

from typing import TYPE_CHECKING

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if TYPE_CHECKING:
    from chell.data.schema import DebugCase

# ---------------------------------------------------------------------------
# Sequence format tokens
# ---------------------------------------------------------------------------

_T_BUGGY = "[BUGGY_CODE]"
_T_QUERY = "[QUERY]"
_T_RESPONSE = "[RESPONSE]"
_T_FIX = "[FIX]"
_T_END = "[END]"


def _format_sequence(case: "DebugCase") -> str:
    """Render a DebugCase into the flat training-sequence string.

    Format::

        [BUGGY_CODE] {buggy_code} [QUERY] {clarification_query}
        [RESPONSE] {expected_user_response} [FIX] {corrected_code} [END]
    """
    return (
        f"{_T_BUGGY} {case.buggy_code} "
        f"{_T_QUERY} {case.clarification_query} "
        f"{_T_RESPONSE} {case.expected_user_response} "
        f"{_T_FIX} {case.corrected_code} "
        f"{_T_END}"
    )


def _prefix_text(case: "DebugCase") -> str:
    """Return the prefix that precedes [FIX] — used to determine label mask length."""
    return (
        f"{_T_BUGGY} {case.buggy_code} "
        f"{_T_QUERY} {case.clarification_query} "
        f"{_T_RESPONSE} {case.expected_user_response} "
        f"{_T_FIX} "
    )


class ChellCollator:
    """Data collator that converts a batch of :class:`~chell.data.schema.DebugCase`
    objects into tokenized training tensors for causal-language-model fine-tuning.

    The input sequence has the structure::

        [BUGGY_CODE] <buggy_code> [QUERY] <query> [RESPONSE] <response> [FIX] <fix> [END]

    Labels are masked (``-100``) for the prefix up to and including the ``[FIX]``
    token so that the model is only trained to predict the corrected code.

    Args:
        tokenizer: A HuggingFace ``PreTrainedTokenizer`` (or ``Fast`` variant).
        max_seq_len: Maximum sequence length; longer sequences are truncated and
            shorter ones are right-padded.
    """

    def __init__(self, tokenizer, max_seq_len: int = 1024) -> None:  # noqa: ANN001
        if not HAS_TORCH:
            raise ImportError(
                "ChellCollator requires PyTorch. "
                "Install it with: pip install torch"
            )
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # Ensure the tokenizer has a padding token so batches can be padded.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(self, batch: list["DebugCase"]) -> dict:
        """Collate a batch of DebugCase objects into model-ready tensors.

        Args:
            batch: A list of :class:`~chell.data.schema.DebugCase` instances.

        Returns:
            A dict with keys:

            * ``"input_ids"``  — ``LongTensor`` of shape ``[batch, max_seq_len]``
            * ``"attention_mask"`` — ``LongTensor`` of shape ``[batch, max_seq_len]``
            * ``"labels"``     — ``LongTensor`` of shape ``[batch, max_seq_len]``;
              prefix tokens are set to ``-100`` (ignored by cross-entropy loss).
        """
        all_input_ids: list[torch.Tensor] = []
        all_attention_masks: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        for case in batch:
            full_text = _format_sequence(case)
            prefix_text = _prefix_text(case)

            # Tokenize full sequence (no special tokens added beyond what the
            # tokenizer inserts by default for the architecture).
            full_enc = self.tokenizer(
                full_text,
                max_length=self.max_seq_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids: torch.Tensor = full_enc["input_ids"].squeeze(0)  # [seq_len]
            attention_mask: torch.Tensor = full_enc["attention_mask"].squeeze(0)

            # Determine how many prefix tokens to mask in the labels.
            prefix_enc = self.tokenizer(
                prefix_text,
                add_special_tokens=False,
                return_tensors="pt",
            )
            prefix_len: int = prefix_enc["input_ids"].shape[1]

            # Build labels: copy input_ids, then mask the prefix with -100.
            labels = input_ids.clone()
            prefix_len_clamped = min(prefix_len, self.max_seq_len)
            labels[:prefix_len_clamped] = -100
            # Also mask padding tokens so loss is not computed on them.
            labels[attention_mask == 0] = -100

            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            all_labels.append(labels)

        return {
            "input_ids": torch.stack(all_input_ids),          # [B, seq_len]
            "attention_mask": torch.stack(all_attention_masks),
            "labels": torch.stack(all_labels),
        }
