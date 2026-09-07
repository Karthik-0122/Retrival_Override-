"""
Phase 4, T18: Ablation method.

DECISION: mean-ablation, computed from faithful-case activations,
chosen as the primary method over zero-ablation and resampling-ablation.
(Rationale below, unchanged from the original version of this file.)

CRITICAL CORRECTION (found via check_head_geometry.py, T19 prep):
Gemma 2 9B has num_attention_heads=16, head_dim=256 -> 16*256=4096,
but hidden_size=3584. These do NOT match. This means the attention
module's OUTPUT PROJECTION (o_proj) maps a wider 4096-dim concatenated
multi-head tensor down to the 3584-dim hidden_size via a dense,
fully-mixing matrix multiply.

This means: a forward hook on the WHOLE self_attn module (its final
output, post-o_proj) sees a tensor where individual heads' contributions
have ALREADY BEEN LINEARLY MIXED TOGETHER. You cannot cleanly slice that
output into per-head chunks -- there is no clean 256-column boundary
that belongs only to "head 7" anymore once o_proj has run.

FIX: hook the INPUT to o_proj instead (a forward PRE-hook on
`layer.self_attn.o_proj`), not the output of self_attn as a whole. The
input to o_proj is the concatenated per-head attention output, shape
(batch, seq_len, num_heads * head_dim) -- e.g. 4096 for both Gemma
(16*256) and Llama (32*128) -- where head boundaries ARE still clean,
un-mixed slices. Ablate there, then let o_proj run normally afterward.

This module is imported by both the GPT-2 practice script (T21) and the
real ablation script (T22) so the ablation logic is identical in both
places.
"""

import torch


class AblationHook:
    """
    Registers as a forward PRE-hook on `layer.self_attn.o_proj` --
    NOT a forward hook on `layer.self_attn` itself. See the module
    docstring above for why: hooking self_attn's final output would see
    heads already mixed together by o_proj's dense projection matrix.

    A pre-hook on o_proj intercepts o_proj's INPUT, which is the
    concatenated (not-yet-mixed) per-head tensor -- shape
    (batch, seq_len, num_heads * head_dim). Head i occupies columns
    [i*head_dim : (i+1)*head_dim] in that tensor, cleanly.

    When armed, replaces the target head's slice with a fixed
    replacement value (the mean faithful-case activation for that head,
    computed ahead of time -- see compute_faithful_means.py, T22a).
    When not armed, passes input through unchanged -- lets you reuse the
    same hook for both "normal generation" and "ablated generation" on
    the same query, just by toggling .armed.
    """

    def __init__(self, head_idx, head_dim, replacement_value):
        self.head_idx = head_idx
        self.head_dim = head_dim
        self.replacement_value = replacement_value  # tensor, shape (head_dim,)
        self.armed = False

    def pre_hook(self, module, args):
        """Forward PRE-hook signature: (module, args) -> modified args or None.
        args[0] is o_proj's input tensor, shape (batch, seq_len, num_heads*head_dim)."""
        if not self.armed:
            return None  # None means "don't modify args"

        hidden_states = args[0]
        h_start = self.head_idx * self.head_dim
        h_end = h_start + self.head_dim

        patched = hidden_states.clone()
        # apply to the LAST token position only (the one about to be
        # generated) -- during .generate() with KV caching, seq_len is 1
        # after the prefill step; this indexing is safe for both the
        # prefill step (seq_len > 1) and subsequent decode steps (seq_len == 1)
        patched[0, -1, h_start:h_end] = self.replacement_value.to(patched.dtype)

        return (patched,) + args[1:]

    def arm(self):
        self.armed = True

    def disarm(self):
        self.armed = False

    def register(self, layer):
        """Registers this hook on the given transformer layer's o_proj as
        a forward PRE-hook. Returns the handle so it can be removed later."""
        return layer.self_attn.o_proj.register_forward_pre_hook(self.pre_hook)
