"""
MLP-level ablation hook. Added after attention-head ablation showed no
causal advantage over a random-head control (see run_ablation.py vs
run_ablation_control.py results). Motivated by the causal-tracing finding
that Gemma's fact-storage peak sits INSIDE the same layer block as the
override signal -- suggesting the causal driver at these layers might be
the MLP sublayer, not attention.

Unlike attention, MLPs have no natural "head" unit -- it's one dense
transformation per layer. So this ablates the WHOLE MLP output at each
target layer (replacing what it contributes to the residual stream),
not a sliced sub-component.

Hook point: a forward HOOK (not pre-hook) on `layer.mlp` directly --
we want to replace the MLP's OUTPUT (its contribution to the residual
stream), not its input.
"""

import torch


class MLPAblationHook:
    def __init__(self, replacement_value):
        self.replacement_value = replacement_value  # tensor, shape (hidden_size,)
        self.armed = False

    def hook(self, module, input, output):
        if not self.armed:
            return output
        patched = output.clone()
        patched[:, -1, :] = self.replacement_value.to(patched.dtype)
        return patched

    def arm(self):
        self.armed = True

    def disarm(self):
        self.armed = False

    def register(self, layer):
        return layer.mlp.register_forward_hook(self.hook)
