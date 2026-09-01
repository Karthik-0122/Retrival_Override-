"""
02_capture_attention_weights.py

Step up from script 1: instead of just printing, STORE what the hook
sees, across ALL layers, in a dict you can inspect afterward. This is
the pattern your actual T5 hook will use -- capture something during
the forward pass, keep it around for analysis after generation finishes.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model.eval()

# This dict is the "capture point" -- hooks write into it, and we read
# from it after the forward pass is done. Using a dict keyed by layer
# name (not a single global variable) is what lets you capture from
# MULTIPLE layers at once, which you'll need for real analysis.
captured_attention = {}

def make_attention_hook(layer_name):
    """
    Returns a hook function that knows which layer it belongs to.
    This "factory" pattern is necessary because register_forward_hook
    only passes (module, input, output) -- it has no way to tell the
    hook which layer it's on unless you bake that in via closure.
    """
    def hook(module, input, output):
        # For GPT-2's attention module with eager attention, output is a
        # tuple; the attention weights are wherever the model puts them
        # (varies by model -- always check with a print(type(output)) /
        # print(len(output)) first when you're on a new architecture).
        # Here we store whatever we got, detached from the compute graph
        # (.detach()) so we're not holding onto gradient history we don't need.
        captured_attention[layer_name] = output
    return hook

# Register the SAME hook pattern on every transformer layer's attention
# module -- this is what "extract attention weights from each head at
# layer 20" (your T2 plan) actually looks like in code, just at every
# layer instead of one.
handles = []
for i, block in enumerate(model.transformer.h):
    h = block.attn.register_forward_hook(make_attention_hook(f"layer_{i}"))
    handles.append(h)

inputs = tokenizer("The capital of France is", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)

# Clean up -- remove every hook we registered. Losing track of handles
# is the single easiest way to leave stale hooks attached.
for h in handles:
    h.remove()

print(f"Captured data from {len(captured_attention)} layers")
print(f"Layers captured: {list(captured_attention.keys())}")

# Simpler alternative for THIS specific case: HF models can just return
# attentions directly via output_attentions=True, no hook needed.
# outputs.attentions is a tuple of (batch, heads, seq, seq) tensors, one per layer.
print(f"\nFor comparison, outputs.attentions has {len(outputs.attentions)} layers")
print(f"Shape of layer 0 attention: {outputs.attentions[0].shape}")
print("(batch, num_heads, query_positions, key_positions)")
print("\nWhy use hooks at all then? Because output_attentions only gives you")
print("attention PATTERNS. Hooks let you grab ANYTHING -- residual stream")
print("values, MLP activations, gradients -- that the model doesn't expose")
print("as a normal output argument.")