"""
Phase 4, T21: GPT-2 ablation practice.

Validates the AblationHook mechanism (mean-ablation via a pre-hook on
the attention output projection) on GPT-2 before spending rented GPU
time on Gemma/Llama. Same discipline as every GPU-bound step so far
this project.

GPT-2 architecture note: HF's GPT2Attention module has a combined QKV
projection and a single output projection called `c_proj` (not `o_proj`
-- naming differs from Llama/Gemma, but the role is identical: it's the
dense matrix that mixes concatenated per-head outputs back down to
hidden_size). We hook GPT-2's `c_proj` the same way we'll hook Gemma/
Llama's `o_proj` -- as a forward PRE-hook, so we intercept the
concatenated, not-yet-mixed per-head tensor.

What this script actually checks:
  1. Does arming the hook change the model's output at all? (sanity --
     if ablating a head changes NOTHING, the hook isn't actually
     wired in correctly)
  2. Does ablating a DIFFERENT head than the one "responsible" for a
     behavior leave that behavior roughly unchanged? (specificity --
     if ablating ANY head changes everything the same way, the hook is
     probably mis-slicing and hitting the whole tensor, not one head)
  3. Does disarming the hook exactly restore original behavior? (the
     hook must be side-effect-free when not armed)

Run from repo root:
    python Tests/Phase_04/practice_ablation_gpt2.py
"""

import sys
import os
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ablation_utils import AblationHook

MODEL_NAME = "gpt2"


def get_next_token_probs(model, input_ids):
    with torch.no_grad():
        out = model(input_ids)
    return torch.softmax(out.logits[0, -1, :], dim=-1)


def main():
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, attn_implementation="eager")
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    model.eval()

    num_heads = model.config.n_head
    head_dim = model.config.n_embd // num_heads
    print(f"GPT-2: num_heads={num_heads}, head_dim={head_dim}, n_embd={model.config.n_embd}")

    prompt = "The capital of France is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # --- baseline: no hook at all ---
    baseline_probs = get_next_token_probs(model, input_ids)
    top5_baseline = torch.topk(baseline_probs, 5)
    print(f"\nPrompt: {prompt!r}")
    print("Baseline top-5 next tokens:")
    for p, idx in zip(top5_baseline.values, top5_baseline.indices):
        print(f"  {tokenizer.decode([idx]):>12s}  p={p.item():.4f}")

    # pick a target layer/head to ablate -- middle of the network, arbitrary for this practice run
    target_layer_idx = model.config.n_layer // 2
    target_head_idx = 0

    # replacement value: for this practice run, just use a zero vector's
    # SHAPE with the SAME mean-magnitude as a real activation, so the
    # test isn't trivially "zero-ablation in disguise" -- pull a real
    # activation from an actual forward pass on a DIFFERENT prompt to
    # act as a stand-in "mean faithful activation"
    donor_prompt = "The weather today is"
    donor_ids = tokenizer(donor_prompt, return_tensors="pt").input_ids

    captured = {}

    def capture_hook(module, args):
        captured["value"] = args[0][0, -1, target_head_idx * head_dim:(target_head_idx + 1) * head_dim].clone()
        return None

    target_layer = model.transformer.h[target_layer_idx]
    capture_handle = target_layer.attn.c_proj.register_forward_pre_hook(capture_hook)
    with torch.no_grad():
        model(donor_ids)
    capture_handle.remove()
    replacement_value = captured["value"]
    print(f"\nCaptured replacement value from donor prompt {donor_prompt!r}, "
          f"layer {target_layer_idx} head {target_head_idx}, shape={tuple(replacement_value.shape)}")

    # --- set up the real ablation hook ---
    # NOTE: AblationHook.register() assumes Llama/Gemma naming
    # (layer.self_attn.o_proj) -- GPT-2 uses layer.attn.c_proj instead,
    # so for this practice script we register directly on c_proj rather
    # than calling .register(), which would raise AttributeError on a
    # GPT2Block (no .self_attn attribute exists on it).
    ablation_hook = AblationHook(target_head_idx, head_dim, replacement_value)
    handle = target_layer.attn.c_proj.register_forward_pre_hook(ablation_hook.pre_hook)

    # --- Check 1: armed vs baseline should differ ---
    ablation_hook.arm()
    ablated_probs = get_next_token_probs(model, input_ids)
    ablation_hook.disarm()

    diff = (baseline_probs - ablated_probs).abs().sum().item()
    print(f"\n[Check 1] Total absolute prob difference (ablated vs baseline): {diff:.6f}")
    if diff < 1e-6:
        print("  FAIL: ablating this head changed NOTHING. Hook is not wired in correctly.")
    else:
        print("  PASS: ablation measurably changed the output distribution.")

    top5_ablated = torch.topk(ablated_probs, 5)
    print("  Ablated top-5 next tokens:")
    for p, idx in zip(top5_ablated.values, top5_ablated.indices):
        print(f"    {tokenizer.decode([idx]):>12s}  p={p.item():.4f}")

    # --- Check 2: disarming should exactly restore baseline ---
    restored_probs = get_next_token_probs(model, input_ids)
    restore_diff = (baseline_probs - restored_probs).abs().sum().item()
    print(f"\n[Check 2] Total absolute prob difference (disarmed vs baseline): {restore_diff:.8f}")
    if restore_diff > 1e-6:
        print("  FAIL: disarming the hook did not exactly restore original behavior -- side effect present.")
    else:
        print("  PASS: disarmed hook is fully transparent, as expected.")

    # --- Check 3: ablating a DIFFERENT head should produce a DIFFERENT (usually smaller or unrelated) effect ---
    other_head_idx = (target_head_idx + 1) % num_heads
    other_captured = {}

    def other_capture_hook(module, args):
        other_captured["value"] = args[0][0, -1, other_head_idx * head_dim:(other_head_idx + 1) * head_dim].clone()
        return None

    other_capture_handle = target_layer.attn.c_proj.register_forward_pre_hook(other_capture_hook)
    with torch.no_grad():
        model(donor_ids)
    other_capture_handle.remove()

    other_hook = AblationHook(other_head_idx, head_dim, other_captured["value"])
    other_handle = target_layer.attn.c_proj.register_forward_pre_hook(other_hook.pre_hook)
    other_hook.arm()
    other_ablated_probs = get_next_token_probs(model, input_ids)
    other_hook.disarm()
    other_handle.remove()

    cross_diff = (ablated_probs - other_ablated_probs).abs().sum().item()
    print(f"\n[Check 3] Difference between ablating head {target_head_idx} vs head {other_head_idx}: {cross_diff:.6f}")
    if cross_diff < 1e-6:
        print("  SUSPICIOUS: ablating two different heads produced IDENTICAL effects.")
        print("  This suggests the hook may be mis-slicing and affecting the whole tensor,")
        print("  not just the intended head. Investigate before trusting T22.")
    else:
        print("  PASS: different heads produce different effects, as expected for correct per-head slicing.")

    handle.remove()
    print("\nDone. If all three checks PASS, the mechanism is validated -- port directly to")
    print("Gemma/Llama in T22 using AblationHook.register() (which uses self_attn.o_proj).")


if __name__ == "__main__":
    main()