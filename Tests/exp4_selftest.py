"""
Self-test for the Experiment 4 feature code (exp4_collect.py: ROI + corrected DLA). About 20 seconds.

Builds tiny random-weight copies (2 layers) of Gemma-2 and Llama-3.1 from their real configs, with
their real tokenizers and eager attention, in the precision the real run uses. The RMSNorm weights are
randomised so that Gemma's (1 + weight) scaling and its post-attention norm are actually exercised.
Then it runs the same functions as the real run and checks:
  * every layer hands its attention weights to the hook (needed for ROI); ROI is finite, in [0, 2]
  * the per-head pieces add up to the o_proj output (head slicing is oriented correctly)
  * Gemma-2: the linearised post-attention norm equals the module's own output
  * the fast DLA formula equals the explicit per-head route
  * the final-norm identity reproduces the model's own logit difference
  * the DLA read position: a number answer ("1957") is read where the digits part, not at the space

    python Tests/exp4_selftest.py                  # on the GPU box; exit code 0 = OK
    python Tests/exp4_selftest.py G L --cpu-ok     # local test with two model folders
"""

import sys
import time
import traceback
from pathlib import Path

t0 = time.time()
import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from exp4_collect import (Probe, cb_prompt, dla_matrix, dla_position, forward, question_span,  # noqa: E402
                          snapshot, span_rows, wp_prompt)
from gpu_utils import MODEL_PATHS, pick_dtype  # noqa: E402

print(f"torch {torch.__version__} | transformers {transformers.__version__}", flush=True)
if not torch.cuda.is_available() and "--cpu-ok" not in sys.argv:
    sys.exit("FAIL: PyTorch cannot see the GPU")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
PATHS = dict(zip(["gemma", "llama"], args)) if len(args) == 2 else {k: MODEL_PATHS[k] for k in ("gemma", "llama")}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
Q = "when was the first manmade object sent into space"
PASSAGES = ["The first manmade object to reach space was a V-2 rocket on 3 October 1942.",
            "Sputnik 1 was launched by the Soviet Union in 1957."]


def tiny(path):
    cfg = AutoConfig.from_pretrained(path)
    d = cfg.to_dict()
    d.update(num_hidden_layers=2, hidden_size=64, intermediate_size=128, num_attention_heads=4,
             num_key_value_heads=2, head_dim=16)
    d.pop("layer_types", None)
    cfg = type(cfg).from_dict(d)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg, attn_implementation="eager")
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name and p.dim() == 1:
                p.copy_(torch.randn_like(p) * 0.5 + (0.0 if "gemma" in cfg.model_type else 1.0))
    return model.to(device=DEVICE, dtype=pick_dtype(path, False)).eval()


def tokenizer(path):
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


failed = []
for name in ("gemma", "llama"):
    path = PATHS[name]
    try:
        tok, model = tokenizer(path), tiny(path)
        L = len(model.model.layers)
        probe = Probe(model)
        problems = []
        # ROI
        s0, e0, ids0 = question_span(tok, cb_prompt(Q), Q)
        forward(model, probe, ids0.to(model.device))
        if len(probe.attn) != L:
            problems.append(f"attention weights reached the hook for {len(probe.attn)} of {L} layers")
        base = span_rows(probe.attn, s0, e0)
        s1, e1, ids1 = question_span(tok, wp_prompt(PASSAGES, Q), Q)
        ids1 = ids1.to(model.device)
        forward(model, probe, ids1)
        first = snapshot(probe)
        ctx = span_rows(probe.attn, s1, e1)
        if (e1 - s1) != (e0 - s0) or e0 - s0 < 3:
            problems.append(f"question spans differ or too short: {e0 - s0} vs {e1 - s1} tokens")
        else:
            roi = torch.stack([1 - torch.nn.functional.cosine_similarity(ctx[l], base[l], dim=-1) for l in range(L)])
            if not torch.isfinite(roi).all() or roi.min() < -1e-3 or roi.max() > 2 + 1e-3:
                problems.append(f"ROI out of range: {roi.min():.3f}..{roi.max():.3f}")
        # DLA identities
        top2 = torch.topk(first["logits"], 2).indices.tolist()
        M, chk = dla_matrix(model, first, top2[1], top2[0], check=True)
        if M.shape != (L, model.config.num_attention_heads) or not torch.isfinite(M).all():
            problems.append(f"DLA matrix shape/values wrong: {tuple(M.shape)}")
        lim = 0.05 if model.dtype != torch.float32 else 1e-3
        for k, v in chk.items():
            if k.endswith("heads_sum_vs_o_proj") or k.endswith("post_norm_linear_vs_module"):
                if not v <= lim:
                    problems.append(f"{k} = {v:.4g} (limit {lim})")
            if k.endswith("fast_vs_explicit") and not v <= 1e-3:
                problems.append(f"{k} = {v:.4g}")
        scale = chk["logit_scale"]
        if not chk["final_norm_identity_abs_err"] <= 0.03 * scale + 0.05:
            problems.append(f"final-norm identity error {chk['final_norm_identity_abs_err']:.4g} (logit scale {scale:.3g})")
        # read position for a number: the space (Llama-3) or space + shared digits (Gemma) are skipped
        gold = tok(" 1942", add_special_tokens=False)["input_ids"]
        pos = dla_position(model, probe, tok, ids1, first, "1942")
        if pos is None:
            problems.append("no DLA read position for a number answer")
        else:
            st, tgt, alt, k_fed, mode = pos
            if tgt != gold[k_fed] or (mode == "diverge" and alt == tgt):
                problems.append(f"read position inconsistent: mode {mode}, fed {k_fed}")
            M2, _ = dla_matrix(model, st, tgt, alt)
            if not torch.isfinite(M2).all():
                problems.append("DLA at the read position not finite")
            print(f"      number answer ' 1942' -> tokens {[tok.decode([t]) for t in gold]}; random tiny model "
                  f"read at token {k_fed} ({mode})", flush=True)
        probe.remove()
        print(f"  {'OK  ' if not problems else 'FAIL'} {name:5s} ROI + DLA ({model.dtype}, {DEVICE}); checks "
              + ", ".join(f"{k}={v:.2g}" for k, v in chk.items() if k not in ("logit_diff", "logit_scale")), flush=True)
        for p in problems:
            print(f"      {p}")
        if problems:
            failed.append(name)
        del model
    except Exception:
        traceback.print_exc()
        print(f"  FAIL {name}", flush=True)
        failed.append(name)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if failed:
    sys.exit(f"EXP4 SELF-TEST FAILED for: {', '.join(failed)}")
print(f"EXP4 SELF-TEST PASSED ({time.time() - t0:.0f}s)")
