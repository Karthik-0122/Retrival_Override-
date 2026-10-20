"""
GPU self-test: can this PyTorch + transformers pair run our exact code?

Builds tiny random-weight copies (2 layers) of Gemma-2, Llama-3.1 and the Qwen
judge from their real configs, with their real tokenizers (small files that
are already in the model cache), puts them on the GPU in the precision the real
run uses, and calls the same gpu_utils functions as the real run:
generate (batched, left-padded, stops at a newline, per-token log-probs) and
score_labels (the judge's yes/no/partial probabilities). About 30 seconds.

    python Tests/gpu_selftest.py                  # on the GPU box; exit code 0 = OK
    python Tests/gpu_selftest.py G L J --cpu-ok   # local test with three model folders
"""

import math
import sys
import time
import traceback
from pathlib import Path

t0 = time.time()
import torch
import transformers
from transformers.utils import is_torch_available

print(f"torch {torch.__version__} (CUDA {torch.version.cuda}) | transformers {transformers.__version__}", flush=True)
if not is_torch_available():
    sys.exit("FAIL: this transformers version refuses this PyTorch version")
if not torch.cuda.is_available() and "--cpu-ok" not in sys.argv:
    sys.exit("FAIL: PyTorch cannot see the GPU")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from gpu_utils import MODEL_PATHS, generate, pick_dtype, score_labels  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
PATHS = dict(zip(["gemma", "llama", "judge"], args)) if len(args) == 3 else MODEL_PATHS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPTS = ["Question: who wrote the play hamlet?\nAnswer:",
           "Paris is the capital and most populous city of France.\n\nQuestion: what is the capital of france?\nAnswer:"]


def tokenizer(path):                                   # same settings as gpu_utils.load_model
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def tiny_model(path):
    cfg = AutoConfig.from_pretrained(path)
    d = cfg.to_dict()
    d.update(num_hidden_layers=2, hidden_size=64, intermediate_size=128, num_attention_heads=4,
             num_key_value_heads=2, head_dim=16)
    d.pop("layer_types", None)                         # rebuilt for 2 layers
    cfg = type(cfg).from_dict(d)
    kw = {"attn_implementation": "eager"} if "gemma" in str(path).lower() else {}
    model = AutoModelForCausalLM.from_config(cfg, **kw)
    return model.to(device=DEVICE, dtype=pick_dtype(path, False)).eval()


def finite(*xs):
    return all(isinstance(x, float) and math.isfinite(x) for x in xs)


failed = []
for name in ("gemma", "llama", "judge"):
    path = PATHS[name]
    try:
        tok, model = tokenizer(path), tiny_model(path)
        if name == "judge":
            msgs = [{"role": "user", "content": "Is 'Paris' a correct answer to: what is the capital of France? "
                                                "Answer yes, no or partial."}]
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            rows = score_labels(model, tok, [p, "Context: none.\n" + p], ["yes", "no", "partial"],
                                token_budget=4000, max_bs=8, desc="  judge scoring")
            ok = len(rows) == 2 and all(finite(*r.values()) and abs(sum(r.values()) - 1) < 1e-3 for r in rows)
            what = "label probabilities"
        else:
            res = generate(model, tok, PROMPTS, max_new_tokens=8, token_budget=4000, max_bs=8,
                           desc=f"  {name} generation")
            ok = len(res) == 2 and all(isinstance(r["text"], str) and finite(r["first_logprob"], r["mean_logprob"])
                                       for r in res)
            what = "batched generation + log-probs"
        print(f"  {'OK  ' if ok else 'FAIL'} {name:5s} {what} ({model.dtype}, {DEVICE})", flush=True)
        if not ok:
            failed.append(name)
        del model
    except Exception:
        traceback.print_exc()
        print(f"  FAIL {name}", flush=True)
        failed.append(name)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if failed:
    sys.exit(f"SELF-TEST FAILED for: {', '.join(failed)}")
print(f"SELF-TEST PASSED ({time.time() - t0:.0f}s)")