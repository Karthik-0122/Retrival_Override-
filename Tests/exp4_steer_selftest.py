"""
Self-test for the steering code (exp4_steer.py). About 30 seconds. Builds tiny random-weight copies of
Gemma-2 / Llama-3.1 from their real configs and tokenizers (float32, so differences are logic, not
rounding) and checks, with the transformers version installed on this box:
  * strength 0 changes nothing (generation identical to no hook)
  * the hook edits only the chosen head's slice, only at the last position ('last') or from the start
    position on ('from')
  * mean head inputs from a left-padded batch equal those from one prompt at a time
  * steered generation in a left-padded batch equals steered generation one prompt at a time
  * the teacher-forced steered log-probability of the first answer token equals the steered generation's
  * strength 0 teacher-forced log-probability equals a plain forward pass

    python Tests/exp4_steer_selftest.py                  # on the GPU box; exit code 0 = OK
    python Tests/exp4_steer_selftest.py G L --cpu-ok     # local test with two model folders
"""

import sys
import traceback
from pathlib import Path

import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from exp4_steer import Steer, build_prompt, head_inputs, teacher_logprob  # noqa: E402
from gpu_utils import MODEL_PATHS, generate  # noqa: E402

print(f"torch {torch.__version__} | transformers {transformers.__version__}", flush=True)
if not torch.cuda.is_available() and "--cpu-ok" not in sys.argv:
    sys.exit("FAIL: PyTorch cannot see the GPU")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
PATHS = dict(zip(["gemma", "llama"], args)) if len(args) == 2 else {k: MODEL_PATHS[k] for k in ("gemma", "llama")}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPTS = [build_prompt(["The first manmade object to reach space was a V-2 rocket on 3 October 1942."],
                        "when was the first manmade object sent into space"),
           build_prompt(["Sputnik 1 was launched by the Soviet Union in 1957.", "It was the first satellite."],
                        "what was the first artificial satellite"),
           build_prompt(["Short."], "who")]


def tiny(path):
    cfg = AutoConfig.from_pretrained(path)
    d = cfg.to_dict()
    d.update(num_hidden_layers=3, hidden_size=64, intermediate_size=128, num_attention_heads=4,
             num_key_value_heads=2, head_dim=16)
    d.pop("layer_types", None)
    cfg = type(cfg).from_dict(d)
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(cfg, attn_implementation="eager")
    return m.to(device=DEVICE, dtype=torch.float32).eval()


failed, checked = [], []
for name in ("gemma", "llama"):
    try:
        tok = AutoTokenizer.from_pretrained(PATHS[name])
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        model = tiny(PATHS[name])
        st = Steer(model)
        hd = st.hd
        problems = []
        heads = [(1, 2), (2, 0)]
        torch.manual_seed(1)
        dirs = {h: torch.randn(hd) for h in heads}
        # 1. strength 0
        plain = generate(model, tok, PROMPTS, max_new_tokens=8, token_budget=10 ** 6)
        st.set(heads, dirs, 0.0)
        zero = generate(model, tok, PROMPTS, max_new_tokens=8, token_budget=10 ** 6)
        st.clear()
        if [r["text"] for r in plain] != [r["text"] for r in zero]:
            problems.append("strength 0 changed the output")
        # 2. slice and position
        seen = {}
        cap = model.model.layers[1].self_attn.o_proj.register_forward_hook(
            lambda m, a, o: seen.__setitem__("x", a[0].detach().clone()))
        ids = tok(PROMPTS[0], return_tensors="pt")["input_ids"].to(DEVICE)
        with torch.no_grad():
            model(input_ids=ids)
            base = seen["x"]
            st.set(heads, dirs, 5.0, mode="last")
            model(input_ids=ids)
            st.clear()
            diff = (seen["x"] - base).abs().sum(-1)[0]          # per position
            sl = (seen["x"] - base)[0, -1].reshape(-1, hd).abs().sum(-1)
            if diff[:-1].max() > 0 or diff[-1] == 0:
                problems.append("'last' mode edited positions other than the last")
            if not (sl[2] > 0 and sl[[0, 1, 3]].max() == 0):
                problems.append(f"'last' mode edited the wrong head slice: {sl.tolist()}")
            st.set(heads, dirs, 5.0, mode="from", start=ids.shape[1] - 3)
            model(input_ids=ids)
            st.clear()
            diff = (seen["x"] - base).abs().sum(-1)[0]
            if diff[:-3].max() > 0 or (diff[-3:] == 0).any():
                problems.append("'from' mode edited the wrong positions")
        cap.remove()
        # 3. batched head inputs
        mb = head_inputs(model, tok, PROMPTS, [1, 2], 10 ** 6, 8)
        singles = [head_inputs(model, tok, [p], [1, 2], 10 ** 6, 8) for p in PROMPTS]
        for l in (1, 2):
            ref = sum(s[l] for s in singles) / len(singles)
            e = (mb[l] - ref).abs().max().item()
            if e > 1e-4:
                problems.append(f"batched head input differs at layer {l}: {e:.2e}")
        # 4. steered generation, batch vs single
        st.set(heads, dirs, 20.0)
        gb = [r["text"] for r in generate(model, tok, PROMPTS, max_new_tokens=8, token_budget=10 ** 6)]
        gs = [generate(model, tok, [p], max_new_tokens=8, token_budget=10 ** 6)[0]["text"] for p in PROMPTS]
        st.clear()
        if gb != gs:
            problems.append(f"steered batch vs single generation differ: {gb} vs {gs}")
        if gb == [r["text"] for r in plain]:
            problems.append("strength 20 did not change any output (hook not applied?)")
        # 5. teacher-forced first token = steered generation's first-step log-probability
        p = PROMPTS[1]
        ids = tok(p, return_tensors="pt")["input_ids"].to(DEVICE)
        st.set(heads, dirs, 20.0)
        with torch.no_grad():
            out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=1, do_sample=False,
                                 return_dict_in_generate=True, output_logits=True, pad_token_id=tok.pad_token_id)
        st.clear()
        first_id = out.sequences[0, -1].item()
        lp_gen = torch.log_softmax(out.logits[0][0].float(), -1)[first_id].item()
        word = tok.decode([first_id])
        a_ids = tok(" " + word.strip(), add_special_tokens=False)["input_ids"]
        if word.strip() and a_ids == [first_id]:
            lp_tf = teacher_logprob(model, tok, p, word, st, heads, dirs, 20.0)
            checked.append(name)
            if abs(lp_tf - lp_gen) > 1e-3:
                problems.append(f"teacher-forced vs generated log-prob: {lp_tf:.4f} vs {lp_gen:.4f}")
        # 6. strength 0 teacher-forced = plain forward
        ans = "Sputnik 1"
        lp0 = teacher_logprob(model, tok, p, ans, st, heads, dirs, 0.0)
        a_ids = tok(" " + ans, add_special_tokens=False, return_tensors="pt")["input_ids"].to(DEVICE)
        full = torch.cat([ids, a_ids], 1)
        with torch.no_grad():
            lg = model(input_ids=full).logits[0].float()
        ref = torch.log_softmax(lg[ids.shape[1] - 1:-1], -1).gather(1, full[0, ids.shape[1]:].unsqueeze(1)).sum().item()
        if abs(lp0 - ref) > 1e-3:
            problems.append(f"strength-0 teacher-forced log-prob {lp0:.4f} vs plain {ref:.4f}")
        print(f"{name}: {'OK' if not problems else 'FAIL'} (steered outputs: {gb}; "
              f"first-token check {'run' if name in checked else 'skipped: token does not round-trip'})")
        for x in problems:
            print("   -", x)
        if problems:
            failed.append(name)
    except Exception:                                                           # noqa: BLE001
        traceback.print_exc()
        failed.append(name)
if failed:
    sys.exit(f"STEERING SELF-TEST FAILED for {failed}")
print("STEERING SELF-TEST PASSED")
