"""
Phase 4 addition: Causal tracing for fact-storage localization.

For each probe fact, per model:
  1. Run the CLEAN prompt, cache every layer's MLP output at the subject
     token positions, record the answer token's probability.
  2. CORRUPT the subject by adding noise to its embeddings, run again,
     confirm the answer's probability drops substantially. This is a
     self-check -- if corruption doesn't degrade the answer, that probe
     is skipped and flagged, since results built on a non-corrupted probe
     are meaningless.
  3. For each layer, patch that layer's CLEAN MLP output back in at the
     subject positions during a corrupted-input run, and record how much
     the answer's probability recovers. High recovery at a layer means
     that layer is causally important for retrieving/storing this fact.

Run from repo root:
    python src/data/extract_causal_tracing.py

Requires: data/final/causal_tracing_probes.jsonl (see
build_causal_tracing_probes.py) -- INSPECT THAT FILE BY HAND FIRST.

Output: data/final/causal_tracing_results.jsonl
  {"query_id": ..., "model": ..., "clean_prob": float,
   "corrupted_prob": float, "recovery_by_layer": [float, ...],
   "corruption_effective": bool}
"""

import json
import gc
from pathlib import Path

import torch
torch.manual_seed(42)  # corruption noise is randomly sampled -- without a
# fixed seed, skip counts vary run-to-run at the SAME noise scale, which
# was confounding comparisons between different NOISE_SCALE values.
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

PROBES_PATH = "data/final/causal_tracing_probes.jsonl"
OUTPUT_PATH = "data/final/causal_tracing_results.jsonl"

MODEL_CONFIGS = {
    "gemma": "/root/models/gemma-2-9b",
    "llama": "/root/models/Llama-3.1-8B",
}

QUANT_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

NOISE_SCALE = 10.0  # multiples of embedding std dev, standard ROME-style corruption strength
CORRUPTION_DROP_THRESHOLD = 0.5  # require at least a 50% relative drop in answer prob to trust the probe


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


class MLPCapture:
    """Captures/patches MLP output at specific token positions, per layer."""

    def __init__(self, num_layers, token_start, token_end):
        self.num_layers = num_layers
        self.token_start = token_start
        self.token_end = token_end
        self.captured = [None] * num_layers  # clean-run cache, filled during clean pass
        self.patch_layer = None  # which layer to patch during a corrupted-with-patch pass
        self._current_layer = 0

    def reset_capture(self, token_start, token_end):
        self.token_start = token_start
        self.token_end = token_end
        self.captured = [None] * self.num_layers
        self.patch_layer = None
        self._current_layer = 0

    def set_patch_layer(self, layer_idx):
        self.patch_layer = layer_idx
        self._current_layer = 0

    def hook(self, module, input, output):
        layer_idx = self._current_layer
        self._current_layer = (self._current_layer + 1) % self.num_layers

        # output shape: (batch, seq_len, hidden)
        if self.patch_layer is None:
            # clean pass: just record this layer's MLP output at the subject span
            self.captured[layer_idx] = output[0, self.token_start:self.token_end, :].detach().clone()
            return output
        elif layer_idx == self.patch_layer and self.captured[layer_idx] is not None:
            # patched pass: overwrite this layer's output at the subject span with the clean version
            patched = output.clone()
            patched[0, self.token_start:self.token_end, :] = self.captured[layer_idx].to(output.dtype)
            return patched
        else:
            return output


def get_answer_token_id(tokenizer, answer_text):
    ids = tokenizer(answer_text, add_special_tokens=False).input_ids
    if not ids:
        return None
    return ids[0]  # first token of the answer as the target


def find_subject_span_in_prompt(full_prompt, subject_text, tokenizer):
    """Locate the subject's token span within the FULL wrapped prompt
    (not the bare question) -- tokenization shifts once the
    Question:/Answer: template is added, so stored offsets from the
    probe-building step don't apply here."""
    idx = full_prompt.find(subject_text)
    if idx == -1:
        return None
    char_start, char_end = idx, idx + len(subject_text)
    encoding = tokenizer(full_prompt, return_tensors="pt", return_offsets_mapping=True)
    token_start = encoding.char_to_token(char_start)
    token_end_char = encoding.char_to_token(char_end - 1)
    token_end = (token_end_char + 1) if token_end_char is not None else None
    if token_start is None or token_end is None:
        return None
    return token_start, token_end


def corrupt_embeddings(model, input_ids, subject_start, subject_end, noise_scale, device):
    embed_layer = model.get_input_embeddings()
    embeds = embed_layer(input_ids).clone()
    # compute std from the SUBJECT tokens specifically, not the whole
    # sequence -- using whole-sequence std dilutes the relative noise
    # applied to a short subject span inside a longer prompt
    subject_embeds = embeds[0, subject_start:subject_end, :]
    std = subject_embeds.std().item()
    noise = torch.randn_like(subject_embeds) * noise_scale * std
    embeds[0, subject_start:subject_end, :] += noise
    return embeds


def run_model_pass(model_key, model_path, probes, device):
    print(f"\nLoading {model_key} ({model_path})...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=QUANT_CONFIG, device_map=device,
        attn_implementation="eager",
    )
    model.eval()

    num_layers = len(model.model.layers)
    capture = MLPCapture(num_layers, 0, 0)
    handles = [layer.mlp.register_forward_hook(capture.hook) for layer in model.model.layers]

    results = []
    skipped_bad_corruption = 0
    skipped_other = 0

    for p in tqdm(probes, desc=f"Causal tracing ({model_key})"):
        try:
            # wrap in the same Question:/Answer: completion template used
            # throughout the rest of this project, so the next-token
            # position is actually primed to predict the answer
            full_prompt = f"Question: {p['question']}\nAnswer:"

            inputs = tokenizer(full_prompt, return_tensors="pt")
            input_ids = inputs.input_ids.to(device)
            answer_token_id = get_answer_token_id(tokenizer, " " + p["answer_text"])
            if answer_token_id is None:
                skipped_other += 1
                continue

            span = find_subject_span_in_prompt(full_prompt, p["subject"], tokenizer)
            if span is None:
                skipped_other += 1
                continue
            subj_start, subj_end = span
            if subj_end > input_ids.shape[1]:
                skipped_other += 1
                continue

            capture.reset_capture(subj_start, subj_end)

            # --- 1. clean pass: cache MLP outputs, get clean answer prob ---
            with torch.no_grad():
                clean_out = model(input_ids)
            clean_prob = torch.softmax(clean_out.logits[0, -1, :], dim=-1)[answer_token_id].item()

            # --- 2. corrupted pass: no patching, confirm answer prob drops ---
            corrupted_embeds = corrupt_embeddings(model, input_ids, subj_start, subj_end, NOISE_SCALE, device)
            capture.set_patch_layer(-1)  # -1 = never matches any real layer index, i.e. no patching
            with torch.no_grad():
                corrupted_out = model(inputs_embeds=corrupted_embeds)
            corrupted_prob = torch.softmax(corrupted_out.logits[0, -1, :], dim=-1)[answer_token_id].item()

            relative_drop = (clean_prob - corrupted_prob) / (clean_prob + 1e-12)
            corruption_effective = relative_drop >= CORRUPTION_DROP_THRESHOLD

            if not corruption_effective:
                skipped_bad_corruption += 1
                continue  # don't trust recovery numbers built on a probe that didn't actually corrupt

            # --- 3. patched passes: one per layer, restore that layer's clean MLP output ---
            recovery_by_layer = []
            for layer_idx in range(num_layers):
                capture.set_patch_layer(layer_idx)
                with torch.no_grad():
                    patched_out = model(inputs_embeds=corrupted_embeds)
                patched_prob = torch.softmax(patched_out.logits[0, -1, :], dim=-1)[answer_token_id].item()
                recovery_by_layer.append(patched_prob)

            results.append({
                "query_id": p["query_id"],
                "model": model_key,
                "clean_prob": clean_prob,
                "corrupted_prob": corrupted_prob,
                "recovery_by_layer": recovery_by_layer,
                "corruption_effective": True,
            })

        except Exception as e:
            print(f"  Skipping {p.get('query_id')} due to error: {e}")
            skipped_other += 1
            continue

    for h in handles:
        h.remove()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Done with {model_key}. Kept {len(results)}, "
          f"skipped {skipped_bad_corruption} (corruption too weak), "
          f"skipped {skipped_other} (other errors).")

    if skipped_bad_corruption > len(probes) * 0.3:
        print(f"  WARNING: over 30% of probes had ineffective corruption for {model_key}.")
        print(f"  Consider increasing NOISE_SCALE (currently {NOISE_SCALE}) and re-running.")

    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU detected -- this will be extremely slow.")

    probes = load_jsonl(PROBES_PATH)
    print(f"Loaded {len(probes)} probes")
    print("Reminder: these probes were auto-extracted with a rough heuristic.")
    print("If results look strange, inspect causal_tracing_probes.jsonl by hand first.\n")

    all_results = []
    for model_key, model_path in MODEL_CONFIGS.items():
        all_results.extend(run_model_pass(model_key, model_path, probes, device))

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    print(f"\nWrote {OUTPUT_PATH} ({len(all_results)} usable rows)")


if __name__ == "__main__":
    main()
