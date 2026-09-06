"""
05_roi_divergence_practice.py

Practice implementation of Option B's ROI metric on GPT-2 (CPU, free)
before building the real version for Gemma/Llama. Computes divergence
between:
  (a) attention over question tokens, WITH a passage present
  (b) attention over the SAME question tokens, with NO context at all
per head, per layer -- not averaged.

Low divergence = model treating the question the same way it would with
zero evidence = hypothesized override signal.

Tests TWO questions -- one well-known, one obscure -- to check whether
the metric responds differently to them in any sensible way.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
model.eval()

num_layers = len(model.transformer.h)


class AttentionCapture:
    """Captures per-layer attention over a specific token span (the
    question), for whichever single forward pass it's attached to."""

    def __init__(self, num_layers, span_start, span_end):
        self.num_layers = num_layers
        self.span_start = span_start
        self.span_end = span_end
        self.per_layer = [None] * num_layers  # each entry: (heads, span_len) tensor
        self._current_layer = 0

    def hook(self, module, input, output):
        layer_idx = self._current_layer
        self._current_layer = (self._current_layer + 1) % self.num_layers

        if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
            return
        attn_weights = output[1]  # (batch, heads, query_len, key_len)

        last_attn = attn_weights[0, :, -1, :]  # (heads, key_len)
        key_len = last_attn.shape[-1]
        s = min(self.span_start, key_len)
        e = min(self.span_end, key_len)
        if e <= s:
            return
        span_attn = last_attn[:, s:e]  # (heads, span_len)
        span_attn = span_attn / (span_attn.sum(dim=-1, keepdim=True) + 1e-12)
        self.per_layer[layer_idx] = span_attn.detach()


def run_capture(prompt: str, span_start: int, span_end: int):
    capture = AttentionCapture(num_layers, span_start, span_end)
    handles = [block.attn.register_forward_hook(capture.hook) for block in model.transformer.h]

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        model(**inputs)

    for h in handles:
        h.remove()
    return capture.per_layer


def cosine_divergence(a, b):
    """1 - cosine_similarity, so 0 = identical, higher = more different."""
    if a is None or b is None:
        return None
    min_heads = min(a.shape[0], b.shape[0])
    a, b = a[:min_heads], b[:min_heads]
    sims = torch.nn.functional.cosine_similarity(a, b, dim=-1)  # per head
    return (1 - sims).tolist()


def find_question_span(full_prompt, question_text, tokenizer):
    """Uses offset_mapping (character-to-token alignment computed on the
    FULL string at once) -- BPE tokenizers don't tokenize consistently
    at boundaries, so tokenizing the prefix separately and adding
    lengths can be off by a token (confirmed: this silently dropped the
    first word in an earlier version of this script)."""
    char_start = full_prompt.index(question_text)
    char_end = char_start + len(question_text)

    encoding = tokenizer(full_prompt, return_tensors="pt", return_offsets_mapping=True)
    offsets = encoding["offset_mapping"][0].tolist()

    token_start = None
    token_end = None
    for i, (s, e) in enumerate(offsets):
        if s == e:
            continue  # special token, no character span
        if token_start is None and e > char_start:
            token_start = i
        if s < char_end:
            token_end = i + 1

    return token_start, token_end


def test_question(label: str, question: str, passage: str):
    print(f"\n{'=' * 70}")
    print(f"TEST: {label}")
    print(f"{'=' * 70}")
    print(f"Question: {question!r}")
    print(f"Passage:  {passage!r}\n")

    no_context_prompt = f"Question: {question}\nAnswer:"
    with_context_prompt = f"{passage}\n\nQuestion: {question}\nAnswer:"

    no_ctx_start, no_ctx_end = find_question_span(no_context_prompt, question, tokenizer)
    with_ctx_start, with_ctx_end = find_question_span(with_context_prompt, question, tokenizer)

    # --- Span verification: both lines below must print the FULL question,
    # nothing more, nothing less. If either looks wrong, stop and debug
    # before trusting any divergence numbers. ---
    no_ctx_ids = tokenizer(no_context_prompt, return_tensors="pt").input_ids
    with_ctx_ids = tokenizer(with_context_prompt, return_tensors="pt").input_ids
    print("No-context question span:  ", repr(tokenizer.decode(no_ctx_ids[0][no_ctx_start:no_ctx_end])))
    print("With-context question span:", repr(tokenizer.decode(with_ctx_ids[0][with_ctx_start:with_ctx_end])))
    print()

    baseline_attn = run_capture(no_context_prompt, no_ctx_start, no_ctx_end)
    context_attn = run_capture(with_context_prompt, with_ctx_start, with_ctx_end)

    print("Per-layer divergence (per head) between with-context and no-context")
    print("attention over the question span:\n")
    layer_means = []
    for layer_idx in range(num_layers):
        div = cosine_divergence(context_attn[layer_idx], baseline_attn[layer_idx])
        if div is None:
            print(f"  layer {layer_idx}: no data captured")
            continue
        mean_div = sum(div) / len(div)
        layer_means.append(mean_div)
        print(f"  layer {layer_idx}: mean divergence = {mean_div:.4f}  "
              f"(per-head: {[f'{v:.3f}' for v in div]})")

    overall = sum(layer_means) / len(layer_means) if layer_means else float("nan")
    print(f"\n  Overall mean divergence across all layers: {overall:.4f}")
    return overall


# --- Test 1: a well-known fact ---
overall_known = test_question(
    "WELL-KNOWN fact",
    "When was the Eiffel Tower completed?",
    "The Eiffel Tower was completed in 1889 in Paris, France.",
)

# --- Test 2: an obscure fact GPT-2 is unlikely to know confidently ---
overall_obscure = test_question(
    "OBSCURE fact",
    "What is the maiden name of the mother of the 1974 winner of the Nobel Prize in Literature?",
    "The 1974 Nobel Prize in Literature was awarded jointly to Eyvind Johnson and Harry Martinson.",
)

print(f"\n{'=' * 70}")
print("COMPARISON")
print(f"{'=' * 70}")
print(f"Well-known fact -- overall mean divergence: {overall_known:.4f}")
print(f"Obscure fact    -- overall mean divergence: {overall_obscure:.4f}")
print(f"Difference: {abs(overall_known - overall_obscure):.4f}")
print("\nWhat to check:")
print("  1. Do the two 'question span' lines above EACH test read the FULL question text,")
print("     nothing more/less? If not, stop -- the span is wrong, ignore the numbers.")
print("  2. Does divergence vary meaningfully across layers/heads in both tests (not flat)?")
print("  3. Do the two tests produce noticeably DIFFERENT overall divergence or per-layer")
print("     patterns? (Don't expect a specific direction to be 'correct' -- GPT-2 is a small,")
print("     weak model. The point is just: does the metric respond to something, or is it")
print("     identical/noise regardless of what question you feed it?)")