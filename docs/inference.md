# Inference Pipeline

How bitnet.c turns a text prompt into generated tokens. This document traces the complete pipeline from raw text input through tokenization, the transformer forward pass, and sampling — with the underlying math and pointers to the scalar reference code.

All code references point to the scalar (non-SIMD) implementations, which are the easiest to read and mathematically equivalent to the optimized NEON/AVX2/WASM backends.

## Overview

```
prompt string
    │
    ▼
┌──────────┐
│ Tokenize │  BPE encode: "Hello world" → [128000, 9906, 1917]
└──────────┘
    │
    ▼  for each token:
┌──────────────────────────────────────────┐
│            Forward Pass                  │
│                                          │
│  1. Embed token → x ∈ ℝ^dim             │
│  2. For each layer l = 0..L-1:           │
│     a. RMSNorm(x)                        │
│     b. QKV projection (ternary matvec)   │
│     c. RoPE positional encoding          │
│     d. KV cache write                    │
│     e. GQA attention                     │
│     f. Output projection + residual      │
│     g. RMSNorm(x)                        │
│     h. FFN (gate * up, activation, down) │
│     i. Residual connection               │
│  3. Final RMSNorm                        │
│  4. Logits = embedding^T @ x             │
└──────────────────────────────────────────┘
    │
    ▼
┌──────────┐
│  Sample  │  argmax / temperature+softmax / top-p nucleus
└──────────┘
    │
    ▼
  output token → decode to text, repeat
```

## 1. Tokenization (BPE)

**Algorithm:** Byte Pair Encoding (GPT-2/GPT-4 style byte-level BPE).

**Math:** BPE is a greedy compression algorithm. Given a vocabulary V of tokens with merge scores, and input text:

1. **Initialize:** Map each byte of the input to its corresponding single-byte token in V. Non-printable bytes (space, tab, etc.) are mapped to Unicode codepoints U+0100..U+0143 before lookup.

2. **Merge loop:** Repeat until no merges remain:
   - For every adjacent pair (tokens[i], tokens[i+1]), concatenate their strings and look up the result in V.
   - Among all pairs that exist in V, pick the one with the highest merge score.
   - Replace the pair with the merged token, reducing the sequence length by 1.

3. **Prepend BOS:** If the model requests it, prepend the beginning-of-sequence token (typically token 128000).

**Code:** `src/tokenizer.c:165-293` (`bn_tokenizer_encode`). The merge loop is at line 256. Vocab lookup uses binary search over a pre-sorted index (`vocab_lookup`, line 50).

**Example:** `"Hello"` → bytes `[72, 101, 108, 108, 111]` → initial tokens `[H, e, l, l, o]` → after merges → `[Hello]` (single token 9906).

## 2. Token Embedding

**Math:** Look up a learned vector for the token:

```
x = E[token]    where E ∈ ℝ^{V×d}, x ∈ ℝ^d
```

For bitnet-b1.58-2B-4T: V = 128,256 vocabulary size, d = 2,560 embedding dimension. The embedding table is stored as FP16 in the GGUF file, converted to FP32 on read.

**Code:** `src/model.c` (`bn_model_embed_token`). For F16 embeddings, each row is 2,560 FP16 values (5,120 bytes) converted to float32.

## 3. The Layer Loop

The model has L = 30 identical transformer layers. Each layer transforms x ∈ ℝ^d through attention and feed-forward blocks with residual connections.

**Code:** `src/transformer.c` (`forward_layers`), the main `for (int l = 0; l < c->n_layers; l++)` loop.

### 3a. RMSNorm (Root Mean Square Layer Normalization)

**Math:** Normalize the activation vector, then scale element-wise by learned weights:

```
RMSNorm(x, w) = w ⊙ (x / √(mean(x²) + ε))
```

Expanded:

```
ss = (1/d) Σᵢ xᵢ²
scale = 1 / √(ss + ε)
outᵢ = xᵢ · scale · wᵢ
```

Where ε = 10⁻⁵ prevents division by zero, and w ∈ ℝ^d are learned per-element scale parameters.

Unlike LayerNorm, RMSNorm does **not** subtract the mean — it only normalizes by the root mean square. This saves one pass over the data and empirically works as well for transformer training.

**Code:** `src/transformer/rmsnorm_scalar.c` (8 lines — the entire algorithm):
```c
float ss = 0.0f;
for (int i = 0; i < size; i++) ss += x[i] * x[i];
ss = 1.0f / sqrtf(ss / size + eps);
for (int i = 0; i < size; i++) out[i] = x[i] * ss * w[i];
```

### 3b. QKV Linear Projections (Ternary Matrix-Vector Multiply)

**Math:** Three linear projections compute Query, Key, and Value vectors:

```
q = W_Q · x_norm     W_Q ∈ ℝ^{d×d}        → q ∈ ℝ^d
k = W_K · x_norm     W_K ∈ ℝ^{kv_dim×d}   → k ∈ ℝ^{kv_dim}
v = W_V · x_norm     W_V ∈ ℝ^{kv_dim×d}   → v ∈ ℝ^{kv_dim}
```

For bitnet-b1.58-2B-4T: d = 2,560, kv_dim = 640 (5 KV heads × 128 head_size), so W_Q is 2560×2560 and W_K, W_V are 640×2560.

**BitNet ternary weights:** In BitNet b1.58, these weight matrices are constrained to {-1, 0, +1}. This means matrix-vector multiply reduces to conditional addition/subtraction — no floating-point multiplications needed for the weight side.

**Q2_K format (2.625 bpw, 84 bytes per 256 elements):** Each block has 64 bytes of 2-bit quantized values (4 per byte), 16 bytes of 4-bit scale+min pairs (one per 16-element group), and FP16 super-block scale `d` and minimum `dmin`. Dequantization: `value = d * (scale & 0xF) * q2_value - dmin * (scale >> 4)` where `q2_value` is a 2-bit unsigned integer [0,3].

The I2_S format packs 4 ternary values per byte using 2-bit encoding: `0 → -1, 1 → 0, 2 → +1`. Each weight row also has a single per-tensor floating-point scale factor s, so the actual computation is:

```
outᵢ = s · Σⱼ ternary(Wᵢⱼ) · xⱼ
```

**Code:** `src/quant/i2s_scalar.c` (`bn_quant_i2s_scalar_range`). The interleaved byte layout packs values from 4 sub-rows of 32 elements into each group of 32 bytes:
```c
uint8_t b = rd[gp];
sum += imap[(b >> 6) & 3] * x[done + 0*32 + gp];  // bits 7-6
sum += imap[(b >> 4) & 3] * x[done + 1*32 + gp];  // bits 5-4
sum += imap[(b >> 2) & 3] * x[done + 2*32 + gp];  // bits 3-2
sum += imap[(b >> 0) & 3] * x[done + 3*32 + gp];  // bits 1-0
```

Where `imap[4] = {-1, 0, 1, 0}` maps 2-bit codes to ternary values.

**Dispatch:** All three projections (Q, K, V) are dispatched as a single batch matvec to the thread pool, sharing the same input vector. See `src/quant/dispatch.c` (`bn_quant_matvec_batch`).

### 3c. RoPE (Rotary Position Embeddings)

**Math:** RoPE encodes position information by rotating pairs of dimensions in the Q and K vectors. For each consecutive pair (x₂ᵢ, x₂ᵢ₊₁):

```
θᵢ = 1 / base^(2i/d_head)         base = 10,000
angle = pos · θᵢ

x'₂ᵢ   = x₂ᵢ   · cos(angle) - x₂ᵢ₊₁ · sin(angle)
x'₂ᵢ₊₁ = x₂ᵢ   · sin(angle) + x₂ᵢ₊₁ · cos(angle)
```

This is a 2D rotation of each pair by an angle proportional to position. Lower-frequency rotations (small i, large θ) encode coarse position; higher-frequency rotations (large i, small θ) encode fine position. The key property: the dot product q·k depends only on the *relative* position difference, not absolute positions.

**Optimization:** The frequencies θᵢ are precomputed once at model load time (`s->rope_freq`). The cos/sin values are computed once per token position (128 trig calls for head_size=128), then reused across all 30 layers and all heads.

**Code:** `src/transformer.c`, inside `forward_layers`. Frequency precomputation is in `src/model.c` (`bn_model_load`). The RoPE application loop:
```c
for (int i = 0; i < dim; i += 2) {
    int fi = (i / 2) % half_head;
    float v0 = s->q[i], v1 = s->q[i + 1];
    s->q[i]     = v0 * rope_cos[fi] - v1 * rope_sin[fi];
    s->q[i + 1] = v0 * rope_sin[fi] + v1 * rope_cos[fi];
}
```

### 3d. KV Cache

After RoPE, the K and V vectors for the current position are written to the **KV cache** — a per-layer ring buffer that stores all previously computed keys and values:

```
key_cache[layer][pos % seq_len]   = k
value_cache[layer][pos % seq_len] = v
```

The ring buffer allows inference to continue past `seq_len` by overwriting the oldest entries (sliding window attention).

**Optional FP16 storage:** With `--kv16`, K and V are converted from FP32 to FP16 before writing, halving the cache memory from ~298 MB to ~149 MB. The conversion uses hardware F16↔F32 instructions where available.

**Code:** `src/transformer.c`, the `if (c->kv_f16) { ... }` branches in `forward_layers`.

### 3e. Grouped-Query Attention (GQA)

**Math:** GQA computes attention scores between each query head and its associated KV heads. With n_heads = 20 query heads and n_kv_heads = 5 KV heads, each KV head is shared by kv_mul = 4 query heads.

For each query head h (h = 0..19):

**Step 1 — Attention scores:** Dot product of this head's query with all cached keys, scaled by √d_head:

```
attᵢ = (q_h · k_{h/kv_mul, pos_i}) / √d_head
```

Where pos_i ranges over all valid cached positions (up to seq_len).

**Step 2 — Softmax:** Convert scores to a probability distribution:

```
αᵢ = exp(attᵢ - max(att)) / Σⱼ exp(attⱼ - max(att))
```

The max subtraction prevents numerical overflow (log-sum-exp trick).

**Step 3 — Weighted value sum:** Output is the attention-weighted combination of cached values:

```
out_h = Σᵢ αᵢ · v_{h/kv_mul, pos_i}
```

All heads are concatenated: out = [out_0 ‖ out_1 ‖ ... ‖ out_19] ∈ ℝ^d.

**Code:** `src/transformer/gqa_scalar.c` (`bn_transformer_gqa_scalar_range`). The three steps are clearly visible: dot product loop (line 33-35), softmax call (line 38), weighted sum loop (line 53-54).

**Flash attention variant:** `src/transformer/gqa_neon.c` (`bn_transformer_flash_gqa_neon_range`) implements online softmax — it computes the attention output in a single pass over the KV cache without materializing the full attention score vector. This uses the online softmax algorithm: maintain a running max and running sum, rescaling the partial output whenever a new maximum is found. Numerically equivalent, but more cache-friendly for long sequences.

### 3f. Output Projection + Residual

**Math:**

```
x_out = W_O · attention_output       W_O ∈ ℝ^{d×d}
x = x + x_out                        (residual connection)
```

The residual connection adds the attention output back to the pre-attention activation, allowing gradients and information to flow directly through the network depth.

**Code:** `src/transformer.c`, the `wo` matvec dispatch followed by the SIMD residual add loop.

### 3g–3i. Feed-Forward Network (FFN)

**Math:** The FFN uses a gated architecture (SwiGLU variant with ReLU² activation for BitNet):

```
x_norm = RMSNorm(x)
gate = W_gate · x_norm         W_gate ∈ ℝ^{h×d}
up   = W_up   · x_norm         W_up   ∈ ℝ^{h×d}

activated = ReLU²(gate) ⊙ up   (element-wise)
down = W_down · activated       W_down ∈ ℝ^{d×h}

x = x + down                   (residual connection)
```

Where h = 6,912 is the hidden dimension (2.7× the model dimension).

**ReLU² activation:** `ReLU²(x) = max(x, 0)²`. This is the squared ReLU — first clamp negatives to zero, then square the result. It provides sharper gating than standard ReLU.

**SiLU alternative:** For non-BitNet models, the activation is SiLU (Sigmoid Linear Unit): `SiLU(x) = x · σ(x) = x / (1 + e^(-x))`.

**Code:** `src/transformer.c`, the `// ---- FFN block ----` section. The gate and up projections are batched into a single matvec dispatch. The ReLU² scalar code:
```c
float g = s->hb[i] > 0 ? s->hb[i] : 0;
s->hb[i] = g * g * s->hb2[i];
```

### Sub-norms (BitNet-specific)

BitNet models add extra RMSNorm layers after the attention output and after the FFN gate, before the output projection. These "sub-norms" stabilize the ternary weight training. Standard transformer models don't have them.

**Code:** The `if (lw->attn_sub_norm)` and `if (lw->ffn_sub_norm)` checks in `forward_layers`.

## 4. Final RMSNorm + Logits

After all L layers, the final activation x ∈ ℝ^d is normalized and projected to vocabulary-sized logits:

**Math:**

```
x_final = RMSNorm(x, w_output_norm)

logitsᵥ = Σᵢ Eᵥᵢ · x_finalᵢ    for each v ∈ {0, ..., V-1}
```

This is a matrix-vector multiply of the embedding matrix transposed (or a separate output weight matrix for untied embeddings) with the final hidden state. The result is V = 128,256 raw logit scores — one per vocabulary token.

**Performance note:** This is the most expensive single operation per token. For the 2B model with INT8 embeddings, it reads 128,256 × 2,560 = 328 MB of weight data — 40% of the per-token memory bandwidth.

**Code:** `src/transformer.c` (`forward_logits`). The scalar kernel is `src/transformer/logits_scalar.c`:
```c
for (int v = v_start; v < v_end; v++) {
    const float *row = emb + (size_t)v * dim;
    float sum = 0.0f;
    for (int d = 0; d < dim; d++) sum += row[d] * x[d];
    lc->logits[v] = sum;
}
```

## 5. Sampling

The logits vector ℓ ∈ ℝ^V is converted to a token selection. Three strategies:

### Greedy (argmax)

```
token = argmax(ℓ)
```

Deterministic. Used when temperature = 0.

**Code:** `src/sampler.c:62-70`.

### Temperature + Multinomial

**Math:** Scale logits by temperature T, convert to probabilities via softmax, then sample:

```
ℓ'ᵢ = ℓᵢ / T

pᵢ = exp(ℓ'ᵢ - max(ℓ')) / Σⱼ exp(ℓ'ⱼ - max(ℓ'))

token ~ Categorical(p)
```

Temperature T < 1 sharpens the distribution (more deterministic), T > 1 flattens it (more random). T = 0 degenerates to argmax.

The categorical sampling draws a uniform random number r ∈ [0,1) and walks the CDF until cumulative probability exceeds r:

```
r ~ Uniform(0,1)
token = min{k : Σᵢ₌₀ᵏ pᵢ > r}
```

**Code:** `src/sampler.c:87-95` (`sample_multinomial`). The RNG is xorshift64 (lines 5-9).

### Top-p (Nucleus) Sampling

**Math:** Restrict sampling to the smallest set of tokens whose cumulative probability exceeds threshold p:

1. Sort tokens by probability descending: p_π(1) ≥ p_π(2) ≥ ...
2. Find the smallest k such that Σᵢ₌₁ᵏ p_π(i) > p
3. Renormalize probabilities over {π(1), ..., π(k)}
4. Sample from the renormalized distribution

This dynamically adjusts the candidate set size — for peaked distributions (confident predictions), few tokens are considered; for flat distributions (uncertain), many are.

An optimization: before sorting, tokens with probability below `(1-p)/(V-1)` are pruned, since they can never be in the nucleus.

**Code:** `src/sampler.c:105-145` (`sample_topp`). Uses a preallocated candidates buffer to avoid per-token malloc.

### Repetition Penalty

Before temperature scaling, recently generated tokens have their logits penalized:

```
if ℓᵢ > 0:  ℓᵢ = ℓᵢ / penalty
if ℓᵢ ≤ 0:  ℓᵢ = ℓᵢ × penalty
```

The asymmetric application (divide positive, multiply negative) ensures the penalty always pushes the logit toward zero, regardless of sign. A ring buffer tracks the last N generated tokens.

**Code:** `src/sampler.c:148-159`.

## 6. Autoregressive Loop

The complete generation loop (simplified):

```
tokens = tokenize(prompt)
for each token in tokens:
    forward(model, token, pos++)        # prefill: build KV cache

for step in range(max_tokens):
    logits = forward(model, last_token, pos++)
    next_token = sample(logits)
    print(decode(next_token))
    if next_token == EOS: break
    last_token = next_token
```

**Prefill optimization:** During prompt processing, logits are only computed for the final token (all earlier tokens only need to populate the KV cache). This is implemented as `bn_transformer_prefill` which calls `forward_layers` for all prompt tokens but `forward_logits` only once.

**Code:** `src/transformer.c:406-412` (`bn_transformer_prefill`), `src/main.c`.

## 7. Parallelism

All matrix-vector multiplies and the GQA attention are parallelized across a persistent pthread thread pool (`src/threadpool.c`).

**Static scheduling:** Each dispatch divides N work items (rows or heads) evenly across T threads. Thread 0 (the main thread) participates as a worker — no thread sits idle during dispatch.

**Dispatch cost:** ~2 microseconds per dispatch (condvar signal + wake). The forward pass does 151 dispatches per token (5 matvec dispatches × 30 layers + 1 logits dispatch).

**Code:** `src/threadpool.c` (`bn_tp_dispatch`), `include/threadpool.h`.

## Appendix: Dimensions for bitnet-b1.58-2B-4T

| Parameter | Value |
|-----------|-------|
| dim (d) | 2,560 |
| hidden_dim (h) | 6,912 |
| n_layers (L) | 30 |
| n_heads | 20 |
| n_kv_heads | 5 |
| head_size (d_head) | 128 |
| kv_dim | 640 |
| kv_mul | 4 |
| vocab_size (V) | 128,256 |
| seq_len | 4,096 |
| Weight type | I2_S (ternary) |
| Embedding type | F16 |
| Activation | ReLU² |
