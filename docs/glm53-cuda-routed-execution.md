# GLM routed expert execution with bounded CUDA banks

This records the supplied-route checkpoint. The subsequent
[complete MoE qualification](glm53-cuda-moe-layer.md) adds routing, the shared
branch and RAM source-cache policy. Remaining-work statements below describe
this earlier checkpoint.

The CUDA expert-bank executor now supports GLM's 288-expert, top-eight
execution geometry, per-submission clamped activation, and Q5_K down weights.
The existing zero-initialized configuration retains 512 experts, top-ten
geometry and ordinary activation when its submission is also zero-initialized.
Grouping and reduction accept explicit expert/top-k geometry.

This is routed **execution with supplied routes**, not a qualified GLM router
or complete MoE layer. GLM routing projection, biased sigmoid selection, shared
expert composition, model bindings, indexed MLA and complete generation remain
open. The ordinary GLM loader remains guarded. The existing CUDA model-owned
expert cache still supports its qualified 512-expert/top-ten bank profile;
generalized streaming banks do not imply GLM VRAM-cache support.

## Checkpoint formats and reference semantics

The actual checkpoint uses the following routed weights in block three:

| Role | GGUF type | Shape per expert | Raw bytes per expert |
|---|---|---|---:|
| Gate | Q4_K (12) | 2048 × 4096 | 4,718,592 |
| Up | Q4_K (12) | 2048 × 4096 | 4,718,592 |
| Down | Q5_K (13) | 4096 × 2048 | 5,767,168 |

Q5_K is GGUF type 13; Q6_K is type 14. Earlier working handoff notes mislabeled
type 13. The directory's 176-byte quant blocks and actual payload sizes confirm
Q5_K. No Q6_K bank capability is added by this change.

The reference uses the original CUDA GGUF routed vector kernels through 64
**total input tokens**, and original grouped routed matrix kernels above 64.
Per-expert row count does not change that precision selection. The merged
original gate/up result is passed to the original CUDA clamped SiLU operation
with limit ten. The SiLU intermediate rounds to BF16 before the product; the
asymmetric clamps and positive-zero addition retain the contract documented in
the [dense-block qualification](glm53-cuda-dense-prefix.md).

The installed GGUF plugin's complete MoE method forwards an activation name
without GLM's clamp configuration. Its ordinary SILU path consequently omits
the model's limit. This capture explicitly composes the original projection,
clamped-activation and reduction operators. It does not claim whole-plugin or
whole-vLLM GLM equivalence, or modify the plugin to manufacture agreement.

## API and ownership

`BnGPUMoEBankConfig` adds `n_experts` and `k`. Zero selects the legacy default
for each field. Explicit expert counts range from 1 to 512, and top-k ranges
from 1 to `min(experts,32)`. The bank pool's dimensions, geometry and payload
budgets are fixed at preparation. Repeated reserve calls must match.

`BnGPUMoEBankResidentPlan` adds `clamped` and `clamp_limit`. These belong to each
submission, so one pool can serve layers with distinct activation metadata.
Zero preserves ordinary activation; the explicit clamped profile requires a
finite nonnegative limit. Validation precedes lease consumption or CUDA work.
The transformer/model binder must provide the anatomy-owned per-layer limit.

Grouping sizes offsets and cursors from the actual expert count, pads its
internal prefix scan safely to 512 threads, and maps rows using the actual
rank count. Reduction restores original token/rank order, rounds each routing
weight product to BF16 and sums those values in F32 rank order before rounding
the final result. Invalid IDs leave gathered activations unchanged and set
status; callers must inspect that status before streaming expert weights.

Input quantization and grouped down scratch are reserved using the actual
rank count. Original activations are quantized once and mapped into expert
order. Each grouped bank queues gate/up projection, activation, down-input
quantization and down projection. Above the matrix threshold it uses the
existing tiled integer-MMA Q4_K/Q5_K cores. Per-expert execution remains
available under the same arithmetic contract. No CPU activation bridge or
weight decoding is introduced.

The two pinned-host/device bank generations retain their event-protected
leases and dedicated copy stream. Acquire may report busy; callers retry the
same pending work. Model weights remain immutable raw CPU-visible data;
request state owns bank/scratch resources. Shared cache binding rejects
unsupported bank geometry before allocating cache binding resources.

## Memory accounting and replay

One GLM routed expert has 15,204,352 raw bytes. Reservation still covers the
largest supported Q5_K gate/up plus Q8_0 down pair, 20,447,232 bytes per entry.
With two bank slots, payload reservation is fixed as follows in both pinned RAM
and GPU memory:

| Experts per slot | Reserved bytes per memory domain |
|---:|---:|
| 1 | 40,894,464 |
| 3 | 122,683,392 |
| 8 | 327,155,712 |

Quantized-input scratch, grouped down scratch, descriptor buffers and activation
buffers have separate sizes and budgets. These payload figures exclude CUDA
allocator/context/event overhead, mapped checkpoint pages and OS file cache.
They are not whole-model RAM/VRAM working-set measurements.

The actual-weight replay uses twelve checkpoint experts, including IDs zero
and 287. Each token has eight supplied distinct routes and F32 routing weights.
It compares up projections, clamped activations, down projections, individually
weighted outputs and final reduction against original captures. Grouped row
order is not assumed; comparisons use the native inverse mapping. It checks
that upload bytes equal exactly the active raw expert payload and that fixed
bank reservations remain unchanged across capacities. This does not qualify
router decisions, prefill/decode admission policy, warm VRAM hits or a bounded
RAM source cache for GLM.

The reference and native harness are
`test/reference_glm_routed_ffn.py` and `test/test_cuda_glm_routed_ffn.c`. The
optional `test_cuda_glm_routed_ffn` target builds the actual-checkpoint replay;
it does not run a model automatically. See the machine-readable record for
final validation counts and commands. The full modern-model goal remains active.

## Qualification results

The final replay passes **165,888,000 direct word comparisons** across 36 cases:
six token counts, three bank capacities, and both per-expert and grouped-bank
execution. Original captures reproduce exactly in a separate run; the reference
asserts that both Q4_K and Q5_K routed operations use the original CUDA extension
and records its binary hash. Earlier expected values and tolerances are unchanged.

The protected clean build is warning-free, and the complete standard suite,
complete CUDA backend suite and diagnostic isolation pass. Existing bank/cache
checks pass, including 192 nonzero grouped matrix/vector differential cases.
Synthetic checks cover 1/288/512 experts, top-k 1/8/32, exact grouping and simple
reduction values, invalid IDs, invalid activation metadata and ownership/budget
failures. The earlier three-block dense replay remains exact for 221,074,320
words. CUDA memcheck and synccheck report zero errors for both the actual routed
replay and the bank tests. A misleading-indentation warning in a new synthetic
test was fixed with a token-identical line break; its bank/cache build is
warning-free and both bank sanitizers were rerun after that formatting fix.

A warm-source-page trace captures four grouped bank submissions at 65 tokens,
capacity three and two bank slots. Grouping/original-input quantization precede
the capture; final routing reduction and comparisons follow it. The captured
expert-bank interval records:

- 16 kernels: eight tiled gate/up or down projection launches, four activations
  and four down-input quantizations.
- 182,452,224 weight-upload bytes and 144 descriptor-upload bytes; no activation
  transfer and no allocation API.
- Four copy-ready device event waits, four nonblocking event queries and one
  explicit final host stream fence, with no host completion fence between banks.

CUPTI also records an unattributed context synchronization after the final
kernel. The post-capture result passes 6,656,000 exact comparisons. This trace
streams the experts again; it measures neither warm VRAM-cache hits nor a cold
source-I/O distribution, and establishes no throughput or vLLM parity claim.
