# GLM model anatomy planning

`BnModelArchOps` now recognizes `glm5next` explicitly and owns its metadata,
layer-kind and tensor-role planning through `bn_model_arch_plan_anatomy`.
`BnModelAnatomy` contains dimensions, numerical configuration, per-layer clamp
values and borrowed tensor indices. It allocates no weights or request state,
reads no tensor payload values, and creates no backend handles. The caller keeps
the merged GGUF file alive for any subsequent tensor access.

The target checkpoint has 46 stored blocks, including one auxiliary MTP block.
Its 45 main blocks contain 34 KDA and 11 indexed-MLA blocks. The explicit
`attention.head_count_kv` array selects each block kind; a periodic schedule is
not substituted. The first three FFNs are dense and the remaining 42 are MoE.
All 1,383 main-model/global tensors bind exactly once. The only unbound tensors
are the 29 auxiliary MTP tensors.

## Shape and numerical contracts

KDA has 64 heads of width 128 and separate Q/K/V convolution weights. Both gate
projection pairs retain their 128-wide intermediate materialization. The plan
declares direct F32 sum-epsilon Q/K normalization, the distinct Q/K and output
normalization epsilons, bounded decay, and stored negative-exponential decay
parameters. It does not alias the older packed GDN tensor roles.

MLA has 64 attention heads with 256-wide query/key and value projections and a
512-wide shared latent representation. These are different dimensions, not the
generic GQA head width inferred from the GGUF latent key length:

| Tensor | GGUF dimensions |
|---|---|
| Query expansion | `[1536,16384]` |
| Latent KV projection | `[4096,512]` |
| Key expansion | `[256,512,64]` |
| Value expansion | `[512,256,64]` |
| Attention output | `[16384,4096]` |

The indexer has 32 heads of width 128, pool size 4 and token top-k 2048. Its
LayerNorm bias, F32 head-weight projection and pool compression tensors have
dedicated roles. They do not inherit the older query-sparse-attention roles'
names or precision assumptions. mHC similarly has dedicated function/base/scale
roles, four streams, 20 Sinkhorn iterations and epsilon 1e-6; the old
HC norm/down/up/inject roles are rejected for this family.

Routing configuration includes 288 experts, top-8 selection, one shared expert,
2048-wide expert/shared FFNs, normalized expert weights and scale 2.5. The full
GGUF metadata also contains `swiglu_clamp_exp` and `swiglu_clamp_shexp` arrays.
The planner preserves each layer's values rather than collapsing them into a
single model-wide default. Both arrays contain 10 for this checkpoint.

Planning checks tensor rank, every dimension, required F32 parameter storage,
quant-format registry loadability/capabilities, row block alignment and byte-size
overflow. It rejects missing or duplicate tensor names, duplicate metadata,
invalid topology/numerical metadata and inconsistent layouts. The output remains
unchanged on failure, including failures in the final main block. Payload bounds
and parameter values remain the responsibility of subsequent tensor loading.

The current planner covers the checked no-RoPE GLM layout, 128-wide indexer,
single-group sigmoid routing and one shared expert. Other geometries require an
explicit implementation rather than being silently coerced into this plan.

## Verification and execution status

`test/reference_glm_anatomy.py` independently extracts 41 architecture metadata
entries and all 1,412 tensor descriptions from the six original GGUF shards,
without accessing weight values. The resulting 124 KB fixture needs no model
file for unit tests. Tests verify complete, unique main-model binding, auxiliary
exclusion, semantic role names, per-layer clamp preservation, malformed input
rejection, and rejection by the incompatible legacy loader.

A separate live C audit opens the real six-shard model with the production GGUF
reader and reproduces the same 1,383/29 binding split and 34/11 layer split. The
anatomy tests also pass with AddressSanitizer and UndefinedBehaviorSanitizer.

A protected clean build completes with zero warnings. Standard and full CUDA
regression suites pass, including CUDA diagnostic isolation. A second independent
directory extraction reproduces the fixture and manifest exactly. Changed
source/fixture hashes and the four protected user binaries remain unchanged
through qualification.

This is an architecture and binding-plan step, not completed GLM inference.
The legacy loader now rejects this explicit topology before attempting GQA/GDN
weight loading. A dedicated weight binder/executor still needs to consume the
plan, connect the qualified KDA operations, and implement indexed MLA and mHC.
Default older-family execution remains on its existing path. CPU and CUDA
whole-model qualification, expert placement, serving and vLLM parity remain
part of the active full goal.

Source hashes, directory provenance, live audit and verification results are
recorded in `glm53-model-anatomy.json`.
