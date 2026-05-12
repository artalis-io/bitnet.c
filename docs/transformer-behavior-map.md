# Transformer Behavior Map

This file freezes the current branch surface of `src/transformer.c` before the
planner/executor split is completed. It is intentionally descriptive: each row
names an existing route that should remain visible in tests or planner policy.

## CPU Decode

| Area | Route | Current decision source | Coverage |
|---|---|---|---|
| Layer kind | full-attention layer vs SSM layer | `full_attn_interval`, `bn_transformer_is_attn_layer` | `test_layer_shape_planning` |
| Layer index | attention and SSM compact indices | `bn_transformer_attn_index`, `bn_transformer_ssm_index` | `test_layer_shape_planning` |
| Q shape | classic Q, gated Q, wide Q | `BnLayerShapePlan.kind` | `test_layer_shape_planning` |
| KV cache | FP32, FP16, TurboQuant | `BnLayerShapePlan.kv_mode` | `test_layer_shape_planning` |
| Attention norms | optional Q/K RMSNorm, per-head stride | `q_norm`, `k_norm`, `qk_norm_per_head` | `test_layer_shape_planning` |
| Attention bias | optional Q/K/V bias | `q_bias`, `k_bias`, `v_bias` | `test_layer_shape_planning` |
| Attention math | standard GQA vs TurboQuant GQA | `kv_mode`, `m->tq_state` | `make test`, coherence |
| SSM block | conv, L2 norm, delta recurrence, gate, output | `BnSSMPlan` and SSM config fields | `test_block_planning`, `test_ssm` |
| FFN kind | dense up-only, dense gate/up, MoE | `BnFFNPlan.kind` | `test_block_planning`, `test_moe` |
| FFN activation | SiLU vs ReLU2 | `BnFFNPlan.activation` | `test_block_planning`, `make test` |
| FFN CPU fast path | AVX2 pre-Q8K for Q4_K/Q6_K | compile target, k-quant type | `make test` |
| FFN CPU fast path | NEON repacked Q4_0 gate/up SiLU | compile target, repack scales | `make test`, coherence |
| Logits | tied F32, tied F16, tied INT8, untied output | `BnLogitsPlan.kind` | `test_block_planning`, `make test` |

## Batch Prefill

| Area | Route | Current decision source | Coverage |
|---|---|---|---|
| Layer kind | batched attention vs SSM | `BnLayerShapePlan.is_attn` | `make test` |
| Q shape | classic/gated/wide per-token extraction | `BnLayerShapePlan` | `make test` |
| KV cache | FP32/FP16/TurboQuant writes | config KV mode | `make test` |
| FFN | dense batch matmul vs batch MoE | `router_weight`, FFN config | `make test`, model matrix smoke |

## GPU Decode

| Area | Route | Current decision source | Coverage |
|---|---|---|---|
| Backend availability | GPU graph vs CPU fallback | backend vtable and validation | `test_gpu_capability_routing`, coherence |
| Capability checks | Q4/Q8/Q5 split, fused gate/up, flash attention | `bn_transformer_gpu_can_*` | `test_gpu_capability_routing` |
| Attention QKV | packed QKV, split stacked QKV, separate Q/K/V | tensor layout, quant type, caps | `test_block_planning`, coherence |
| Gated Q | deinterleave Q then sigmoid gate | `BnLayerShapePlan.q_gated` | `test_layer_shape_planning` |
| Q/K norms | per-head RMSNorm shader when uploaded | norm GPU handles | `make BN_ENABLE_METAL=1 test_coherence` |
| Attention kernel | flash attention vs scores/softmax/combine | `BnAttentionPlan.use_flash`, caps | `test_block_planning` |
| SSM | CPU fallback unless forced GPU graph | placement/fallback policy | `test_block_planning`, coherence |
| MoE | CPU fallback in default GPU-resident path | `BnFFNPlan.kind`, `BnMoEPlan` | `test_block_planning` |
| Dense FFN | fused Q4_0 gate/up, Q4_K split, Q8 split, separate matvecs | `emit_gpu_dense_ffn_ops`, caps | `test_block_planning`, coherence |
| Logits | GPU matvec or CPU fallback for oversized binding | binding size and logits plan | coherence |

## Verification Gates

For transformer architecture changes, use this minimum gate:

1. `make test_transformer`
2. `make clean && make bitnet`
3. `make test`
4. `make BN_ENABLE_METAL=1 bitnet test_coherence`
5. `./test_coherence models/qwen2.5-3b-instruct-q4_0.gguf --metal`
6. Short bitnet vs `llama-bench` CPU/Metal checkpoint on the same local model

