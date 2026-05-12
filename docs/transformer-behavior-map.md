# Transformer Behavior Map

This file records the planner and executor decisions that should stay visible in
tests as the transformer evolves. It reflects the post-split architecture.

## CPU Decode

| Area | Decision source | Coverage |
|---|---|---|
| Attention layer vs SSM layer | `BnLayerShapePlan`, `BnModelArchOps` | `test_layer_shape_planning`, `test_block_planning` |
| Compact attention/SSM index | planner helpers | `test_layer_shape_planning` |
| Classic, gated, or wide Q | layer shape plan | `test_layer_shape_planning` |
| Q/K norms and per-head norms | attention plan | `test_block_planning` |
| Q/K/V bias | attention plan | `test_block_planning` |
| FP32, FP16, or TurboQuant KV | KV plan and session config | `test_layer_shape_planning`, `make test` |
| Standard GQA vs TurboQuant GQA | KV mode and TurboQuant state | `make test`, coherence |
| SSM conv/delta/gate/output path | SSM plan | `test_ssm`, `test_block_planning` |
| Dense FFN vs MoE/shared expert | FFN and MoE plans | `test_moe`, `test_block_planning` |
| SiLU vs ReLU2 activation | FFN plan and arch ops | `test_block_planning` |
| Logits tied/untied and dtype | logits plan | `test_block_planning`, `make test` |

## GPU Decode

| Area | Decision source | Coverage |
|---|---|---|
| GPU graph vs CPU fallback | backend vtable, caps, placement plan | `test_gpu_capability_routing`, coherence |
| QKV packed/split/separate | tensor roles, quant registry, backend caps | `test_block_planning` |
| Fused Q/K RoPE | backend capability and attention shape | backend matrix tests |
| Flash attention vs scores/softmax/combine | attention plan and backend caps | `test_block_planning` |
| Dense FFN fusion/split | quant registry and backend layout | backend matrix tests |
| MoE backend placement | MoE plan and backend caps | `test_block_planning` |
| SSM backend placement | SSM plan and backend caps | `test_block_planning`, coherence |
| Logits GPU vs CPU fallback | logits plan and binding limits | coherence |
| Shader lowering | backend-private `src/gpu_shader.h` | backend matrix tests |

## Architecture Boundaries

| Boundary | Expected rule |
|---|---|
| Model family | Add or update `BnModelArchOps`; avoid transformer cross-product branches. |
| Quant format | Add `BnQuantFormatOps` metadata and kernels; avoid model-family branches. |
| Backend layout | Use `BnBackendModel` and `backend_layout`; do not mutate model anatomy. |
| Backend execution | Lower op codes privately; do not expose shader IDs in public headers. |
| Session state | Keep activation, KV, SSM, and MoE scratch request-local. |

## Minimum Gates

```bash
make test_transformer
make test
make clean
make bitnet
make BN_ENABLE_METAL=1 test_coherence
./test_coherence models/qwen2.5-3b-instruct-q4_0.gguf --metal
./test/backend_matrix.sh
```
