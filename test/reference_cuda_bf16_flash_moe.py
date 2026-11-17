"""Actual Flash FFN HC, loaded gates, GGUF routed/shared experts and HC post.

Fresh captures only. Pinned image 745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29,
plugin e454affe with documented expert-count, metadata-index and Q5_1 tile bounds.
Quantizer, dot-product, activation and reduction formulas are unchanged. A reference may hold
one layer's experts in VRAM; target runtime placement must remain bounded.
"""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import gguf
import numpy as np
import torch
import torch.nn.functional as F
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
from vllm.models.qwen4_exp.common.hyperconnection import GatedResidual, HyperConnectionConfig
from vllm_gguf_plugin.quantization import fused_moe as moe_module
from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf

p = argparse.ArgumentParser()
p.add_argument('directory', type=Path)
p.add_argument('inventory', type=Path)
p.add_argument('--layers', type=int, nargs='+', default=[2, 3, 47])
p.add_argument('--tokens', type=int, nargs='+', default=[1, 5, 17, 63, 64, 65, 128, 129, 257])
p.add_argument('--verify-existing', action='store_true')
p.add_argument('--manifest-name', default='manifest.json')
a = p.parse_args()
a.directory.mkdir(exist_ok=a.verify_existing)
torch.set_default_dtype(torch.bfloat16)
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
inventory = json.loads(a.inventory.read_text())
readers = [gguf.GGUFReader(s['path']) for s in inventory['shards']]
tensors = {t.name: t for r in readers for t in r.tensors}
files, weights, cases = {}, {}, []


def save(name, x):
    data = x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    target = a.directory / name
    if a.verify_existing:
        assert target.read_bytes() == data, name
    else:
        assert not target.exists(), name
        target.write_bytes(data)
    files[name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), dtype=str(x.dtype), shape=list(x.shape))


def tensor(layer, suffix):
    name = f'blk.{layer}.' + suffix
    t = tensors[name]
    if name not in weights:
        weights[name] = dict(type=t.tensor_type.name, shape=[int(v) for v in t.shape], bytes=int(t.n_bytes),
                             raw_sha256=hashlib.sha256(memoryview(t.data).cast('B')).hexdigest())
    return t


def raw(layer, suffix):
    return torch.from_numpy(np.array(tensor(layer, suffix).data, copy=True)).cuda()


def decoded(layer, suffix):
    t = tensor(layer, suffix)
    value = torch.from_numpy(np.array(gguf.dequantize(t.data, t.tensor_type), copy=True)).float()
    if suffix == 'hc_ffn_norm.weight':
        value = value - 1.0  # Recover Gamma+1 storage before BF16 conversion.
    return value.bfloat16().cuda()


original_linear = F.linear
original_vec = moe_module.ops.ggml_moe_a8_vec
original_mmq = moe_module.ops.ggml_moe_a8
original_activation = moe_module.apply_moe_activation
original_sum = moe_module.ops.moe_sum
prefix = ''
projection = 0
dispatch = []


def observe_projection(original, mode, *args, **kwargs):
    global projection
    out = original(*args, **kwargs)
    save(prefix + ('-routed-gateup.bin' if projection == 0 else '-routed-down.bin'), out)
    dispatch.append(mode)
    projection += 1
    return out


def observe_activation(kind, out, inp, *args, **kwargs):
    result = original_activation(kind, out, inp, *args, **kwargs)
    save(prefix + '-routed-activation.bin', out)
    return result


def observe_sum(inp, out):
    save(prefix + '-routed-weighted.bin', inp)
    return original_sum(inp, out)


config = VllmConfig()
with set_current_vllm_config(config), patch('vllm.model_executor.parameter.get_tensor_model_parallel_rank', return_value=0), patch('vllm.model_executor.parameter.get_tensor_model_parallel_world_size', return_value=1):
    for layer in a.layers:
        with torch.device('cuda'):
            gate = ReplicatedLinear(2560, 512, bias=False, disable_tp=True)
            shared_gate = ReplicatedLinear(2560, 1, bias=False, disable_tp=True)
        for module, suffix in [(gate, 'ffn_gate_inp.weight'), (shared_gate, 'ffn_gate_inp_shexp.weight')]:
            t = tensor(layer, suffix)
            module.weight_loader(module.weight, torch.from_numpy(np.array(t.data, copy=True)).reshape(module.weight.shape))
        hc = GatedResidual(HyperConnectionConfig(hc_count=4, hidden_size=2560, hc_lowrank=320,
                            hc_per_branch_norm=True, params_dtype=torch.bfloat16), use_combine=True).cuda()
        with torch.no_grad():
            for suffix, parameter in [('norm', hc.hc_norm.weight), ('down', hc.input_mix_weight_down.weight),
                                      ('up', hc.input_mix_weight_up.weight), ('inject', hc.block_inject_weight.weight)]:
                parameter.copy_(decoded(layer, 'hc_ffn_' + suffix + '.weight'))
        w13 = torch.cat([raw(layer, 'ffn_gate_exps.weight'), raw(layer, 'ffn_up_exps.weight')], dim=1)
        w2 = raw(layer, 'ffn_down_exps.weight')
        typ1 = int(tensor(layer, 'ffn_gate_exps.weight').tensor_type)
        assert tensor(layer, 'ffn_gate_exps.weight').tensor_type == tensor(layer, 'ffn_up_exps.weight').tensor_type
        typ2 = int(tensor(layer, 'ffn_down_exps.weight').tensor_type)
        shared13 = torch.cat([raw(layer, 'ffn_gate_shexp.weight'), raw(layer, 'ffn_up_shexp.weight')], dim=0)
        shared2 = raw(layer, 'ffn_down_shexp.weight')
        assert all(tensor(layer, s).tensor_type == gguf.GGMLQuantizationType.Q8_0 for s in
                   ['ffn_gate_shexp.weight', 'ffn_up_shexp.weight', 'ffn_down_shexp.weight'])
        activation = SiluAndMul()
        for nt in a.tokens:
            prefix = f'layer{layer}-{nt}'
            i = torch.arange(nt * 10240, dtype=torch.int64, device='cuda')
            hidden = (((i * 37 + 33 + layer * 13) % 509 - 254).float() / 128).bfloat16().reshape(nt, 10240)
            projection, dispatch = 0, []
            with torch.inference_mode():
                x, residual = hc.mix(hidden)
                logits = gate(x)[0]
                shared_logits = shared_gate(x)[0]
                topw, ids, source = fused_topk(x, logits, 10, True)
                with patch.object(moe_module.ops, 'ggml_moe_a8_vec', side_effect=lambda *args, **kw: observe_projection(original_vec, 'vec', *args, **kw)), patch.object(moe_module.ops, 'ggml_moe_a8', side_effect=lambda *args, **kw: observe_projection(original_mmq, 'mmq', *args, **kw)), patch.object(moe_module, 'apply_moe_activation', side_effect=observe_activation), patch.object(moe_module.ops, 'moe_sum', side_effect=observe_sum):
                    routed = moe_module._fused_moe_gguf(x, w13, w2, topw, ids, typ1, typ2, 'silu')
                shared_projected = _fused_mul_mat_gguf(x, shared13, 8)
                shared_act = activation(shared_projected)
                shared_down = _fused_mul_mat_gguf(shared_act, shared2, 8)
                shared_sigmoid = torch.sigmoid(shared_logits)
                shared = shared_sigmoid * shared_down
                block = routed + shared
                combined = hc.combine(block, residual)
            assert projection == 2 and dispatch == (['mmq', 'mmq'] if nt > 64 else ['vec', 'vec'])
            for name, value in [('hc-hidden', hidden), ('input', x), ('logits', logits), ('shared-logits', shared_logits),
                                ('weights', topw), ('ids', ids), ('source', source), ('routed', routed),
                                ('shared-gateup', shared_projected), ('shared-activation', shared_act),
                                ('shared-down', shared_down), ('shared-sigmoid', shared_sigmoid), ('shared', shared),
                                ('block', block), ('hc-combined', combined)]:
                save(prefix + '-' + name + '.bin', value)
            high = int((ids >= 256).sum().item())
            assert torch.isfinite(combined).all()
            cases.append(dict(name=prefix, layer=layer, tokens=nt, gate_type=typ1, down_type=typ2,
                              dispatch=dispatch, high_expert_slots=high))
            print(prefix, dispatch, 'high_expert_slots', high, flush=True)
        del w13, w2, shared13, shared2, gate, shared_gate, hc
manifest = dict(profile='HC FFN + GGUF A8 routed/shared experts, expert-count, metadata-index and Q5_1 partial-tile corrections; unchanged quantizer/dot/activation/reduction formulas',
                inventory=str(a.inventory), weights=weights, files=files, cases=cases)
assert any(c['high_expert_slots'] > 0 for c in cases)
target = a.directory / a.manifest_name
data = json.dumps(manifest, indent=2)
if target.exists():
    assert target.read_text() == data
else:
    target.write_text(data)
print('FLASH_MOE_CAPTURED', len(cases), flush=True)
