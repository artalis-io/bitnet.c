"""Whole-model source oracle with reconstructed heavy experts and greedy decode.

Original group B/M, zero padding, folded projections and fixed-K gather are
preserved. Output directory argv[1] is immutable; no engine code supplies gold.
"""
from pathlib import Path
import hashlib

heavy_parent = Path(__file__).with_name('reference_native_forward_grouped.py')
assert hashlib.sha256(heavy_parent.read_bytes()).hexdigest() == 'bf51ba777bfccec8d6864656a4e802da1d5bfb120cb37ab17435be0f68a8aef5'
heavy_code = heavy_parent.read_text()

# Reuse the unchanged original whole-layer projection function, not a candidate
# decoder/GEMM implementation. The parent verifies this capture's digest and
# the original compiled helper/library and upstream source digests before use.
heavy_setup = '''small_ffn = ffn
heavy_tree = gn['ast'].parse(group_source.read_text())
heavy_projection = [node for node in heavy_tree.body if isinstance(node, gn['ast'].FunctionDef) and node.name == 'projection']
assert len(heavy_projection) == 1
exec(compile(gn['ast'].Module(body=heavy_projection, type_ignores=[]), str(group_source), 'exec'), gn)
'''
assert heavy_code.count('small_ffn = ffn\n') == 1
heavy_code = heavy_code.replace('small_ffn = ffn\n', heavy_setup)

heavy_old = "    assert not groups, 'extend original grouped reconstruction before using >256 rows/expert'"
heavy_body = '''    heavy_kinds = []
    gn['experts'] = experts
    for group in groups:
        ee, M = group['experts'], group['rows']
        assert len(ee) <= gn['batch_cap'] and M > 256
        slab = torch.zeros((len(ee), M, D), dtype=torch.float16, device='cuda')
        for b, e in enumerate(ee):
            start = sum(cnt[:e])
            slab[b, :cnt[e]] = x[tokens[start:start+cnt[e]]]
        # Explicit source FP32-accumulation policy, matching f16acc_enabled=0.
        up, uk = gn['projection'](ee, 1, slab, M, 0, 0)
        gate, gk = gn['projection'](ee, 0, slab, M, 0, 0)
        activated = torch.empty_like(up)
        assert not libs['fold'].ref_act(gate.data_ptr(), up.data_ptr(), activated.data_ptr(), activated.numel(), 10., stream)
        down, dk = gn['projection'](ee, 2, activated, M, 1, 0)
        scratch[group['slot']:group['slot']+len(ee)*M].copy_(down.reshape(-1, D))
        heavy_kinds.append([gk, uk, dk])
    del gn['experts']'''
assert heavy_code.count(heavy_old) == 1
heavy_code = heavy_code.replace(heavy_old, heavy_body)
heavy_old_record = 'tiers=tiers, heavy_groups=groups))'
assert heavy_code.count(heavy_old_record) == 1
heavy_code = heavy_code.replace(heavy_old_record, 'tiers=tiers, heavy_groups=groups, group_kinds=heavy_kinds))')

heavy_old_specs = "new_specs = \"specs=[('prefill9',9,0,True),('greedy-small',1,9,False),('prefill33',33,0,True),('chunk17',17,33,False)]+[(f'greedy-{i}',1,50+i,False)for i in range(5)]\""
heavy_new_specs = "new_specs = \"specs=[('prefill385',385,0,True)]+[(f'greedy-{i}',1,385+i,False)for i in range(5)]\""
assert heavy_code.count(heavy_old_specs) == 1
heavy_code = heavy_code.replace(heavy_old_specs, heavy_new_specs)
heavy_old_count = "assert len(group_cases) == 120 and {c['rows'] for c in group_cases} == {9, 17, 33}"
assert heavy_code.count(heavy_old_count) == 1
heavy_code = heavy_code.replace(heavy_old_count, "assert len(group_cases) == 40 and {c['rows'] for c in group_cases} == {385}")
heavy_scope = 'Actual 40-layer prefill385 with original heavy expert reconstruction and five self-fed greedy steps; context390; f16acc0; no long indexed selection or physical SWA ring wrap'
for heavy_old_scope in [
    'Actual 40-layer grouped/fused prefill9/33, resumed17 and six autoregressive greedy steps; context55; no heavy reconstruction or long indexed selection',
    'Actual 40-layer grouped/fused prefill9/33, resumed17 and six source self-fed greedy steps; context55; no heavy reconstruction or long indexed selection',
]:
    assert heavy_code.count(heavy_old_scope) == 1
    heavy_code = heavy_code.replace(heavy_old_scope, heavy_scope)

# Context390 fits the original 768-row SWA rings and 1024-token pool capacity;
# pooled lengths remain below top512, so original dense bypass remains valid.
exec(compile(heavy_code, str(heavy_parent), 'exec'))
heavy_cases = [case for case in group_cases if case['heavy_groups']]
assert heavy_cases, 'fixture must execute actual reconstructed heavy experts'
emit('heavy-provenance.json', (json.dumps(dict(
    scope=heavy_scope, grouped_parent_sha256=sha(heavy_parent.read_bytes()),
    producer_sha256=sha(Path(__file__).read_bytes()), context=390,
    source_projection='Unmodified projection AST from pinned original layer capture',
    f16acc_enabled=0, batch_cap=gn['batch_cap'], heavy_layers=len(heavy_cases),
    heavy_groups=sum(len(case['heavy_groups']) for case in heavy_cases),
    max_batch=max(len(group['experts']) for case in heavy_cases for group in case['heavy_groups']),
    max_rows=max(group['rows'] for case in heavy_cases for group in case['heavy_groups']),
    source_swa_window=128, source_ring_rows=768, source_pool_tokens=1024,
    source_index_selection=False,
), indent=2)+'\n').encode())
