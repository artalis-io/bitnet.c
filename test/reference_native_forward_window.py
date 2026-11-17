"""Actual source prefill across the 128-token SWA and 256-token page edges.

Uses preserved original grouped MoE and whole-model source components. The
output directory argv[1] is immutable. No candidate code computes expectations.
"""
from pathlib import Path
import hashlib

window_parent = Path(__file__).with_name('reference_native_forward_grouped.py')
assert hashlib.sha256(window_parent.read_bytes()).hexdigest() == 'bf51ba777bfccec8d6864656a4e802da1d5bfb120cb37ab17435be0f68a8aef5'
window_code = window_parent.read_text()

window_old_specs = "new_specs = \"specs=[('prefill9',9,0,True),('greedy-small',1,9,False),('prefill33',33,0,True),('chunk17',17,33,False)]+[(f'greedy-{i}',1,50+i,False)for i in range(5)]\""
window_new_specs = "new_specs = \"specs=[('prefill129',129,0,True),('chunk129',129,129,False)]+[(f'greedy-{i}',1,258+i,False)for i in range(5)]\""
assert window_code.count(window_old_specs) == 1
window_code = window_code.replace(window_old_specs, window_new_specs)
window_old_count = "assert len(group_cases) == 120 and {c['rows'] for c in group_cases} == {9, 17, 33}"
assert window_code.count(window_old_count) == 1
window_code = window_code.replace(window_old_count, "assert len(group_cases) == 80 and {c['rows'] for c in group_cases} == {129}")
window_scope = 'Actual 40-layer prefill129, resumed129 and five source self-fed greedy steps; context263; sliding128 and compressed page boundary; no heavy reconstruction or long indexed selection'
for window_old_scope in [
    'Actual 40-layer grouped/fused prefill9/33, resumed17 and six autoregressive greedy steps; context55; no heavy reconstruction or long indexed selection',
    'Actual 40-layer grouped/fused prefill9/33, resumed17 and six source self-fed greedy steps; context55; no heavy reconstruction or long indexed selection',
]:
    assert window_code.count(window_old_scope) == 1
    window_code = window_code.replace(window_old_scope, window_scope)

# The original source's 768-row rings and 1024-token compressed-pool capacity
# suffice for context263. Dense pooled attention remains below top512; this
# fixture makes no claim about long index selection or physical ring wrap.
exec(compile(window_code, str(window_parent), 'exec'))
emit('window-provenance.json', (json.dumps(dict(
    scope=window_scope, grouped_parent_sha256=sha(window_parent.read_bytes()),
    producer_sha256=sha(Path(__file__).read_bytes()), context=263,
    source_swa_window=128, source_ring_rows=768, source_pool_tokens=1024,
    source_index_selection=False, grouped_blocks=len(group_cases),
    expert_tiles=sorted({t['tile_m'] for c in group_cases for t in c['tiers']}),
), indent=2)+'\n').encode())
