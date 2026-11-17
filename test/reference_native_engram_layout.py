"""Emit immutable layout goldens from pinned Engram capture and NumPy 2.3.5."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('reference', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
assert np.__version__ == '2.3.5'
manifest_raw = (a.reference / 'manifest.json').read_bytes()
assert hashlib.sha256(manifest_raw).hexdigest() == '92875f3309c982a04a8d7bbea0221f729454af3c4be387019e610e97df8778ae'
raw = (a.reference / 'layout.json').read_bytes()
assert hashlib.sha256(raw).hexdigest() == json.loads(manifest_raw)['layout.json']['sha256']
layout = json.loads(raw)
lines = ['/* Independent NumPy 2.3.5 and original Engram source layout goldens.',
         ' * Original layout SHA256: ' + hashlib.sha256(raw).hexdigest() + ' */']
for field in ['sizes', 'offsets', 'multipliers']:
    values = [layout[str(l)][field] for l in [1, 14]]
    lines.append('static const uint64_t native_hash_ref_' + field + '[2][' + str(len(values[0])) + '] = {')
    lines.extend('    {' + ','.join('UINT64_C(' + str(n) + ')' for n in row) + '},' for row in values)
    lines.append('};')
lines.append('static const uint64_t native_hash_ref_first15_rows = UINT64_C(' + str(sum(layout['1']['sizes'][:15])) + ');')
lines.append('static const int native_hash_ref_vocab[2] = {99092,2147483647};')
lines.append('static const uint64_t native_hash_ref_draws[2][128][16] = {')
for vocab in [99092, 2147483647]:
    lines.append('  {')
    for layer in range(128):
        vals = np.random.default_rng(10007 * layer).integers(0, (2**63-1)//vocab//2, 16, dtype=np.int64)
        lines.append('    {' + ','.join('UINT64_C(' + str(2*int(n)+1) + ')' for n in vals) + '},')
    lines.append('  },')
lines.append('};')
result = ('\n'.join(lines) + '\n').encode()
if a.output.exists():
    assert a.output.read_bytes() == result, 'refusing to replace differing reference'
else:
    a.output.write_bytes(result)
print(json.dumps({'bytes': len(result), 'sha256': hashlib.sha256(result).hexdigest(), 'numpy': np.__version__}))
