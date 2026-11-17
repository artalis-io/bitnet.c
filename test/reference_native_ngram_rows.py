"""Extract small immutable C goldens from the original Engram hash capture."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('reference', type=Path)
parser.add_argument('output', type=Path)
a = parser.parse_args()
raw = (a.reference / 'manifest.json').read_bytes()
assert hashlib.sha256(raw).hexdigest() == '92875f3309c982a04a8d7bbea0221f729454af3c4be387019e610e97df8778ae'
manifest = json.loads(raw)
def read(name):
    data = (a.reference / name).read_bytes()
    assert len(data) == manifest[name]['bytes'] and hashlib.sha256(data).hexdigest() == manifest[name]['sha256']
    return data
cases = json.loads(read('cases.json'))
selected = ['hash-1-2-7-pad-boundaries', 'hash-14-2-7-pad-boundaries', 'hash-1-2-4-text', 'hash-14-1-3-repeated']
lines = ['/* Unmodified original source hash outputs; see reference_native_ngram_rows.py. */']
for i, prefix in enumerate(selected):
    for field, kind in [('history', 'uint32_t'), ('rows', 'uint64_t')]:
        data = read(prefix + '-' + field + '.i64le')
        values = struct.unpack('<' + 'q' * (len(data)//8), data)
        assert all(0 <= n <= (2**32-1 if field == 'history' else 2**63-1) for n in values)
        lines.append('static const ' + kind + ' native_ngram_ref_' + field + '_' + str(i) + '[] = {')
        lines.append('    ' + ','.join(('UINT32_C(' if field == 'history' else 'UINT64_C(') + str(n) + ')' for n in values))
        lines.append('};')
lines.append('typedef struct {int layer;size_t batch,length;const uint32_t *history;const uint64_t *rows;} NativeNGramReference;')
lines.append('static const NativeNGramReference native_ngram_reference[] = {')
for i, prefix in enumerate(selected):
    case = next(c for c in cases if c['prefix'] == prefix)
    lines.append('    {' + ','.join(map(str,[case['layer'],case['batch'],case['length']])) + ',native_ngram_ref_history_' + str(i) + ',native_ngram_ref_rows_' + str(i) + '},')
lines.append('};')
result = ('\n'.join(lines) + '\n').encode()
if a.output.exists():
    assert a.output.read_bytes() == result, 'refusing to replace differing reference'
else:
    a.output.write_bytes(result)
print(json.dumps({'cases':len(selected),'bytes':len(result),'sha256':hashlib.sha256(result).hexdigest()}))
