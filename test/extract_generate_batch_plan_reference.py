"""Serialize captured scheduler metadata without running the native planner.

Usage: python3 test/extract_generate_batch_plan_reference.py REFERENCE.json OUTPUT.h
Expected values come only from the independent original-runtime capture.
"""
import hashlib
import json
from pathlib import Path
import sys

source, output = map(Path, sys.argv[1:])
r = json.loads(source.read_text())
assert r['schema'] == 4
header = ['/* Captured independent scheduler metadata. Source SHA256: '
          + hashlib.sha256(source.read_bytes()).hexdigest() + ' */',
          'typedef struct {int count,id[8],tokens[8],position[8],role[8];} BnGenerateBatchTrace;',
          'static const BnGenerateBatchTrace generate_batch_trace[] = {']
assert len(r['full_batch_logits']) == len(r['execution_roles']) == 134
for m, roles in zip(r['full_batch_logits'], r['execution_roles']):
    n = m['shape'][0]
    assert 1 <= n <= 8
    fields = [m['request_ids'], m['scheduled'], m['computed'], roles]
    assert all(len(x) == n for x in fields)
    header.append('{' + ','.join([str(n)] + ['{' + ','.join(map(str, x)) + '}'
                                          for x in fields]) + '},')
header.append('};')
output.write_text('\n'.join(header) + '\n')
