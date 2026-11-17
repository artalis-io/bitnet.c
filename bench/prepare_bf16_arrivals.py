"""Prepare BGA1/BGA4 controlled-step or BGA2/BGA3 wall-clock workloads.

The payload contains arrivals, prompts and greedy output expectations only.
Captured query lengths, positions and execution roles are never runtime inputs.
Wall-clock workloads omit output expectations: their numerical correctness
must be qualified independently for their actual execution geometry.
Schema 1 workload specifications contain prompts, outputs_each and offered
arrival deadlines without reference continuations. Schema 4 references retain
the controlled-step protocol and existing clock-workload compatibility.
"""
import argparse
import json
from pathlib import Path
import struct

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('reference', type=Path)
ap.add_argument('output', type=Path)
ap.add_argument('--arrivals-ms', type=int, nargs='+',
                help='one fixed wall-clock arrival offset per request')
ap.add_argument('--mixed-budget', type=int,
                help='optional token cap for steps with active decodes')
a = ap.parse_args()
r = json.loads(a.reference.read_text())
assert r['schema'] in (1, 4)
streams = r['streams']
engine = r['engine_request']
if r['schema'] == 1:
    outputs = r['outputs_each']
    assert all('tokens' not in stream for stream in streams)
    if a.arrivals_ms is None:
        a.arrivals_ms = r['offered_arrival_ms']
    mixed_budget = r.get('mixed_token_budget', 0) if a.mixed_budget is None else a.mixed_budget
else:
    outputs = len(streams[0]['tokens'])
    mixed_budget = 0 if a.mixed_budget is None else a.mixed_budget
assert 1 <= len(streams) <= 64 and 1 <= outputs <= 256
fields = [len(streams), outputs, r['max_live'], engine['max_parallel_prefills'],
          engine['long_prefill_token_threshold'], r['token_budget'], r['capacity'],
          engine['kv_cache_memory_bytes'] // 1048576]
if a.arrivals_ms is not None:
    assert len(a.arrivals_ms) == len(streams)
    assert all(0 <= t <= 3600000 for t in a.arrivals_ms)
assert mixed_budget == 0 or r['max_live'] <= mixed_budget <= r['token_budget']
if mixed_budget:
    fields.append(mixed_budget)
magic = (b'BGA3' if mixed_budget else b'BGA2') if a.arrivals_ms is not None else (b'BGA4' if mixed_budget else b'BGA1')
data = bytearray(magic)

def integer(n):
    assert 0 <= n <= 2147483647
    data.extend(struct.pack('<i', n))

for n in fields:
    integer(n)
last = -1
for i, stream in enumerate(streams):
    arrival = stream['arrival_step'] if a.arrivals_ms is None else a.arrivals_ms[i]
    assert arrival >= last
    last = arrival
    assert r['schema'] == 1 or len(stream['tokens']) == outputs
    integer(last)
    integer(len(stream['input_ids']))
    for n in stream['input_ids'] + (stream['tokens'] if a.arrivals_ms is None else []):
        integer(n)
a.output.write_bytes(data)
