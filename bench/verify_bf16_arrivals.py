"""Verify controlled-step arrival geometry, outputs, timestamps and KV charges."""
import argparse
import json
import math
from pathlib import Path
from verify_bf16_capacity_pool import validate_capacity_pool

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('log', type=Path)
ap.add_argument('--reference', type=Path, required=True)
a = ap.parse_args()
r = json.loads(a.reference.read_text())
records = [json.loads(x) for x in a.log.read_text().splitlines() if x.startswith('{')]
summary, = [x for x in records if x['kind'] == 'summary']
assert summary['correct'] is True
streams = r['streams']
matrices = r['full_batch_logits']
assert summary['requests'] == len(streams)
assert summary['outputs_each'] == len(streams[0]['tokens'])
assert summary['steps'] == len(matrices)
assert type(summary['peak_live']) is int and 1 <= summary['peak_live'] <= r['max_live']
steps = [x for x in records if x['kind'] == 'step']
assert len(steps) == len(matrices)
phases = dict(prefill_calls=0, mixed_calls=0, decode_calls=0)
for i, (step, matrix, roles) in enumerate(zip(steps, matrices, r['execution_roles'])):
    expected = [list(x) for x in zip(matrix['request_ids'], matrix['scheduled'], matrix['computed'], roles)]
    assert step['call'] == i and step['rows'] == expected, ('geometry', i)
    assert math.isfinite(step['duration_ms']) and step['duration_ms'] > 0
    phases['decode_calls' if all(roles) else 'mixed_calls' if any(roles) else 'prefill_calls'] += 1
assert all(summary[k] == v for k, v in phases.items())
requests = [x for x in records if x['kind'] == 'request']
assert len(requests) == len(streams) and [x['id'] for x in requests] == list(range(len(streams)))
arrivals = []
for request, stream in zip(requests, streams):
    assert request['arrival_step'] == stream['arrival_step']
    assert request['tokens'] == stream['tokens']
    delivered = request['delivery_ms']
    assert len(delivered) == len(stream['tokens'])
    assert all(math.isfinite(t) for t in delivered)
    assert 0 <= request['submitted_ms'] <= request['admitted_ms'] <= delivered[0]
    assert all(b > a for a, b in zip(delivered, delivered[1:]))
    assert delivered[-1] <= summary['elapsed_ms']
    arrivals.extend([(request['admitted_ms'], 1), (delivered[-1], -1)])
# At equal timestamps completions precede any following admission.
live = peak = 0
for _, delta in sorted(arrivals):
    live += delta
    assert 0 <= live <= r['max_live']
    peak = max(peak, live)
assert live == 0 and peak == summary['peak_live']
pool, = [x for x in records if x['kind'] == 'kv_pool']
per = r['capacity'] * 16 * 2 * 1024 * 2
assert pool['scope'] == 'kv_payload_only' and pool['kv_bytes_per_slot'] == per
if any(x['kind'] == 'kv_admission' for x in records):
    validate_capacity_pool(records, [len(s['input_ids']) for s in streams], len(streams[0]['tokens']),
                          r['max_live'], r['capacity'], r['engine_request']['long_prefill_token_threshold'],
                          r['engine_request']['kv_cache_memory_bytes'])
else:
    assert pool['created'] == pool['allocated_slots'] == peak
    assert pool['reused'] == len(streams) - pool['created']
    assert pool['discarded'] == pool['leased_slots'] == 0
    assert pool['reserved_kv_bytes'] == pool['peak_kv_bytes'] == per * pool['allocated_slots']
    assert pool['peak_kv_bytes'] <= pool['budget_bytes'] == r['engine_request']['kv_cache_memory_bytes']
assert math.isfinite(summary['elapsed_ms']) and summary['elapsed_ms'] > 0
rate = 1000 * sum(len(x['tokens']) for x in streams) / summary['elapsed_ms']
assert math.isclose(rate, summary['tokens_per_second'], rel_tol=1e-7)
print(json.dumps(dict(correct=True, steps=len(steps), sampled_heads=sum(len(x['tokens']) for x in streams),
                     peak_live=peak, created=pool['created'], reused=pool['reused'],
                     reserved_kv_bytes=pool['reserved_kv_bytes'], diagnostic=summary['diagnostic'])))
