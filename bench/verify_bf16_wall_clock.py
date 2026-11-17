"""Validate fixed offered deadlines separately from independent numerical proof."""
import argparse
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
from verify_bf16_capacity_pool import validate_capacity_pool

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('log', type=Path)
ap.add_argument('--workload', type=Path, required=True)
ap.add_argument('--reference', type=Path)
ap.add_argument('--raw', type=Path)
ap.add_argument('--prefill-policy', choices=['fixed', 'reclaim'], default='fixed')
a = ap.parse_args()
assert not a.raw or a.reference, 'raw comparison requires an independent reference'
data = a.workload.read_bytes()
assert data[:4] in (b'BGA2', b'BGA3')
nr, outputs, limit, prefills, chunk, budget, capacity, kv_mib = struct.unpack_from('<8i', data, 4)
assert 1 <= nr <= 64 and 1 <= limit <= 16 and 1 <= prefills <= limit
assert 1 <= chunk <= capacity <= 8192 and 1 <= budget <= 8192
mixed_budget = struct.unpack_from('<i', data, 36)[0] if data[:4] == b'BGA3' else 0
assert mixed_budget == 0 or limit <= mixed_budget <= budget
offset = 40 if data[:4] == b'BGA3' else 36
prompts, offered = [], []
for i in range(nr):
    deadline, n = struct.unpack_from('<2i', data, offset)
    offset += 8
    assert 0 <= deadline <= 3600000 and (not offered or deadline >= offered[-1])
    assert 1 <= n <= capacity and outputs - 1 <= capacity - n
    offered.append(deadline)
    prompts.append(list(struct.unpack_from(f'<{n}i', data, offset)))
    offset += n * 4
assert offset == len(data)
records = [json.loads(x) for x in a.log.read_text().splitlines() if x.startswith('{')]
policies = [x for x in records if x['kind'] == 'prefill_policy']
assert len(policies) <= 1
if policies:
    assert policies[0] == dict(kind='prefill_policy', policy=a.prefill_policy)
else:
    assert a.prefill_policy == 'fixed', 'reclaimed quotas require explicit log metadata'
mode, = [x for x in records if x['kind'] == 'arrival_mode']
assert mode == dict(kind='arrival_mode', mode='wall_clock', reference_checked=False, outputs_valid=True)
summary, = [x for x in records if x['kind'] == 'summary']
assert summary['correct'] is None
assert summary['requests'] == nr and summary['outputs_each'] == outputs
assert math.isfinite(summary['elapsed_ms']) and summary['elapsed_ms'] > offered[-1]
assert math.isclose(summary['tokens_per_second'], 1000 * nr * outputs / summary['elapsed_ms'], rel_tol=1e-7)
arrivals = [x for x in records if x['kind'] == 'arrival']
requests = [x for x in records if x['kind'] == 'request']
steps = [x for x in records if x['kind'] == 'step']
assert len(arrivals) == len(requests) == nr and len(steps) == summary['steps']
assert [x['id'] for x in arrivals] == [x['id'] for x in requests] == list(range(nr))
for i, (arrival, request) in enumerate(zip(arrivals, requests)):
    assert arrival['offered_ms'] == offered[i]
    assert abs(request['submitted_ms'] - offered[i]) <= .000002
    assert request['submitted_ms'] <= arrival['worker_enqueue_ms'] + .000002
    assert arrival['worker_enqueue_ms'] <= request['admitted_ms'] + .000002
    assert request['arrival_step'] == arrival['enqueue_step']
    delivery = request['delivery_ms']
    assert len(delivery) == len(request['tokens']) == outputs
    assert all(math.isfinite(t) for t in delivery)
    assert request['admitted_ms'] <= delivery[0] <= delivery[-1] <= summary['elapsed_ms']
    assert all(b > a for a, b in zip(delivery, delivery[1:]))
    assert all(isinstance(t, int) and t >= 0 for t in request['tokens'])
positions, generated = [0] * nr, [0] * nr
phases = dict(prefill_calls=0, mixed_calls=0, decode_calls=0)
for call, step in enumerate(steps):
    assert step['call'] == call and math.isfinite(step['duration_ms']) and step['duration_ms'] > 0
    rows = step['rows'];assert 1 <= len(rows) <= limit
    assert len({x[0] for x in rows}) == len(rows)
    assert sum(x[1] for x in rows) <= budget
    pending_count = sum(position < len(prompts[i]) for i, n, position, role in rows)
    assert pending_count <= prefills
    step_budget = mixed_budget if mixed_budget and pending_count < len(rows) else budget
    assert sum(x[1] for x in rows) <= step_budget
    share = (step_budget - (len(rows) - pending_count)) // pending_count if pending_count else 0
    if pending_count and a.prefill_policy == 'reclaim':
        caps = [len(prompts[i]) - position for i, n, position, role in rows
                if position < len(prompts[i])]
        available = step_budget - (len(rows) - pending_count)
        share = max(q for q in range(1, chunk + 1)
                    if sum(min(cap, q) for cap in caps) <= available)
    tail = False;publication = []
    for i, n, position, role in rows:
        assert 0 <= i < nr and call >= arrivals[i]['enqueue_step']
        assert position == positions[i] and n > 0 and position + n <= capacity
        pending = position < len(prompts[i])
        if pending:
            assert n == min(chunk, share, len(prompts[i]) - position)
            tail = True
        else:
            assert n == 1
        assert role == (0 if tail else 1)
        positions[i] += n
        if not pending or positions[i] == len(prompts[i]):
            assert generated[i] < outputs
            publication.append(requests[i]['delivery_ms'][generated[i]])
            generated[i] += 1
    assert len(set(publication)) <= 1, 'cohort token publication timestamp changed'
    flags = [x[3] for x in rows]
    phases['decode_calls' if all(flags) else 'mixed_calls' if any(flags) else 'prefill_calls'] += 1
assert all(n == outputs for n in generated)
assert positions == [len(p) + outputs - 1 for p in prompts]
assert all(summary[k] == v for k, v in phases.items())
events = [(r['admitted_ms'], 1) for r in requests] + [(r['delivery_ms'][-1], -1) for r in requests]
live = peak = 0
for _, delta in sorted(events):
    live += delta
    assert 0 <= live <= limit
    peak = max(peak, live)
assert live == 0 and peak == summary['peak_live']
pool, = [x for x in records if x['kind'] == 'kv_pool']
per = capacity * 16 * 2 * 1024 * 2
assert pool['scope'] == 'kv_payload_only' and pool['kv_bytes_per_slot'] == per
if any(x['kind'] == 'kv_admission' for x in records):
    validate_capacity_pool(records, list(map(len, prompts)), outputs, limit, capacity, chunk, kv_mib * 1048576)
else:
    assert pool['created'] == pool['allocated_slots'] == peak
    assert pool['created'] + pool['reused'] == nr
    assert pool['discarded'] == pool['leased_slots'] == 0
    assert pool['reserved_kv_bytes'] == pool['peak_kv_bytes'] == per * peak <= pool['budget_bytes'] == kv_mib * 1048576
qualified = False
if a.reference:
    ref = json.loads(a.reference.read_text())
    assert ref['schema'] == 4 and len(ref['streams']) == nr
    assert ref['capacity'] == capacity and ref['token_budget'] == budget and ref['max_live'] == limit
    assert [s['input_ids'] for s in ref['streams']] == prompts
    assert [s['tokens'] for s in ref['streams']] == [r['tokens'] for r in requests]
    assert len(ref['full_batch_logits']) == len(steps)
    for step, matrix, roles in zip(steps, ref['full_batch_logits'], ref['execution_roles']):
        expected = [list(x) for x in zip(matrix['request_ids'], matrix['scheduled'], matrix['computed'], roles)]
        assert step['rows'] == expected
    if a.raw:
        subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / 'test/verify_bf16_mixed_reference.py'),
                        str(a.raw), '--reference', str(a.reference)], check=True)
    qualified = True
print(json.dumps(dict(clock_accounting_valid=True, independent_compute_reference_matches=qualified,
                     requests=nr, steps=len(steps), peak_live=peak,
                     max_worker_enqueue_delay_ms=max(x['worker_enqueue_ms'] - x['offered_ms'] for x in arrivals),
                     max_total_queue_ms=max(x['admitted_ms'] - offered[i] for i, x in enumerate(requests)))))
