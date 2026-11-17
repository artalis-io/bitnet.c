"""Synthetic accounting logs must not masquerade as numerical qualification."""
import copy
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
records = [
 dict(kind='arrival_mode', mode='wall_clock', reference_checked=False, outputs_valid=True),
 dict(kind='summary', correct=None, requests=2, outputs_each=2, steps=6, peak_live=1,
      elapsed_ms=80, tokens_per_second=50, prefill_calls=4, mixed_calls=0, decode_calls=2),
 dict(kind='kv_pool', scope='kv_payload_only', kv_bytes_per_slot=1048576, created=1,
      allocated_slots=1, reused=1, discarded=0, leased_slots=0,
      reserved_kv_bytes=1048576, peak_kv_bytes=1048576, budget_bytes=1048576)]
for call in range(6):
    i, position = divmod(call, 3)
    records.append(dict(kind='step', call=call, duration_ms=5,
                        rows=[[i, 1, position, int(position == 2)]]))
for i in range(2):
    records.append(dict(kind='arrival', id=i, offered_ms=5+5*i,
                        worker_enqueue_ms=6+9*i, enqueue_step=i))
    records.append(dict(kind='request', id=i, arrival_step=i, submitted_ms=5+5*i,
                        admitted_ms=7+28*i, delivery_ms=[20+40*i, 30+40*i], tokens=[i, i]))
with tempfile.TemporaryDirectory() as temporary:
    p = Path(temporary)
    work = p / 'clock.bin'
    work.write_bytes(b'BGA2' + struct.pack('<8i', 2, 2, 1, 1, 1, 1, 16, 1)
                     + struct.pack('<4i', 5, 2, 0, 0) + struct.pack('<4i', 10, 2, 1, 1))
    log = p / 'run.jsonl'
    def verify(rows, options=()):
        log.write_text('\n'.join(json.dumps(x) for x in rows) + '\n')
        return subprocess.run([sys.executable, str(root/'bench/verify_bf16_wall_clock.py'),
                               str(log), '--workload', str(work), *options], capture_output=True, text=True)
    good = verify(records)
    assert good.returncode == 0, good.stderr
    result = json.loads(good.stdout)
    assert result['clock_accounting_valid'] and not result['independent_compute_reference_matches']
    declared = [dict(kind='prefill_policy', policy='fixed'), *records]
    assert verify(declared).returncode == 0
    assert verify([declared[0], *declared]).returncode != 0
    assert verify([dict(kind='prefill_policy', policy='unknown'), *records]).returncode != 0
    assert verify([dict(kind='prefill_policy', policy='reclaim'), *records]).returncode != 0
    assert verify(records, ['--prefill-policy', 'reclaim']).returncode != 0
    legacy = work.read_bytes()
    work.write_bytes(b'BGA3' + legacy[4:36] + struct.pack('<i', 1) + legacy[36:])
    assert verify(records).returncode == 0
    work.write_bytes(b'BGA3' + legacy[4:36] + struct.pack('<i', 2) + legacy[36:])
    assert verify(records).returncode != 0  # cap exceeds the full budget
    work.write_bytes(legacy)
    for change in ['shifted_submission', 'early_enqueue', 'prefill_role', 'token', 'kv', 'self_certified']:
        bad = copy.deepcopy(records)
        if change == 'shifted_submission':
            next(x for x in bad if x['kind']=='request')['submitted_ms'] = 6
        elif change == 'early_enqueue':
            next(x for x in bad if x['kind']=='arrival')['worker_enqueue_ms'] = 4
        elif change == 'prefill_role':
            next(x for x in bad if x['kind']=='step')['rows'][0][3] = 1
        elif change == 'token':
            next(x for x in bad if x['kind']=='request')['tokens'][0] = -1
        elif change == 'kv':
            next(x for x in bad if x['kind']=='kv_pool')['reserved_kv_bytes'] -= 1
        else:
            next(x for x in bad if x['kind']=='summary')['correct'] = True
        assert verify(bad).returncode != 0, change
    # Independent two-lane reclaimed schedule: clipped two-token prompt leaves
    # six tokens for the long prompt, rather than a fixed quota of four.
    work.write_bytes(b'BGA2' + struct.pack('<8i', 2, 1, 2, 2, 8, 8, 16, 2)
                     + struct.pack('<4i', 0, 2, 0, 0)
                     + struct.pack('<2i', 0, 10) + struct.pack('<10i', *([1]*10)))
    reclaimed = [dict(kind='prefill_policy', policy='reclaim'),
                 dict(kind='arrival_mode', mode='wall_clock', reference_checked=False, outputs_valid=True),
                 dict(kind='summary', correct=None, requests=2, outputs_each=1, steps=2, peak_live=2,
                      elapsed_ms=30, tokens_per_second=2000/30, prefill_calls=2, mixed_calls=0, decode_calls=0),
                 dict(kind='kv_pool', scope='kv_payload_only', kv_bytes_per_slot=1048576, created=2,
                      allocated_slots=2, reused=0, discarded=0, leased_slots=0,
                      reserved_kv_bytes=2097152, peak_kv_bytes=2097152, budget_bytes=2097152),
                 dict(kind='step', call=0, duration_ms=5, rows=[[0,2,0,0],[1,6,0,0]]),
                 dict(kind='step', call=1, duration_ms=5, rows=[[1,4,6,0]])]
    for i in range(2):
        reclaimed += [dict(kind='arrival', id=i, offered_ms=0, worker_enqueue_ms=1, enqueue_step=0),
                      dict(kind='request', id=i, arrival_step=0, submitted_ms=0, admitted_ms=2,
                           delivery_ms=[10+10*i], tokens=[i])]
    opts = ['--prefill-policy', 'reclaim']
    good = verify(reclaimed, opts)
    assert good.returncode == 0, good.stderr
    assert verify(reclaimed).returncode != 0
    assert verify(reclaimed[1:], opts).returncode != 0
    for n in [4, 5, 7]:
        bad = copy.deepcopy(reclaimed)
        next(x for x in bad if x['kind']=='step')['rows'][1][1] = n
        assert verify(bad, opts).returncode != 0
print('BF16 wall-clock verifier fixed/reclaimed accounting, rejection and qualification scope PASSED')
