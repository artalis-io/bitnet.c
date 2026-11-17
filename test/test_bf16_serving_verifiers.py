#!/usr/bin/env python3
"""Model-free serving evidence checks: missing data, admission bounds and I/O."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory() as directory:
        p = Path(directory)
        streams = [dict(input_ids=[1, 2, 3], tokens=[(i + j) % 3 for j in range(64)], logit_sha256=[]) for i in range(3)]
        raw = {}
        for request, stream in enumerate(streams):
            for step in range(64):
                data = struct.pack('<3H', 0x3f80 + request, 0x4000 + step, 0x4040)
                raw[request, step] = data
                (p / f'request-{request}-logits-{step}.bin').write_bytes(data)
                stream['logit_sha256'].append(hashlib.sha256(data).hexdigest())
        waves, matrices = [[0, 1], [2]], []
        for wave, ids in enumerate(waves):
            for step in range(64):
                matrices.append(dict(step=wave * 64 + step, output_step=step, row_request_ids=ids,
                                     shape=[len(ids), 3], dtype='bfloat16',
                                     sha256=hashlib.sha256(b''.join(raw[i, step] for i in ids)).hexdigest()))
        ref = dict(schema=3, streams=streams, max_live=2, admission_waves=waves,
                   prefill_chunk_tokens=3, engine_request=dict(max_model_len=8192, max_num_seqs=2),
                   norm_policy={}, full_batch_logits=matrices)
        rows = [dict(kind='bf16_policy', admitted={}),
                dict(kind='config', requests=3, max_live=2, peak_live=2, prompt_tokens=3, output_tokens=64,
                     prefill_chunk=3, maxseq=8192, reserved_context_slots_peak=16384, packed_token_budget=6,
                     packed_prefill=True, matches_reference=True, reference_kind='external_tokens',
                     all_arrive_at_zero=True, serialized_execution=False, mixed_lengths=False,
                     diagnostic_logits_dump=False)]
        for request, stream in enumerate(streams):
            start, admitted = (2, request) if request < 2 else (67, 66)
            delivery = [start + i for i in range(64)]
            rows.append(dict(kind='request', id=request, prompt_tokens=3, tokens=stream['tokens'],
                             admitted_ms=admitted, ttft_ms=delivery[0], completed_ms=delivery[-1], delivery_ms=delivery))
        rows += [dict(kind='summary', elapsed_ms=131, prefill_phase_ms=5, decode_phase_ms=126,
                      output_tokens_per_second=192000 / 131, decode_calls=189, prefill_calls=2, prefill_tokens=9),
                 dict(kind='batch', prefill_batches=2, decode_batches=126)]
        reference, log = p / 'reference.json', p / 'log.jsonl'
        checks = 0

        def check(script, good, candidate=None, records=None, extra=()):
            nonlocal checks
            reference.write_text(json.dumps(ref if candidate is None else candidate))
            log.write_text('\n'.join(json.dumps(r) for r in (rows if records is None else records)))
            target = p if script.startswith('test/') else log
            result = subprocess.run([sys.executable, str(ROOT / script), str(target), '--reference', str(reference), *extra], capture_output=True, text=True)
            assert (result.returncode == 0) == good, (script, result.stdout, result.stderr)
            checks += 1
            return result.stdout

        raw_script, timing_script = 'test/verify_bf16_serving_reference.py', 'bench/verify_bf16_serving_batch.py'
        check(raw_script, True);check(timing_script, True)
        bad = copy.deepcopy(ref);bad['full_batch_logits'].pop();check(raw_script, False, candidate=bad)
        bad = copy.deepcopy(ref);bad['full_batch_logits'][64]['row_request_ids'] = [0];check(raw_script, False, candidate=bad)
        bad = copy.deepcopy(ref);bad['full_batch_logits'][0]['sha256'] = '0' * 64;check(raw_script, False, candidate=bad)
        path = p / 'request-2-logits-63.bin';path.write_bytes(b'\0' + raw[2, 63][1:]);check(raw_script, False);path.write_bytes(raw[2, 63])
        bad = copy.deepcopy(rows);bad[2]['tokens'][0] = 99;check(timing_script, False, records=bad)
        bad = copy.deepcopy(rows);bad[4]['admitted_ms'] = 10;check(timing_script, False, records=bad)
        bad = copy.deepcopy(rows);bad[1]['packed_token_budget'] = 9;check(timing_script, False, records=bad)
        bad = copy.deepcopy(rows);bad[2]['delivery_ms'][2] = 0;check(timing_script, False, records=bad)
        bad = copy.deepcopy(rows);bad[1]['diagnostic_logits_dump'] = True;check(timing_script, False, records=bad)
        result = json.loads(check(timing_script, True, records=bad, extra=('--correctness-only',)))
        assert not result['performance_reported'] and 'elapsed_ms' not in result
        print(f'BF16 serving evidence verifiers PASSED: {checks} checks')


if __name__ == '__main__':
    main()
