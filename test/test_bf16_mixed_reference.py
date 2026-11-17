#!/usr/bin/env python3
"""Model-free validation of mixed replay identity and complete-logit checks."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import sys
import subprocess
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from verify_bf16_mixed_batch import validate
from prepare_bf16_mixed_reference import DEFAULT, schedule, validate_reference
from verify_bf16_mixed_reference import verify


class MixedReferenceTests(unittest.TestCase):
    def setUp(self):
        self.reference = json.loads(DEFAULT.read_text())

    @staticmethod
    def simple_history(outputs):
        r = dict(schema=4, token_budget=1, capacity=max(16, outputs), max_live=2,
                 identity_proof=dict(mapping={'r0': 0, 'r1': 1}),
                 streams=[], forward_trace=[], full_batch_logits=[], execution_roles=[],
                 engine_request={})
        for request in range(2):
            stream = dict(input_ids=[request], tokens=[(request + j) % 3 for j in range(outputs)],
                          logit_sha256=['0' * 64] * outputs, arrival_step=request * outputs)
            r['streams'].append(stream)
            for step in range(outputs):
                call = len(r['forward_trace'])
                r['forward_trace'].append(dict(req_ids=['r%d' % request], computed=[step], scheduled=[1],
                    positions=[step], input_ids=[request if step == 0 else stream['tokens'][step - 1]]))
                r['full_batch_logits'].append(dict(call=call, dtype='bfloat16', shape=[1, 3],
                    request_ids=[request], computed=[step], scheduled=[1], output_steps=[step], sha256='0' * 64))
                r['execution_roles'].append([int(step > 0)])
        return r

    def test_serving_output_counts_and_frozen_standalone(self):
        for outputs in (1, 64, 128, 256):
            with self.subTest(outputs=outputs), tempfile.TemporaryDirectory() as tmp:
                self.reference = self.simple_history(outputs)
                validate_reference(self.reference)
                if outputs == 64:
                    self.assertEqual(schedule(self.reference)[:4], b'BMS1')
                else:
                    with self.assertRaises(ValueError):
                        schedule(self.reference)
                directory = Path(tmp)
                self.synthetic_raw(directory)
                self.assertEqual(verify(self.reference, directory)['heads'], 2 * outputs)
                late = directory / ('logits-%d.bin' % (2 * outputs - 1))
                data = late.read_bytes()
                late.write_bytes(b'X' + data[1:])
                with self.assertRaises(ValueError):
                    verify(self.reference, directory)

    def test_reject_incomplete_or_oversized_output_histories(self):
        with self.assertRaises(ValueError):
            validate_reference(self.simple_history(257))
        for mutation in [
            lambda r: (r['streams'][1]['tokens'].pop(), r['streams'][1]['logit_sha256'].pop()),
            lambda r: (r['forward_trace'].pop(), r['full_batch_logits'].pop(), r['execution_roles'].pop()),
            lambda r: r['full_batch_logits'][-1]['output_steps'].__setitem__(0, 128),
            lambda r: (r['streams'][0].update(tokens=[], logit_sha256=[])),
        ]:
            candidate = self.simple_history(128)
            mutation(candidate)
            with self.assertRaises(ValueError):
                validate_reference(candidate)

    def test_controlled_peak_is_usage_not_limit(self):
        reference = self.simple_history(64)
        per = reference['capacity'] * 65536
        reference['engine_request']['kv_cache_memory_bytes'] = per * 2
        records = []
        for m, roles in zip(reference['full_batch_logits'], reference['execution_roles']):
            records.append(dict(kind='step', call=m['call'], duration_ms=1,
                rows=[[m['request_ids'][0], 1, m['computed'][0], roles[0]]]))
        requests = []
        for i, stream in enumerate(reference['streams']):
            submitted = 1 + 128 * i
            admitted = submitted + .1
            requests.append(dict(kind='request', id=i, arrival_step=stream['arrival_step'],
                submitted_ms=submitted, admitted_ms=admitted, tokens=stream['tokens'],
                delivery_ms=[admitted + 1 + 2 * j for j in range(64)]))
        elapsed = requests[-1]['delivery_ms'][-1] + 1
        records += [dict(kind='summary', correct=True, requests=2, outputs_each=64, steps=128,
                    peak_live=1, elapsed_ms=elapsed, tokens_per_second=128000 / elapsed,
                    prefill_calls=2, mixed_calls=0, decode_calls=126, diagnostic=True),
                    dict(kind='kv_pool', scope='kv_payload_only', kv_bytes_per_slot=per,
                    created=1, allocated_slots=1, reused=1, discarded=0, leased_slots=0,
                    reserved_kv_bytes=per, peak_kv_bytes=per, budget_bytes=2 * per)] + requests
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            ref_path, log_path = directory / 'reference.json', directory / 'run.jsonl'
            def check(candidate, candidate_reference=reference):
                ref_path.write_text(json.dumps(candidate_reference))
                log_path.write_text('\n'.join(json.dumps(x) for x in candidate))
                return subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] /
                    'bench/verify_bf16_arrivals.py'), str(log_path), '--reference', str(ref_path)],
                    capture_output=True, text=True)
            good = check(records)
            self.assertEqual(good.returncode, 0, good.stderr)
            self.assertEqual(json.loads(good.stdout)['peak_live'], 1)
            full = copy.deepcopy(reference)
            full['max_live'] = 1
            self.assertEqual(check(records, full).returncode, 0)  # Existing full-capacity case.
            for kind in ('reported_peak', 'pool_count', 'kv_charge', 'over_limit'):
                candidate, ref = copy.deepcopy(records), copy.deepcopy(reference)
                if kind == 'reported_peak':
                    next(x for x in candidate if x['kind'] == 'summary')['peak_live'] = 2
                elif kind == 'pool_count':
                    pool = next(x for x in candidate if x['kind'] == 'kv_pool')
                    pool.update(created=2, allocated_slots=2, reused=0,
                                reserved_kv_bytes=2 * per, peak_kv_bytes=2 * per)
                elif kind == 'kv_charge':
                    next(x for x in candidate if x['kind'] == 'kv_pool')['reserved_kv_bytes'] -= 1
                else:
                    ref['max_live'] = 1
                    tail = [x for x in candidate if x['kind'] == 'request']
                    tail[1].update(submitted_ms=tail[0]['submitted_ms'], admitted_ms=tail[0]['admitted_ms'],
                                   delivery_ms=list(tail[0]['delivery_ms']))
                self.assertNotEqual(check(candidate, ref).returncode, 0, kind)

    def test_portable_schedule(self):
        blob = schedule(self.reference)
        self.assertEqual(blob[:4], b'BMS1')
        self.assertEqual(hashlib.sha256(blob).hexdigest(),
                         '54a2b9838bbb472e1d8ab39e7866e6015d3dd707d37180c45e3cef19ae31b595')
        self.assertEqual(len(self.reference['full_batch_logits']), 66)

    def test_reject_mutated_identity(self):
        mutations = [
            lambda r: r.update(schema=3),
            lambda r: r.update(token_budget=1),
            lambda r: r.update(capacity=1),
            lambda r: r['execution_roles'][1].__setitem__(0, 0),
            lambda r: r['full_batch_logits'][0]['request_ids'].__setitem__(1, 1),
            lambda r: r['full_batch_logits'][0]['output_steps'].__setitem__(0, 1),
            lambda r: r['forward_trace'][0]['input_ids'].__setitem__(0, 99),
            lambda r: r['forward_trace'][0]['positions'].__setitem__(0, 1),
            lambda r: r['full_batch_logits'].pop(),
            lambda r: r['streams'][0]['tokens'].pop(),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                r = copy.deepcopy(self.reference)
                mutation(r)
                with self.assertRaises((ValueError, IndexError, KeyError)):
                    schedule(r)

    def synthetic_raw(self, directory):
        r = self.reference
        for m in r['full_batch_logits']:
            rows = []
            m['shape'][1] = 3
            for request, step in zip(m['request_ids'], m['output_steps']):
                raw = bytes([request, 63, step % 256, 64, m['call'] % 256, 65])
                rows.append(raw)
                if step >= 0:
                    r['streams'][request]['logit_sha256'][step] = hashlib.sha256(raw).hexdigest()
            raw = b''.join(rows)
            m['sha256'] = hashlib.sha256(raw).hexdigest()
            (directory / ('logits-%d.bin' % m['call'])).write_bytes(raw)
        return r

    def repeated_waves(self, waves):
        base = self.reference
        result = copy.deepcopy(base)
        result.update(streams=[], forward_trace=[], full_batch_logits=[], execution_roles=[], max_live=4)
        result['identity_proof'] = dict(mapping={})
        for wave in range(waves):
            offset = wave * len(base['streams'])
            names = {name: '%d:%s' % (wave, name) for name in base['identity_proof']['mapping']}
            result['identity_proof']['mapping'].update(
                {names[name]: request + offset for name, request in base['identity_proof']['mapping'].items()})
            result['streams'].extend(copy.deepcopy(base['streams']))
            for frame, matrix, roles in zip(base['forward_trace'], base['full_batch_logits'], base['execution_roles']):
                frame, matrix = copy.deepcopy(frame), copy.deepcopy(matrix)
                frame['req_ids'] = [names[name] for name in frame['req_ids']]
                matrix['call'] = len(result['full_batch_logits'])
                matrix['request_ids'] = [request + offset for request in matrix['request_ids']]
                result['forward_trace'].append(frame)
                result['full_batch_logits'].append(matrix)
                result['execution_roles'].append(list(roles))
        return result

    def test_serving_histories_beyond_standalone_replay(self):
        for waves in (4, 12):
            with self.subTest(waves=waves), tempfile.TemporaryDirectory() as tmp:
                self.reference = self.repeated_waves(waves)
                # Independent logical histories, only four active per cohort.
                self.assertEqual(len(self.reference['streams']), waves * 4)
                self.assertEqual(len(self.reference['forward_trace']), waves * 66)
                validate_reference(self.reference)
                with self.assertRaises(ValueError):
                    schedule(self.reference)
                directory = Path(tmp)
                self.synthetic_raw(directory)
                self.assertEqual(verify(self.reference, directory)['heads'], waves * 260)
                # Late corruption must remain visible after the old call bound.
                late = directory / 'logits-260.bin'
                raw = late.read_bytes()
                late.write_bytes(b'X' + raw[1:])
                with self.assertRaises(ValueError):
                    verify(self.reference, directory)
            self.reference = json.loads(DEFAULT.read_text())

    def test_serving_validation_bounds_and_late_identity(self):
        reference = self.repeated_waves(12)
        for mutation in [lambda r: r.update(max_live=3),
                         lambda r: r.update(max_live=17),
                         lambda r: r.update(streams=r['streams'] * 2),
                         lambda r: r.update(forward_trace=r['forward_trace'] * 6,
                                            full_batch_logits=r['full_batch_logits'] * 6,
                                            execution_roles=r['execution_roles'] * 6),
                         lambda r: r['forward_trace'][300]['positions'].__setitem__(0, 99),
                         lambda r: r['execution_roles'][300].__setitem__(0, 0),
                         lambda r: r['full_batch_logits'][300]['request_ids'].__setitem__(0, 0)]:
            candidate = copy.deepcopy(reference)
            mutation(candidate)
            with self.assertRaises(ValueError):
                validate_reference(candidate)

    def test_complete_matrix_and_row_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            r = self.synthetic_raw(directory)
            self.assertEqual(verify(r, directory)['heads'], 260)
            path = directory / 'logits-0.bin'
            raw = path.read_bytes()
            path.write_bytes(b'X' + raw[1:])
            with self.assertRaises(ValueError):
                verify(r, directory)
            path.write_bytes(raw[:-1])
            with self.assertRaises(ValueError):
                verify(r, directory)
            path.write_bytes(raw)
            r['streams'][0]['logit_sha256'][0] = '0' * 64
            with self.assertRaises(ValueError):
                verify(r, directory)
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                verify(r, directory)

    def serving_records(self):
        r = self.reference
        config = dict(kind='config', fused_steps=True, packed_prefill=True, matches_reference=True,
                      all_arrive_at_zero=True, serialized_execution=False, reference_kind='external_tokens',
                      mixed_lengths=True, requests=4, max_live=4, peak_live=4, output_tokens=64,
                      maxseq=8192, packed_token_budget=256, prefill_chunk=64, prompt_tokens=129,
                      reserved_context_slots_peak=32768, diagnostic_logits_dump=False)
        records = [config, dict(kind='bf16_policy', admitted=r['norm_policy'])]
        deliveries = [[None] * 64 for _ in range(4)]
        for m, roles in zip(r['full_batch_logits'], r['execution_roles']):
            rows = [dict(request=i, position=p, tokens=n, role=role) for i, p, n, role in
                    zip(m['request_ids'], m['computed'], m['scheduled'], roles)]
            records.append(dict(kind='mixed_step', call=m['call'], rows=rows,
                                has_prefill=m['call'] < 3, duration_ms=1))
            for request, step in zip(m['request_ids'], m['output_steps']):
                if step >= 0:
                    deliveries[request][step] = m['call'] + 1
        for i, stream in enumerate(r['streams']):
            records.append(dict(kind='request', id=i, tokens=stream['tokens'], prompt_tokens=len(stream['input_ids']),
                                admitted_ms=0, ttft_ms=deliveries[i][0], completed_ms=deliveries[i][-1], delivery_ms=deliveries[i]))
        records.extend([dict(kind='summary', prefill_calls=3, mixed_phase_calls=3, prefill_tokens=268,
                             decode_calls=252, prefill_phase_ms=0, mixed_phase_ms=3, decode_phase_ms=63,
                             elapsed_ms=67, output_tokens_per_second=256000/67),
                        dict(kind='batch', prefill_batches=2, mixed_batches=1, decode_batches=63)])
        return records

    def test_serving_geometry_and_timing(self):
        records = self.serving_records()
        self.assertTrue(validate(self.reference, records)['performance_qualified'])
        mutations = [
            lambda x: x[0].update(fused_steps=False),
            lambda x: x[0].update(diagnostic_logits_dump=True),
            lambda x: x[0].update(packed_token_budget=255),
            lambda x: x[2]['rows'][0].update(role=1),
            lambda x: x[3]['rows'][0].update(position=99),
            lambda x: x[2].update(duration_ms=float('nan')),
            lambda x: x[-2].update(mixed_phase_ms=4),
            lambda x: x[-1].update(decode_batches=64),
            lambda x: x[-3]['delivery_ms'].__setitem__(1, 0),
        ]
        for mutation in mutations:
            candidate = copy.deepcopy(records)
            mutation(candidate)
            with self.assertRaises(ValueError):
                validate(self.reference, candidate)
        records[0]['diagnostic_logits_dump'] = True
        result = validate(self.reference, records, correctness_only=True)
        self.assertFalse(result['performance_qualified'])
        self.assertNotIn('elapsed_ms', result)

    def test_retained_pool_accounting(self):
        records = self.serving_records()
        records[0]['kv_pool_enabled'] = True
        per = 8192 * 65536
        pool = dict(kind='kv_pool', scope='kv_payload_only', budget_bytes=per*4,
                    kv_bytes_per_slot=per, allocated_slots=4, leased_slots=0, created=4,
                    reused=0, discarded=0, reserved_kv_bytes=per*4, peak_kv_bytes=per*4)
        records.append(pool)
        self.assertTrue(validate(self.reference, records)['correct'])
        for key, bad in [('budget_bytes', per*4-1), ('reserved_kv_bytes', 0),
                         ('kv_bytes_per_slot', per-1), ('leased_slots', 1), ('reused', 1),
                         ('scope', 'total_vram')]:
            candidate = copy.deepcopy(records)
            candidate[-1][key] = bad
            with self.assertRaises(AssertionError):
                validate(self.reference, candidate)


if __name__ == '__main__':
    unittest.main()
