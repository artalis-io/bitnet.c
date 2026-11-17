#!/usr/bin/env python3
"""Validate uniform bounded admission waves against an independent reference.

Diagnostic logit dumps can be checked with --correctness-only; their disk I/O
is always excluded from performance summaries. This does not qualify mixed or
staggered arrivals, or a production KV allocator.
"""
import argparse
from verify_bf16_kv_pool import validate_pool
import json
import math
from pathlib import Path
from verify_bf16_packed_batch import percentiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--correctness-only', action='store_true')
    args = parser.parse_args()
    ref = json.loads(args.reference.read_text())
    assert ref['schema'] == 3
    streams, limit = ref['streams'], ref['max_live']
    count = len(streams)
    assert 1 <= limit <= 16 and limit < count <= 64
    waves = [list(range(i, min(i + limit, count))) for i in range(0, count, limit)]
    assert ref['admission_waves'] == waves
    prompt = len(streams[0]['input_ids'])
    chunk, capacity = ref['prefill_chunk_tokens'], ref['engine_request']['max_model_len']
    assert prompt >= 2 and chunk == prompt and limit * chunk <= 8192
    assert prompt + 64 <= capacity and ref['engine_request']['max_num_seqs'] == limit
    assert all(len(s['input_ids']) == prompt and len(s['tokens']) == 64 for s in streams)
    rows = [json.loads(l) for l in args.log.read_text().splitlines() if l.strip()]

    def one(kind):
        found = [r for r in rows if r['kind'] == kind]
        assert len(found) == 1, (kind, len(found))
        return found[0]

    config, summary, batch = one('config'), one('summary'), one('batch')
    validate_pool(config, rows)
    assert not config.get('fused_steps', False), 'mixed steps require schema-4 qualification'
    diagnostic = config.get('diagnostic_logits_dump', False)
    assert not diagnostic or args.correctness_only, 'diagnostic disk I/O is not a timing run'
    assert one('bf16_policy')['admitted'] == ref['norm_policy']
    assert config['packed_prefill'] and config['matches_reference']
    assert config['reference_kind'] == 'external_tokens'
    assert config['all_arrive_at_zero'] and not config['serialized_execution']
    assert not config['mixed_lengths']
    assert (config['requests'], config['max_live'], config['peak_live'],
            config['prompt_tokens'], config['output_tokens'], config['prefill_chunk'],
            config['packed_token_budget'], config['maxseq']) == (
                count, limit, limit, prompt, 64, chunk, limit * chunk, capacity)
    assert config['reserved_context_slots_peak'] == limit * capacity
    assert batch['prefill_batches'] == summary['prefill_calls'] == len(waves)
    assert batch['decode_batches'] == len(waves) * 63
    assert summary['decode_calls'] == count * 63 and summary['prefill_tokens'] == count * prompt
    for key in ('elapsed_ms', 'prefill_phase_ms', 'decode_phase_ms', 'output_tokens_per_second'):
        assert math.isfinite(summary[key]) and summary[key] > 0
    assert math.isclose(summary['output_tokens_per_second'], count * 64000 / summary['elapsed_ms'], rel_tol=1e-6, abs_tol=1e-5)
    requests = [r for r in rows if r['kind'] == 'request']
    assert len(requests) == count and {r['id'] for r in requests} == set(range(count))
    requests = sorted(requests, key=lambda r: r['id'])
    ttft, itl, admission, admitted_ttft, events = [], [], [], [], []
    for row, expected in zip(requests, streams):
        assert row['tokens'] == expected['tokens'], row['id']
        assert row['prompt_tokens'] == prompt
        delivery = row['delivery_ms']
        assert len(delivery) == 64 and all(math.isfinite(t) for t in delivery)
        assert all(a <= b for a, b in zip(delivery, delivery[1:]))
        assert 0 <= row['admitted_ms'] <= delivery[0]
        assert abs(delivery[0] - row['ttft_ms']) < 1e-5
        assert abs(delivery[-1] - row['completed_ms']) < 1e-5
        assert delivery[-1] <= summary['elapsed_ms']
        ttft.append(delivery[0]);admission.append(row['admitted_ms'])
        admitted_ttft.append(delivery[0] - row['admitted_ms'])
        itl.extend(b - a for a, b in zip(delivery, delivery[1:]))
        events.extend([(row['admitted_ms'], 1), (row['completed_ms'], -1)])
    for previous, current in zip(waves, waves[1:]):
        assert min(requests[i]['admitted_ms'] for i in current) >= max(requests[i]['completed_ms'] for i in previous)
    live, peak = 0, 0
    for _, delta in sorted(events):
        live += delta;peak = max(peak, live)
        assert 0 <= live <= limit
    assert live == 0 and peak == limit
    result = {'reference_tokens_matched': count * 64, 'requests': count,
              'peak_live_proven': peak, 'admission_waves': len(waves),
              'prefill_batches': len(waves), 'decode_batches': len(waves) * 63,
              'diagnostic_logits_dump': diagnostic, 'performance_reported': not args.correctness_only}
    if not args.correctness_only:
        result.update(elapsed_ms=summary['elapsed_ms'], output_tokens_per_second=summary['output_tokens_per_second'],
                      prefill_phase_ms=summary['prefill_phase_ms'], decode_batch_ms=summary['decode_phase_ms'] / (len(waves) * 63),
                      ttft=percentiles(ttft), inter_token_latency=percentiles(itl),
                      admission_wait=percentiles(admission), admitted_ttft=percentiles(admitted_ttft))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
