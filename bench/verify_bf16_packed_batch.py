#!/usr/bin/env python3
"""Validate a traced serving case before summarizing timings.

The C benchmark must already have completed with --packed-reference. This
checks its emitted streams against independent original tokens and the exact
reference geometry/policy; it does not compare performance against vLLM.
Uniform prompts use the independent reference's explicit chunk geometry.
"""
import argparse
from verify_bf16_kv_pool import validate_pool
import json
import math
from pathlib import Path


def percentiles(values):
    values = sorted(values)
    assert values and all(math.isfinite(x) and x >= 0 for x in values)
    return {'samples': len(values), 'method': 'nearest_rank',
            **{f'p{p}_ms': values[math.ceil(p * len(values) / 100) - 1]
               for p in (50, 95, 99)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--reference', type=Path, default=Path(__file__).resolve().parents[1]
                        / 'test/bf16_cuda_packed_traced_reference.json')
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    assert reference['schema'] == 2
    streams = reference['streams']
    count = len(streams)
    assert 1 <= count <= 16 and all(len(s['tokens']) == 64 for s in streams)
    prompt_tokens = len(streams[0]['input_ids'])
    assert prompt_tokens >= 2 and all(len(s['input_ids']) == prompt_tokens for s in streams)
    prefill_tokens = count * prompt_tokens
    capacity = reference['engine_request']['max_model_len']
    chunk = reference.get('prefill_chunk_tokens', prompt_tokens)
    assert isinstance(chunk, int) and 1 <= chunk <= capacity
    budget = count * chunk
    prefill_batches = (prompt_tokens + chunk - 1) // chunk
    assert budget <= 8192 and prompt_tokens + 64 <= capacity
    rows = [json.loads(line) for line in args.log.read_text().splitlines() if line.strip()]

    def one(kind):
        matches = [r for r in rows if r['kind'] == kind]
        assert len(matches) == 1, (kind, len(matches))
        return matches[0]

    config, summary, batches = one('config'), one('summary'), one('batch')
    validate_pool(config, rows)
    assert not config.get('fused_steps', False), 'mixed steps require schema-4 qualification'
    assert one('bf16_policy')['admitted'] == reference['norm_policy']
    assert config['packed_prefill'] and config['matches_reference']
    assert not config.get('diagnostic_logits_dump', False), 'diagnostic disk I/O is not a timing run'
    assert config['reference_kind'] == 'external_tokens'
    assert (config['requests'], config['max_live'], config['peak_live'],
            config['prompt_tokens'], config['output_tokens'], config['prefill_chunk'],
            config['packed_token_budget'], config['maxseq']) == (count, count, count, prompt_tokens, 64, chunk,
                                                               budget, capacity)
    assert not config['mixed_lengths']
    assert batches['prefill_batches'] == summary['prefill_calls'] == prefill_batches
    assert batches['decode_batches'] == 63
    assert summary['decode_calls'] == count * 63 and summary['prefill_tokens'] == prefill_tokens
    requests = [r for r in rows if r['kind'] == 'request']
    assert len(requests) == count and {r['id'] for r in requests} == set(range(count))
    ttft, inter_token = [], []
    for row in requests:
        expected = reference['streams'][row['id']]
        assert row['tokens'] == expected['tokens'], row['id']
        assert row['prompt_tokens'] == len(expected['input_ids']) == prompt_tokens
        delivery = row['delivery_ms']
        assert len(delivery) == 64 and all(math.isfinite(t) for t in delivery)
        assert 0 <= row['admitted_ms'] <= delivery[0]
        assert abs(delivery[0] - row['ttft_ms']) < 1e-5
        assert abs(delivery[-1] - row['completed_ms']) < 1e-5
        assert delivery[-1] <= summary['elapsed_ms']
        ttft.append(row['ttft_ms'])
        inter_token.extend(b - a for a, b in zip(delivery, delivery[1:]))
    for key in ('elapsed_ms', 'prefill_phase_ms', 'decode_phase_ms', 'output_tokens_per_second'):
        assert math.isfinite(summary[key]) and summary[key] > 0, key
    assert math.isclose(summary['output_tokens_per_second'], count * 64000 / summary['elapsed_ms'],
                        rel_tol=1e-6, abs_tol=1e-5)
    print(json.dumps({'reference_tokens_matched': count * 64, 'prefill_batches': prefill_batches,
                      'decode_batches': 63, 'elapsed_ms': summary['elapsed_ms'],
                      'output_tokens_per_second': summary['output_tokens_per_second'],
                      'prefill_phase_tokens_per_second': prefill_tokens * 1000 / summary['prefill_phase_ms'],
                      'decode_phase_tokens_per_second': count * 63000 / summary['decode_phase_ms'],
                      'ttft': percentiles(ttft), 'inter_token_latency': percentiles(inter_token)}, indent=2))


if __name__ == '__main__':
    main()
