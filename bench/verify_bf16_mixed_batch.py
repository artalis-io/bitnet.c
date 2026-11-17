#!/usr/bin/env python3
"""Qualify the bounded schema-4 mixed serving case; diagnostic I/O is untimed."""
import argparse
from verify_bf16_kv_pool import validate_pool
import hashlib
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'test'))
from prepare_bf16_mixed_reference import DEFAULT, require, schedule
from verify_bf16_packed_batch import percentiles


def validate(reference, records, correctness_only=False, logits=None):
    schedule(reference)
    def one(kind):
        values = [x for x in records if x.get('kind') == kind]
        require(len(values) == 1, 'expected one ' + kind)
        return values[0]
    config, summary, batch = one('config'), one('summary'), one('batch')
    validate_pool(config, records)
    policy = one('bf16_policy')['admitted']
    streams = reference['streams']
    count = len(streams)
    require(policy == reference['norm_policy'], 'arithmetic policy differs')
    require(config['fused_steps'] and config['packed_prefill'] and config['matches_reference'], 'mixed mode required')
    require(config['all_arrive_at_zero'] and not config['serialized_execution'], 'arrival/execution scope differs')
    require(config['reference_kind'] == 'external_tokens' and config['mixed_lengths'], 'reference geometry differs')
    require((config['requests'], config['max_live'], config['peak_live'], config['output_tokens'], config['maxseq'], config['packed_token_budget']) ==
            (count, count, count, 64, reference['capacity'], reference['token_budget']), 'capacity or budget differs')
    require(config['prefill_chunk'] == reference['engine_request']['long_prefill_token_threshold'], 'chunk differs')
    require(config['prompt_tokens'] == max(len(s['input_ids']) for s in streams), 'prompt length differs')
    require(config['reserved_context_slots_peak'] == count * reference['capacity'], 'reservation differs')
    diagnostic = config.get('diagnostic_logits_dump', False)
    require(not diagnostic or correctness_only, 'diagnostic disk I/O is not a timing run')
    require(logits is None or (diagnostic and correctness_only), 'logit dumps require diagnostic correctness mode')
    steps = [x for x in records if x.get('kind') == 'mixed_step']
    require(len(steps) == len(reference['full_batch_logits']), 'step count differs')
    expected_p = expected_m = expected_d = 0
    phase_count = 0
    phase_ms = decode_ms = 0.0
    prompt_tokens = decode_tokens = 0
    for call, (step, matrix, roles) in enumerate(zip(steps, reference['full_batch_logits'], reference['execution_roles'])):
        expected = [dict(request=i, position=p, tokens=n, role=role) for i, p, n, role in
                    zip(matrix['request_ids'], matrix['computed'], matrix['scheduled'], roles)]
        require(step['call'] == call and step['rows'] == expected, 'mixed geometry differs at call %d' % call)
        duration = step['duration_ms']
        require(math.isfinite(duration) and duration > 0, 'invalid duration')
        has_prefill = False
        for row in expected:
            pending = row['position'] < len(streams[row['request']]['input_ids'])
            has_prefill |= pending
            if pending:
                prompt_tokens += row['tokens']
            else:
                decode_tokens += row['tokens']
        require(step['has_prefill'] == has_prefill, 'logical phase differs')
        if has_prefill:
            phase_count += 1
            phase_ms += duration
        else:
            decode_ms += duration
        if all(role == 0 for role in roles):
            expected_p += 1
        elif all(role == 1 for role in roles):
            expected_d += 1
        else:
            expected_m += 1
    require((batch['prefill_batches'], batch['mixed_batches'], batch['decode_batches']) == (expected_p, expected_m, expected_d), 'backend counters differ')
    require((summary['prefill_calls'], summary['mixed_phase_calls'], summary['prefill_tokens'], summary['decode_calls']) ==
            (phase_count, phase_count, prompt_tokens, decode_tokens), 'logical counters differ')
    require(summary['prefill_phase_ms'] == 0, 'mixed phase mislabeled as pure prefill')
    require(abs(summary['mixed_phase_ms'] - phase_ms) < .001 and abs(summary['decode_phase_ms'] - decode_ms) < .001, 'phase sums differ')
    elapsed = summary['elapsed_ms']
    require(math.isfinite(elapsed) and elapsed >= phase_ms + decode_ms, 'invalid elapsed time')
    require(abs(summary['output_tokens_per_second'] - count * 64 * 1000 / elapsed) < .001, 'throughput differs')
    requests = [x for x in records if x.get('kind') == 'request']
    require([x['id'] for x in requests] == list(range(count)), 'request identities differ')
    ttft, intervals = [], []
    for request, stream in zip(requests, streams):
        require(request['tokens'] == stream['tokens'] and request['prompt_tokens'] == len(stream['input_ids']), 'output stream differs')
        times = request['delivery_ms']
        require(len(times) == 64 and all(math.isfinite(t) for t in times), 'invalid delivery times')
        require(0 <= request['admitted_ms'] <= times[0] and all(b > a for a, b in zip(times, times[1:])) and times[-1] <= elapsed, 'delivery order differs')
        require(request['ttft_ms'] == times[0] and request['completed_ms'] == times[-1], 'delivery summary differs')
        ttft.append(times[0])
        intervals.extend(b - a for a, b in zip(times, times[1:]))
    if logits is not None:
        for request, stream in enumerate(streams):
            for output, expected in enumerate(stream['logit_sha256']):
                raw = (logits / ('request-%d-logits-%d.bin' % (request, output))).read_bytes()
                require(len(raw) == reference['full_batch_logits'][0]['shape'][1] * 2 and hashlib.sha256(raw).hexdigest() == expected, 'sampled logits differ')
    result = dict(correct=True, steps=len(steps), sampled_heads=count * 64, performance_qualified=not correctness_only)
    if not correctness_only:
        result.update(elapsed_ms=elapsed, output_tokens_per_second=summary['output_tokens_per_second'],
                      initial_mixed_phase_ms=phase_ms, remaining_decode_mean_step_ms=decode_ms / (len(steps) - phase_count),
                      last_request_ttft_ms=max(ttft), ttft_ms=percentiles(ttft), inter_token_ms=percentiles(intervals))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--reference', type=Path, default=DEFAULT)
    parser.add_argument('--correctness-only', action='store_true')
    parser.add_argument('--logits', type=Path)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.log.read_text().splitlines() if line.startswith('{')]
    print(json.dumps(validate(json.loads(args.reference.read_text()), records, args.correctness_only, args.logits), indent=2))


if __name__ == '__main__':
    main()
