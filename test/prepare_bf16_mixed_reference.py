#!/usr/bin/env python3
"""Validate an independently captured mixed trace and serialize its exact inputs."""
import argparse
import json
from pathlib import Path
import struct

DEFAULT = Path(__file__).with_name('bf16_cuda_mixed_reference.json')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_reference(reference, *, replay=False):
    """Validate logical histories; optionally serialize the bounded BMS1 replay.

    Serving can recycle at most sixteen live sessions across sixty-four logical
    requests, each producing 1–256 outputs. The standalone C replay retains its
    sixteen-request/256-call/64-output bounds.
    """
    r = reference
    require(r['schema'] == 4, 'expected mixed schema 4')
    streams = r['streams']
    nr, budget, capacity = len(streams), r['token_budget'], r['capacity']
    frames, matrices, roles = r['forward_trace'], r['full_batch_logits'], r['execution_roles']
    require(1 <= nr <= (16 if replay else 64) and 1 <= budget <= 8192 and 1 <= capacity <= 8192, 'invalid bounds')
    outputs = len(streams[0]['tokens'])
    require(1 <= outputs <= 256 and (not replay or outputs == 64), 'invalid output count')
    require(all(len(s['tokens']) == len(s['logit_sha256']) == outputs for s in streams),
            'inconsistent output counts')
    live = r.get('max_live', min(nr, 16))
    require(type(live) is int and 1 <= live <= min(nr, 16), 'invalid live request bound')
    require(1 <= len(frames) <= (256 if replay else 4096) and len(frames) == len(matrices) == len(roles), 'invalid call counts')
    mapping = r['identity_proof']['mapping']
    positions = [0] * nr
    sampled = [set() for _ in streams]
    blob = bytearray(b'BMS1' + struct.pack('<4i', nr, budget, capacity, len(frames))) if replay else None
    for call, (frame, matrix, kinds) in enumerate(zip(frames, matrices, roles)):
        ids = matrix['request_ids']
        computed, counts = matrix['computed'], matrix['scheduled']
        width = len(ids)
        require(matrix['call'] == call and matrix['dtype'] == 'bfloat16', 'invalid matrix identity')
        require(1 <= width <= live and len(set(ids)) == width, 'duplicate/invalid requests')
        require(len(computed) == len(counts) == len(kinds) == len(matrix['output_steps']) == width, 'row geometry')
        require(matrix['shape'][0] == width and matrix['shape'][1] > 0, 'matrix shape')
        require([mapping[x] for x in frame['req_ids']] == ids and frame['computed'] == computed and frame['scheduled'] == counts, 'trace identity mismatch')
        require(all(type(x) is int and x in (0, 1) for x in kinds), 'invalid role')
        require(kinds == sorted(kinds, reverse=True), 'decode prefix required')
        logical_prefill = [n > 1 or p < len(streams[i]['input_ids']) for i, p, n in zip(ids, computed, counts)]
        boundary = next((i for i, flag in enumerate(logical_prefill) if flag), width)
        require(kinds == [1] * boundary + [0] * (width - boundary), 'role differs from captured metadata contract')
        require(all(type(n) is int and n > 0 for n in counts) and sum(counts) <= budget, 'token budget')
        if replay:
            blob.extend(struct.pack('<i', width))
        offset = 0
        expected_positions = []
        for i, p, n, kind, step in zip(ids, computed, counts, kinds, matrix['output_steps']):
            require(type(i) is int and 0 <= i < nr, 'invalid request')
            stream = streams[i]
            require(p == positions[i] and p + n <= capacity and (kind == 0 or n == 1), 'invalid position/decode width')
            tokens = frame['input_ids'][offset:offset + n]
            require(tokens == (stream['input_ids'] + stream['tokens'])[p:p + n] and len(tokens) == n, 'input token mismatch')
            require(all(type(x) is int and 0 <= x <= 2147483647 for x in tokens), 'invalid token')
            expected_step = p + n - len(stream['input_ids'])
            require(step == expected_step and step < outputs, 'output step mismatch')
            if step >= 0:
                require(step not in sampled[i], 'duplicate sampled head')
                sampled[i].add(step)
            if replay:
                blob.extend(struct.pack('<4i', i, p, kind, n))
                blob.extend(struct.pack('<' + str(n) + 'i', *tokens))
            expected_positions.extend(range(p, p + n))
            positions[i] += n
            offset += n
        require(offset == len(frame['input_ids']) and frame['positions'] == expected_positions, 'flattened trace mismatch')
    require(all(x == set(range(outputs)) for x in sampled), 'missing sampled outputs')
    return bytes(blob) if replay else None


def schedule(reference):
    return validate_reference(reference, replay=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--reference', type=Path, default=DEFAULT)
    args = parser.parse_args()
    args.output.write_bytes(schedule(json.loads(args.reference.read_text())))


if __name__ == '__main__':
    main()
