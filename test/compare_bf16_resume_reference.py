#!/usr/bin/env python3
"""Fail-closed comparison of resumed BF16 runtime dumps to a frozen oracle."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


def required_payloads(ref):
    arch = ref['architecture']
    phases = ref['tokens']
    if type(arch['n_layers']) is not int or arch['n_layers'] < 1 or len(arch['layers']) != arch['n_layers']:
        raise ValueError('incomplete layer coverage')
    if ref['schema'] != 1 or len(phases) != 2 or any(len(x) < 2 for x in phases):
        raise ValueError('invalid reference schema/phases')
    before = len(ref['inputs']['initial']) + len(phases[0]) - 1 + len(ref['inputs']['resume'])
    if ref['positions'] != {'before': before, 'after': before + 1}:
        raise ValueError('misaligned resume positions')
    shapes = {}
    for phase, pos in ref['positions'].items():
        for layer, kind in enumerate(arch['layers']):
            fields = {'state': ('float32', arch['state']), 'history': ('bfloat16', arch['history'])} if kind == 'gdn' else {'key': ('bfloat16', [pos] + arch['kv']), 'value': ('bfloat16', [pos] + arch['kv'])} if kind == 'attention' else None
            if fields is None:
                raise ValueError('unknown layer kind')
            for field, spec in fields.items():
                shapes[f'{phase}-l{layer}-{field}.bin'] = spec
    shapes.update({'prefill-logits.bin': ('bfloat16', [arch['vocab']]),
                   'logits.bin': ('bfloat16', [arch['vocab']]),
                   'hidden-output.bin': ('bfloat16', [arch['hidden']])})
    return shapes


def validate(ref, dumps, log, model):
    expected = required_payloads(ref)
    if set(ref['payloads']) != set(expected):
        raise ValueError('incomplete or extra reference payload coverage')
    records = [json.loads(line) for line in log.read_text().splitlines() if line.startswith('{')]
    inputs = [r for r in records if r.get('kind') == 'inputs']
    phases = [r for r in records if r.get('kind') == 'phase']
    if len(inputs) != 1 or {k: inputs[0][k] for k in ['initial', 'resume']} != ref['inputs']:
        raise ValueError('native input token IDs do not match oracle')
    if len(phases) != 2 or [r['phase'] for r in phases] != [0, 1] or [r['tokens'] for r in phases] != ref['tokens']:
        raise ValueError('native output phases do not match oracle')
    digest = hashlib.sha256()
    with model.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    if digest.hexdigest() != ref['model_sha256']:
        raise ValueError('checkpoint hash does not match oracle')
    values = 0
    for name, (dtype, shape) in expected.items():
        record = ref['payloads'][name]
        if record['dtype'] != dtype or record['shape'] != shape or not shape or any(type(n) is not int or n < 1 for n in shape):
            raise ValueError(f'invalid tensor metadata: {name}')
        count = math.prod(shape)
        size = count * (4 if dtype == 'float32' else 2)
        data = (dumps / name).read_bytes()
        if len(data) != size or hashlib.sha256(data).hexdigest() != record['sha256']:
            raise ValueError(f'tensor size/hash mismatch: {name}')
        values += count
    return {'passed': True, 'values': values, 'payloads': len(expected),
            'scope': ref['scope'], 'reference_sha256': hashlib.sha256(json.dumps(ref, sort_keys=True).encode()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=Path(__file__).with_name('bf16_resume_reference.json'))
    parser.add_argument('--dumps', type=Path, required=True)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = validate(json.loads(args.reference.read_text()), args.dumps, args.log, args.model)
    except (KeyError, ValueError, OSError, TypeError) as error:
        print(json.dumps({'passed': False, 'error': str(error)}))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    sys.exit(main())
