#!/usr/bin/env python3
"""Verify every BF16 output word by independent original-runtime row hashes.

Run reference_cuda_bf16_packed first, then pass its output directory here.
This command never generates or updates the frozen expected values.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--reference', type=Path,
                        default=Path(__file__).with_name('bf16_cuda_packed_traced_reference.json'))
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    assert reference['schema'] == 2
    streams = reference['streams']
    assert 1 <= len(streams) <= 16
    assert len(reference['full_batch_logits']) == 64
    assert [m['step'] for m in reference['full_batch_logits']] == list(range(64))
    vectors = 0
    for matrix in reference['full_batch_logits']:
        step = matrix['step']
        data = (args.output / f'logits-{step}.bin').read_bytes()
        rows, cols = matrix['shape']
        assert matrix['dtype'] == 'bfloat16' and rows == len(streams)
        assert len(data) == rows * cols * 2, (step, len(data))
        assert hashlib.sha256(data).hexdigest() == matrix['sha256'], ('batch logits', step)
        for row, stream in enumerate(streams):
            raw = data[row * cols * 2:(row + 1) * cols * 2]
            assert hashlib.sha256(raw).hexdigest() == stream['logit_sha256'][step], (step, row)
            vectors += 1
    assert vectors == 64 * len(streams)
    print(f'PACKED_REFERENCE PASSED: {vectors} complete original BF16 logit vectors')


if __name__ == '__main__':
    main()
