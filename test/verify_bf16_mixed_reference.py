#!/usr/bin/env python3
"""Verify every BF16 head in an untimed mixed replay, including partial prefill."""
import argparse
import hashlib
import json
from pathlib import Path
from prepare_bf16_mixed_reference import DEFAULT, require, validate_reference


def verify(reference, directory):
    validate_reference(reference)
    heads = 0
    words = 0
    for matrix in reference['full_batch_logits']:
        raw = (directory / ('logits-%d.bin' % matrix['call'])).read_bytes()
        rows, vocab = matrix['shape']
        require(len(raw) == rows * vocab * 2, 'matrix byte count')
        require(hashlib.sha256(raw).hexdigest() == matrix['sha256'], 'matrix hash mismatch at call %d' % matrix['call'])
        for row, (request, step) in enumerate(zip(matrix['request_ids'], matrix['output_steps'])):
            if step >= 0:
                data = raw[row * vocab * 2:(row + 1) * vocab * 2]
                require(hashlib.sha256(data).hexdigest() == reference['streams'][request]['logit_sha256'][step], 'sampled head hash mismatch')
        heads += rows
        words += rows * vocab
    return {'matrices': len(reference['full_batch_logits']), 'heads': heads, 'bf16_words': words, 'performance_qualified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--reference', type=Path, default=DEFAULT)
    args = parser.parse_args()
    print(json.dumps(verify(json.loads(args.reference.read_text()), args.directory)))


if __name__ == '__main__':
    main()
