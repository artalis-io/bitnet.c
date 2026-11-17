#!/usr/bin/env python3
"""Check every original BF16 logit word for bounded, uniform request waves."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    ref = json.loads(args.reference.read_text())
    assert ref['schema'] == 3
    streams, limit = ref['streams'], ref['max_live']
    assert 1 <= limit <= 16 and limit < len(streams) <= 64
    waves = [list(range(i, min(i + limit, len(streams))))
             for i in range(0, len(streams), limit)]
    assert ref['admission_waves'] == waves
    assert all(len(s['tokens']) == len(s['logit_sha256']) == 64 for s in streams)
    matrices = ref['full_batch_logits']
    assert len(matrices) == 64 * len(waves)
    words, vectors, columns = 0, 0, None
    for step, matrix in enumerate(matrices):
        ids, output_step = waves[step // 64], step % 64
        assert matrix['step'] == step and matrix['output_step'] == output_step
        assert matrix['row_request_ids'] == ids and matrix['dtype'] == 'bfloat16'
        rows, cols = matrix['shape']
        assert rows == len(ids) and isinstance(cols, int) and cols > 0
        if columns is None:
            columns = cols
        assert columns == cols
        digest = hashlib.sha256()
        for request in ids:
            raw = (args.output / f'request-{request}-logits-{output_step}.bin').read_bytes()
            assert len(raw) == cols * 2, (request, output_step, len(raw))
            assert hashlib.sha256(raw).hexdigest() == streams[request]['logit_sha256'][output_step], (request, output_step)
            digest.update(raw)
            vectors += 1
            words += cols
        assert digest.hexdigest() == matrix['sha256'], ('matrix', step)
    assert vectors == len(streams) * 64
    print(f'SERVING_REFERENCE PASSED: {vectors} complete original BF16 logit vectors, '
          f'{words} words, {len(waves)} bounded request waves')


if __name__ == '__main__':
    main()
