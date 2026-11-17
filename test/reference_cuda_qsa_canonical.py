"""Build an independent deterministic QSA ranking oracle from observed scores.
Original pinned outputs remain immutable and separate. This defines lower-ID
score ties and chronological emission; it does not claim literal parity with
the pinned arrival-order-dependent index buffer. Native tests recompute scores.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('observed', type=Path)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    original = args.observed / 'manifest.json'
    cases = []

    def save(name, data):
        path = args.directory / name
        if path.exists():
            assert path.read_bytes() == data, name
        else:
            path.write_bytes(data)

    for case in json.loads(original.read_text()):
        if case['name'].startswith('tie-'):
            continue
        scores = np.fromfile(args.observed / case['scores']['file'], dtype='<f4')
        scores = scores.reshape(case['scores']['shape'])
        tokens, position = case['tokens'], case['position']
        packed = np.full((tokens, 2052), -1, dtype='<i4')
        for row in range(tokens):
            visible = (position + row + 1) // 4
            indices = np.arange(visible, dtype=np.int32)
            values = scores[row, :visible]
            assert np.isfinite(values).all() and (values >= 0).all()
            selected = indices[np.lexsort((indices, -values))[:512]]
            groups = np.sort(selected)
            expanded = (groups[:, None] * 4 + np.arange(4)).reshape(-1)
            tail = np.arange(visible * 4, position + row + 1, dtype=np.int32)
            selected_tokens = np.concatenate([expanded, tail])
            packed[row, :len(selected_tokens)] = selected_tokens
            packed[row, -1] = len(selected_tokens)
        name = case['name'] + '-canonical.bin'
        data = packed.tobytes()
        save(name, data)
        cases.append(dict(name=case['name'], position=position, tokens=tokens,
            score=case['scores'], canonical=dict(file=name,
                sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))))
    manifest = dict(policy='exact observed F32 score rank; lower-index ties; chronological group/token order; ratio4; budget2048; trailing count',
        original_manifest_sha256=hashlib.sha256(original.read_bytes()).hexdigest(), cases=cases)
    save('manifest.json', json.dumps(manifest, indent=2).encode())
    print('QSA_CANONICAL_VERIFIED', len(cases), 'cases')


if __name__ == '__main__':
    main()
