#!/usr/bin/env python3
"""Real CLI CP profile regression against independent original vLLM captures.

Requires the pinned Qwen27 GGUF and CUDA SM120 + MKL VML. Excluded from
make test. Stock8192 and matched-arithmetic512 reference streams are distinct.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', default='./bitnet')
    parser.add_argument('--model', required=True)
    parser.add_argument('--output')
    args = parser.parse_args()
    binary = str(Path(args.binary).resolve())
    model = Path(args.model).resolve()
    output = Path(args.output or tempfile.mkdtemp(prefix='bitnet-cli-bf16-cp-'))
    output.mkdir(parents=True, exist_ok=True)
    fixture = json.loads(Path(__file__).with_name('bf16_cli_cp_reference.json').read_text())
    digest = hashlib.sha256()
    with model.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    assert digest.hexdigest() == fixture['model_sha256'], 'requires pinned Qwen27 GGUF'
    results = {}

    def run(name, flags, *, expected=None, success=True, stdin=None):
        command = [binary, str(model), *flags]
        result = subprocess.run(command, input=stdin, text=True, capture_output=True, env=os.environ)
        text = result.stdout + result.stderr
        (output / (name + '.log')).write_text(text)
        tokens = [int(value) for value in re.findall(r'^token_id=(\d+)$', result.stderr, re.M)]
        results[name] = {'exit': result.returncode, 'command': command, 'tokens': tokens}
        (output / 'results.json').write_text(json.dumps(results, indent=2))
        assert (result.returncode == 0) == success, (name, result.returncode, text[-2000:])
        assert 'falling back' not in text, name
        if expected is not None:
            assert tokens == expected, (name, tokens, expected)
        if success:
            assert 'Resident BF16 CUDA request prepared' in text, name
        else:
            assert 'Model loaded' not in text and not tokens, name
        return text, tokens

    for index, flags in enumerate([
        ['--cuda-bf16-profile', 'cp-fp32'],
        ['--cuda-bf16-profile', 'eager'],
        ['--cuda-bf16', '--cuda-bf16-profile', 'invalid'],
        ['--cuda-bf16', '--cuda-bf16-profile'],
    ]):
        run('reject-' + str(index), flags, success=False)

    base = ['--cuda-bf16', '--cuda-bf16-profile', 'cp-fp32', '--maxseq', '8192',
            '--temp', '0', '--repeat-penalty', '1', '--quiet', '--token-ids']
    for batch in [8192, 512]:
        for label in ['short', 'medium', 'long', 'repeat4097']:
            case = fixture['cases'][label]
            flags = base + ['--prefill-batch', str(batch), '--prompt-token-ids',
                            ','.join(map(str, case['prompt_ids'])), '-n', '64']
            text, _ = run(f'{label}-batch{batch}', flags,
                          expected=case['expected'][str(batch)])
            assert 'cp-fp32' in text and 'CUDA weights uploaded' not in text

    short = fixture['cases']['short']
    eager = ['--cuda-bf16', '--cuda-bf16-profile', 'eager', '--maxseq', '128',
             '--temp', '0', '--repeat-penalty', '1', '--quiet', '--token-ids',
             '--prompt-token-ids', ','.join(map(str, short['prompt_ids'])), '-n', '8']
    run('explicit-eager', eager, expected=short['expected']['8192'][:8])
    medium = fixture['cases']['medium']
    text, tokens = run('benchmark-reset', base + ['--prefill-batch', '512',
                       '--prompt-token-ids', ','.join(map(str, medium['prompt_ids'])),
                       '-n', '8', '--benchmark-runs', '2', '--benchmark-warmup-runs', '1'])
    assert text.count('Benchmark sample') == 2 and not tokens
    text, tokens = run('no-logits-reset', base + ['--prefill-batch', '512',
                       '--prompt-token-ids', ','.join(map(str, medium['prompt_ids'])),
                       '-n', '0', '--benchmark-runs', '2'])
    assert text.count('Benchmark sample') == 2 and not tokens
    chat = ['--cuda-bf16', '--cuda-bf16-profile', 'cp-fp32', '--chat', '--maxseq', '128',
            '--temp', '0', '--repeat-penalty', '1', '--token-ids', '-n', '4']
    message = 'hello ' * 60
    text, tokens = run('chat-reset', chat,
                       stdin=message + '\n/reset\n' + message + '\n/quit\n')
    assert '[conversation reset]' in text and len(tokens) == 8 and tokens[:4] == tokens[4:]
    print('CLI BF16 CP integration PASSED:', output)


if __name__ == '__main__':
    main()
