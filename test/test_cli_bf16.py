#!/usr/bin/env python3
"""Real CLI integration against frozen independent Qwen27 reference tokens.

Requires the pinned GGUF and a CUDA + MKL VML build; excluded from make test.
The fixture records reference image, model, plugin, and capture source hashes.
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
    output = Path(args.output or tempfile.mkdtemp(prefix='bitnet-cli-bf16-'))
    output.mkdir(parents=True, exist_ok=True)
    fixture = json.loads(Path(__file__).with_name('bf16_cli_reference.json').read_text())
    digest = hashlib.sha256()
    with model.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    assert digest.hexdigest() == fixture['model_sha256'], 'fixture requires pinned Qwen27 GGUF'
    results = {}

    def run(name, flags, *, success=True, stdin=None, expected=None, env_extra=None):
        command = [binary, str(model), *flags]
        result = subprocess.run(command, input=stdin, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, env={**os.environ, **(env_extra or {})})
        text = result.stdout + result.stderr
        (output / (name + '.log')).write_text(text)
        tokens = [int(value) for value in re.findall(r'^token_id=(\d+)$', result.stderr, re.M)]
        results[name] = {'exit': result.returncode, 'command': command, 'tokens': tokens}
        (output / 'results.json').write_text(json.dumps(results, indent=2))
        assert (result.returncode == 0) == success, (name, result.returncode, text[-2000:])
        assert 'falling back' not in text, name
        if expected is not None:
            assert tokens == expected, (name, tokens, expected)
        return text, tokens

    for index, flags in enumerate([
        ['--cuda-bf16', '--draft', 'unused.gguf'],
        ['--cuda-bf16', '--kv16'], ['--cuda-bf16', '--kv-tq', '3'],
        ['--cuda-bf16', '--flash'], ['--cuda-bf16', '--gpu-cpu-logits'],
        ['--cuda-bf16', '--webgpu'], ['--prefill-batch', '64'],
        ['--cuda-bf16', '--prefill-batch', '0'],
    ]):
        text, tokens = run('reject-' + str(index), flags, success=False)
        assert 'Model loaded' not in text and not tokens

    base = ['--cuda-bf16', '--maxseq', '2048', '--temp', '0',
            '--repeat-penalty', '1', '--quiet', '--token-ids']
    for label in ['short', 'medium', 'long']:
        case = fixture['cases'][label]
        flags = base + ['--prompt-token-ids', ','.join(map(str, case['prompt_ids'])), '-n', '8']
        text, _ = run(label, flags, expected=case['expected'])
        assert 'Resident BF16 CUDA model prepared' in text
        assert 'CUDA weights uploaded' not in text
        if label == 'medium':
            run('chunk64', flags + ['--prefill-batch', '64'], expected=case['expected'])
    short = ['--prompt-token-ids', ','.join(map(str, fixture['cases']['short']['prompt_ids']))]
    text, tokens = run('benchmark-reset', base + short + ['-n', '8', '--benchmark-runs', '2',
                       '--benchmark-warmup-runs', '1', '--bench-warmup-tokens', '2'])
    assert text.count('Benchmark sample') == 2 and not tokens
    text, _ = run('no-logits-reset', base + short + ['-n', '0', '--benchmark-runs', '2'])
    assert text.count('Benchmark sample') == 2
    run('tokenwise', base + short + ['-n', '0', '--no-prefill'])
    chat = ['--cuda-bf16', '--chat', '--maxseq', '128', '--temp', '0',
            '--repeat-penalty', '1', '--token-ids', '-n', '2']
    message = 'hello ' * 60
    text, tokens = run('chat-reset', chat, stdin=message + '\n/reset\n' + message + '\n/quit\n')
    assert '[conversation reset]' in text and len(tokens) == 4 and tokens[:2] == tokens[2:]
    text, rollover = run('chat-rollover', chat, stdin=(message + '\n') * 3 + '/quit\n')
    assert text.count('[context full') == 2
    assert len(rollover) == 6 and rollover[:2] == rollover[2:4] == rollover[4:]
    context = fixture['chat_context']
    position_chat = chat + ['--maxseq', str(context['capacity'])]
    text, tokens = run('chat-final-token', position_chat,
                       stdin=context['message'] + '\n/quit\n')
    assert len(tokens) == context['generated_tokens']
    assert '[' + str(context['expected_position']) + '/' + str(context['capacity']) + ']' in text
    medium = fixture['cases']['medium']['prompt_ids']
    _, tokens = run('capacity-error', base + ['--maxseq', '128', '--prompt-token-ids',
                    ','.join(map(str, medium)), '-n', '8'], success=False)
    assert not tokens
    _, tokens = run('no-device', base + short + ['-n', '1'], success=False,
                    env_extra={'CUDA_VISIBLE_DEVICES': ''})
    assert not tokens
    print('CLI BF16 integration PASSED:', output)


if __name__ == '__main__':
    main()
