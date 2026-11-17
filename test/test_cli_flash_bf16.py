#!/usr/bin/env python3
"""Flash CLI regression against a frozen independent whole-model capture.

Requires real model files and CUDA+MKL; excluded from make test. The reference
must use the qualified explicit GDN/selection profile and the same prompt chunk
schedule as the CLI. This verifies correctness, not vLLM performance parity.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', default='./bitnet')
    parser.add_argument('--model', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--prefill-batch', type=int)
    parser.add_argument('--gpu-cache-mb', type=int)
    parser.add_argument('--preload', action='store_true')
    parser.add_argument('--maxseq', type=int, default=8192)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert args.gpu_cache_mb is None or args.gpu_cache_mb >= 0
    assert not args.preload or args.gpu_cache_mb
    reference = Path(args.reference)
    raw = (reference / 'manifest.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == args.manifest_sha256
    manifest = json.loads(raw)
    assert manifest['layer_limit'] == 48
    assert manifest['gdn_inverse_warps'] == 4 and manifest['qsa_selection'] == 'canonical'
    phases = manifest['phases']
    prompts = [phase for phase in phases if phase['prefill']]
    assert prompts and phases[:len(prompts)] == prompts
    prompt = sum((phase['tokens'] for phase in prompts), [])
    batch = min(args.prefill_batch or 1025, args.maxseq)
    assert batch > 0
    assert [phase['tokens'] for phase in prompts] == [prompt[i:i + batch] for i in range(0, len(prompt), batch)], 'reference chunk schedule differs from CLI'
    position = 0
    for phase in phases:
        assert phase['position'] == position
        position += len(phase['tokens'])
    assert position <= args.maxseq
    expected = []
    for phase in phases[len(prompts) - 1:]:
        name = f"phase{phase['phase']}-logits.bin"
        logits = (reference / name).read_bytes()
        assert hashlib.sha256(logits).hexdigest() == manifest['files'][name]['sha256']
        assert len(logits) == 248320 * 2
        values = [struct.unpack('<f', b'\0\0' + logits[i:i + 2])[0] for i in range(0, len(logits), 2)]
        assert max(range(len(values)), key=values.__getitem__) == phase['argmax']
        expected.append(phase['argmax'])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    base = [str(Path(args.binary).resolve()), str(Path(args.model).resolve()),
            '--cuda-bf16', '--maxseq', str(args.maxseq), '--temp', '0',
            '--repeat-penalty', '1', '--quiet', '--token-ids',
            '--prompt-token-ids', ','.join(map(str, prompt))]
    if args.prefill_batch is not None:
        base += ['--prefill-batch', str(args.prefill_batch)]
    if args.gpu_cache_mb is not None:
        base += ['--gpu-cache-mb', str(args.gpu_cache_mb)]
    if args.preload:
        base += ['--gpu-preload-experts']
    results = {'reference_manifest_sha256': args.manifest_sha256, 'cases': {}}

    def run(name, flags, tokens, *, command_base=None, stdin=None):
        command = (base if command_base is None else command_base) + flags
        result = subprocess.run(command, input=stdin, env=os.environ, capture_output=True, text=True)
        text = result.stdout + result.stderr
        (output / (name + '.log')).write_text(text)
        actual = [int(value) for value in re.findall(r'^token_id=(\d+)$', result.stderr, re.M)]
        results['cases'][name] = {'command': command, 'exit': result.returncode, 'tokens': actual}
        (output / 'results.json').write_text(json.dumps(results, indent=2))
        assert result.returncode == 0 and (tokens is None or actual == tokens), (name, result.returncode, actual, tokens, text[-2000:])
        assert 'Resident BF16 CUDA model prepared' in text and 'falling back' not in text
        if args.gpu_cache_mb and command_base is None:
            assert 'BF16 expert cache cumulative:' in text
        if args.preload:
            assert 'BF16 expert preload: experts=24576 upload_bytes=77017907200 resident_bytes=77017907200 staging_bytes=3993600' in text
            for stats in re.findall(r'^BF16 expert cache cumulative: (.*)$', text, re.M):
                fields = {k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', stats)}
                assert fields['hits'] > 0 and fields['misses'] == fields['miss_bytes'] == fields['upload_bytes'] == fields['evictions'] == 0
                assert fields['resident_bytes'] == 77017907200
        return text

    run('mmap', ['-n', str(len(expected))], expected)
    run('pread64', ['--pread', '--cache-mb', '64', '-n', str(len(expected))], expected)
    if len(prompt) <= 129:
        text = run('reset', ['--pread', '--cache-mb', '64', '-n', '0', '--benchmark-runs', '2'], [])
        assert text.count('Benchmark sample') == 2
        chat = base[:2] + ['--cuda-bf16', '--chat', '--maxseq', '128', '--temp', '0',
                          '--repeat-penalty', '1', '--token-ids', '-n', '2']
        if args.gpu_cache_mb is not None:
            chat += ['--gpu-cache-mb', str(args.gpu_cache_mb)]
        if args.preload:
            chat += ['--gpu-preload-experts']
        message = 'hello ' * 60
        text = run('chat-reset', [], None, command_base=chat,
                   stdin=message + '\n/reset\n' + message + '\n/quit\n')
        tokens = results['cases']['chat-reset']['tokens']
        assert '[conversation reset]' in text and len(tokens) == 4 and tokens[:2] == tokens[2:]
        text = run('chat-rollover', [], None, command_base=chat,
                   stdin=(message + '\n') * 3 + '/quit\n')
        tokens = results['cases']['chat-rollover']['tokens']
        assert text.count('[context full') == 2
        assert len(tokens) == 6 and tokens[:2] == tokens[2:4] == tokens[4:]
    print('Flash CLI reference, mmap/pread and applicable reset checks PASSED')


if __name__ == '__main__':
    main()
