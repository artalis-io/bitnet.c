"""Serial, paired process-to-first-byte/exit timing of the two native CLI modes.

Uses a preserved independent text reference for assertions, never as model input
IDs. Includes process/model/tokenizer setup and final cleanup. No token-level ITL
or prepared-request latency is inferred from pipe reads. Caches start empty in
each child; filesystem warmth and GPU clocks are uncontrolled.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', default='./bitnet')
    parser.add_argument('--model', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--pairs', type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.pairs <= 20:
        parser.error('--pairs must be 1..20')
    binary = Path(args.binary).resolve()
    model = Path(args.model).resolve()
    reference = Path(args.reference).resolve()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    metadata = reference / 'text-reference.json'
    text = json.loads(metadata.read_text())
    prompt = reference / text['prompt']
    expected = reference / text['outputs'][1]['full']
    assert digest(prompt) == text['files'][text['prompt']]['sha256']
    assert digest(expected) == text['files'][text['outputs'][1]['full']]['sha256']
    wanted = expected.read_bytes()
    assert wanted and len(text['generated_ids']) == 5
    assert digest(model / 'tokenizer.json') == text['tokenizer']['sha256']
    pins = {str(f): digest(f) for f in [binary, metadata, prompt, expected,
            model / 'config.json', model / 'tokenizer.json', model / 'model.safetensors.index.json']}
    env = dict(os.environ)
    env.pop('LD_LIBRARY_PATH', None)
    env.pop('LD_PRELOAD', None)
    assert env.get('CUDA_VISIBLE_DEVICES') and ',' not in env['CUDA_VISIBLE_DEVICES']
    assert env.get('BN_CUDA_DEVICE') == '0'
    hardware = subprocess.check_output([
        'nvidia-smi', '--id=' + env['CUDA_VISIBLE_DEVICES'],
        '--query-gpu=name,uuid,driver_version', '--format=csv,noheader'], text=True).strip()
    plan = dict(scope='Fresh CLI process: launch to first observed stdout byte and process exit',
                pairs=args.pairs, order='Alternating committed/prediction and prediction/committed pairs',
                prefill_tokens=512, maxseq=4096, new_tokens=5,
                ram_cache_mib=640, gpu_cache_mib=640, admit_decode=False,
                source_pins=pins, hardware=hardware,
                environment={k: env.get(k) for k in ['CUDA_VISIBLE_DEVICES', 'BN_CUDA_DEVICE', 'CUDA_DEVICE_ORDER']},
                limitations=['No prepared-request TTFT, token ITL or vLLM comparison',
                             'Fresh managed caches per process; OS file cache and clocks uncontrolled',
                             'Pipe observation includes host scheduling; reads may coalesce chunks',
                             'Small paired diagnostic; no statistical significance or general speedup claim'])
    (out / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    runs = []
    for pair in range(args.pairs):
        order = [False, True] if pair % 2 == 0 else [True, False]
        for predict in order:
            for path, sha in pins.items():
                assert digest(path) == sha, path
            label = f'{pair}-' + ('prediction' if predict else 'committed')
            command = [str(binary), '--native-text', str(model), '--profile', 'exl3-int4-f16',
                       '--prompt-file', str(prompt), '-n', '5', '--token-ids',
                       '--prefill-tokens', '512', '--maxseq', '4096',
                       '--ram-cache-mb', '640', '--gpu-cache-mb', '640']
            if predict:
                command.append('--prediction-first')
            first = None
            stdout = bytearray()
            with (out / (label + '.stderr')).open('wb') as error:
                start = time.perf_counter_ns()
                child = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=error)
                with child.stdout as pipe:
                    while True:
                        chunk = os.read(pipe.fileno(), 4096)
                        now = time.perf_counter_ns()
                        if not chunk:
                            break
                        if first is None:
                            first = now
                        stdout.extend(chunk)
                code = child.wait()
                end = time.perf_counter_ns()
            stderr = (out / (label + '.stderr')).read_bytes()
            (out / (label + '.stdout')).write_bytes(stdout)
            assert code == 0 and bytes(stdout) == wanted and first is not None, label
            assert [int(x) for x in re.findall(rb'^token_id=(\d+)$', stderr, re.M)] == text['generated_ids'], label
            position = text['token_count'] + 5 - int(predict)
            assert f'native_text_result generated=5 position={position} output_bytes={len(wanted)}'.encode() in stderr, label
            stats = re.search(rb'^native_text_result generated=5 position=\d+ output_bytes=\d+ ram_hits=(\d+) ram_misses=(\d+) upload_bytes=(\d+)$', stderr, re.M)
            assert stats, label
            runs.append(dict(pair=pair, prediction_first=predict, command=command, exit=code,
                             first_byte_ms=(first-start)/1e6, process_exit_ms=(end-start)/1e6,
                             ram_hits=int(stats[1]), ram_misses=int(stats[2]), expert_upload_bytes=int(stats[3]),
                             stdout_sha256=hashlib.sha256(stdout).hexdigest()))
            (out / 'runs.json').write_text(json.dumps(runs, indent=2) + '\n')
            print(label, runs[-1]['first_byte_ms'], runs[-1]['process_exit_ms'], flush=True)
    comparisons = []
    for pair in range(args.pairs):
        old = next(r for r in runs if r['pair'] == pair and not r['prediction_first'])
        new = next(r for r in runs if r['pair'] == pair and r['prediction_first'])
        comparisons.append(dict(pair=pair, first_byte_delta_ms=new['first_byte_ms']-old['first_byte_ms'],
                                exit_delta_ms=new['process_exit_ms']-old['process_exit_ms'],
                                expert_upload_delta_bytes=new['expert_upload_bytes']-old['expert_upload_bytes']))
    for path, sha in pins.items():
        assert digest(path) == sha, path
    result = dict(plan=plan, runs=runs, paired_comparisons=comparisons,
                  median_first_byte_delta_ms=statistics.median(r['first_byte_delta_ms'] for r in comparisons),
                  median_exit_delta_ms=statistics.median(r['exit_delta_ms'] for r in comparisons))
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print('PAIRED CLI LATENCY CHECKS PASSED', len(runs), flush=True)


if __name__ == '__main__':
    main()
