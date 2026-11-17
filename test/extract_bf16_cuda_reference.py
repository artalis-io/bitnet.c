#!/usr/bin/env python3
"""Freeze original FlashInfer/CUDA graphs-disabled reference hashes only.
Usage: extract_bf16_cuda_reference.py REFERENCE_DIR MODEL.gguf OUTPUT.json [--case CASE]
No native outputs, execution policy, or runtime measurements supply goldens.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('reference', type=Path)
parser.add_argument('model', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--case', choices=['long', 'medium', 'short', 'repeat4097'], default='long')
args = parser.parse_args()
root, model, output = args.reference, args.model, args.output
check = json.loads((root/'reference-check.json').read_text())
cases = ['long', 'medium', 'short', 'repeat4097']
anchor = cases.index(args.case)
prompts = json.loads((root/'prompts.json').read_text())
prefix = len(prompts[args.case][0]['prompt_token_ids'])
assert prefix > 0
assert check['tokens_identical'] and check['all_raw_logits_identical'] and check['cases'] == cases
request = json.loads((root/'engine-request.json').read_text())
assert Path(request['model']).name == model.name
assert request['additional_config'] == {'gdn_prefill_backend': 'flashinfer', 'gdn_decode_kernel': 'cuda'}
assert request['compilation_config']['cudagraph_mode'] == 'NONE'
assert request['async_scheduling'] is False and request['max_num_seqs'] == 1
status = json.loads((root/'capture-status.json').read_text())[0]['logits']
assert len(status['control']) == len(status['observed']) == 256
assert all(a['sha256'] == b['sha256'] and a['shape'] == b['shape'] for a, b in zip(status['control'], status['observed']))
manifest = {r['file']: r for r in json.loads((root/'capture/manifest.json').read_text())}

def load(name):
    row = manifest[name]
    data = (root/'capture'/name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == row['sha256'], name
    dtype = {'torch.bfloat16': np.uint16, 'torch.float32': np.float32, 'torch.int64': np.int64}[row['dtype']]
    return np.frombuffer(data, dtype).reshape(row['shape'])

def record(array):
    array = np.ascontiguousarray(array)
    return {'shape': list(array.shape), 'dtype': 'float32' if array.dtype == np.float32 else 'bfloat16',
            'bytes': array.nbytes, 'sha256': hashlib.sha256(array.tobytes()).hexdigest()}

assert load('positions.bin').tolist() == [prefix]
selected = json.loads((root/'capture/selected-logits-index.json').read_text())
assert selected['index'] == anchor * 64 + 1
payloads = {'prefill-logits.bin': record(load(f'control-logits-{anchor*64}.bin')),
            'hidden-output.bin': record(load('hidden-output.bin')),
            'logits.bin': record(load('selected-logits.bin'))}
kinds = []
for phase, position in [('before', prefix), ('after', prefix + 1)]:
    layers = json.loads((root/'capture'/f'{phase}-layers.json').read_text())
    assert [r['layer'] for r in layers] == list(range(64))
    for layer in layers:
        index = layer['layer']; kind = layer['kind']
        if phase == 'before': kinds.append(kind)
        else: assert kind == kinds[index]
        if kind == 'gdn':
            for field, shape in [('state', [48,128,128]), ('history', [3,10240])]:
                array = load(f'{phase}-l{index}-{field}.bin')
                assert list(array.shape) == [1] + shape
                payloads[f'{phase}-l{index}-{field}.bin'] = record(array[0])
        else:
            assert kind == 'attention'
            cache = load(f'{phase}-l{index}-cache-blocks.bin')
            blocks = {value: offset for offset, value in enumerate(layer['blocks'])}
            page = cache.shape[2]
            assert cache.shape[1] == 4 and cache.shape[3] == 512
            combined = np.stack([cache[blocks[layer['block_table'][0][t//page]], :, t%page, :] for t in range(position)])
            payloads[f'{phase}-l{index}-key.bin'] = record(combined[:,:,:256])
            payloads[f'{phase}-l{index}-value.bin'] = record(combined[:,:,256:])
assert len(payloads) == 259
stages = {}
for row in json.loads((root/'stage-manifest.json').read_text()):
    data = (root/'stages'/row['file']).read_bytes()
    assert hashlib.sha256(data).hexdigest() == row['sha256']
    assert row['shape'] == [prefix,5120] and row['dtype'] == 'torch.bfloat16'
    stages[row['file']] = {'shape': row['shape'], 'dtype': 'bfloat16', 'bytes': len(data), 'sha256': row['sha256']}
expected_stages = {f'l{layer}-{role}.bin' for layer in range(64) for role in ['input-norm','block-output','ffn-norm','ffn-output']}
expected_stages.add('l64-input-norm.bin')
assert set(stages) == expected_stages
tokens = json.loads((root/'control-tokens.json').read_text())
streams = {}
for index, case in enumerate(cases):
    assert len(tokens[case]) == 64
    logits = [record(load(f'control-logits-{index*64+step}.bin')) for step in range(64)]
    assert all(row['shape'] == [1,248320] for row in logits)
    streams[case] = {'input_ids': prompts[case][0]['prompt_token_ids'], 'tokens': tokens[case], 'logits': logits}
assert len(streams['long']['input_ids']) == 1025
model_hash = hashlib.sha256()
with model.open('rb') as file:
    while data := file.read(16*1024*1024): model_hash.update(data)
result = {'schema': 1, 'model_sha256': model_hash.hexdigest(),
          'image': 'sha256:745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29',
          'scope': 'Original compiled FlashInfer/CUDA, synchronous graphsNONE, budget8192, BF16/F32 SSM. Long zero-state1025 prefill then first decode; four64-output streams. Not stock graph-mode reproducibility or performance qualification.',
          'engine_request': request, 'architecture': {'n_layers': 64, 'layers': kinds, 'state': [48,128,128], 'history': [3,10240], 'kv': [4,256]},
          'positions': {'before': prefix, 'after': prefix + 1}, 'payloads': payloads, 'stages': stages, 'streams': streams}
assert payloads['logits.bin']['sha256'] == streams[args.case]['logits'][1]['sha256']
if args.case != 'long':
    validation = json.loads((root/'reference-validation.json').read_text())
    assert validation['all256control_observed_logits_equal'] and validation['zero_state_prefix']
    assert validation['prefix_tokens'] == validation['before_position'] == prefix
    assert validation['after_position'] == prefix + 1 and validation['stages'] == 257
    result['scope'] = (f'Original compiled FlashInfer/CUDA, synchronous graphsNONE, budget8192, '
                       f'BF16/F32 SSM. {args.case} zero-state{prefix} prefill then first decode; '
                       'four64-output streams. Separate captured execution variant; not a '
                       'replacement for the frozen cold-compile reference or a performance result.')
    result['anchor_case'] = args.case
    result['capture_provenance'] = {
        'validation': validation,
        'reference_script_sha256': hashlib.sha256((root/'reference.py').read_bytes()).hexdigest(),
        'stage_manifest_sha256': hashlib.sha256((root/'stage-manifest.json').read_bytes()).hexdigest(),
    }
output.write_text(json.dumps(result, indent=2) + '\n')
print(output, hashlib.sha256(output.read_bytes()).hexdigest())
