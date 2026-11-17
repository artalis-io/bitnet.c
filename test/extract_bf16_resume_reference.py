#!/usr/bin/env python3
"""Freeze only original vLLM streaming-reference hashes, never native outputs.
Usage: extract_bf16_resume_reference.py REFERENCE_DIR OUTPUT.json
"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

p = Path(sys.argv[1]); capture = p/'capture'
check = json.loads((p/'reference-check.json').read_text())
assert check['tokens_identical'] and check['all_raw_logits_identical']
status = json.loads((p/'capture-status.json').read_text())[0]['logits']
assert len(status['control']) == len(status['observed']) == 16
assert all(a['sha256'] == b['sha256'] for a,b in zip(status['control'],status['observed']))
manifest = {v['file']:v for v in json.loads((capture/'manifest.json').read_text())}
def load(name):
    r = manifest[name]; data = (capture/name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == r['sha256']
    dtype = {'torch.bfloat16':np.uint16,'torch.float32':np.float32,'torch.int64':np.int64}[r['dtype']]
    return np.frombuffer(data,dtype).reshape(r['shape'])
trace = json.loads((p/'forward-trace.json').read_text())
assert [r['positions'][0] for r in trace] == [0,129,130,131,132,133,134,135,136,200,201,202,203,204,205,206]
assert trace[8]['gdn']['num_prefills'] == 1 and trace[8]['gdn']['has_initial_state']['values'] == [True]
inputs = json.loads((p/'inputs.json').read_text())
assert trace[0]['input_ids'] == inputs['initial'] and trace[8]['input_ids'] == inputs['resume']
assert load('positions.bin').tolist() == [200]
selected = json.loads((capture/'selected-logits-index.json').read_text())['index']; assert selected == 9
ref = {'schema':1,'scope':check['scope'],'model_sha256':'3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e',
       'reference_image':'sha256:745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29',
       'inputs':{k:inputs[k] for k in ['initial','resume']},'tokens':json.loads((p/'control-tokens.json').read_text()),
       'positions':{'before':200,'after':201},'architecture':{'n_layers':64,'layers':['gdn' if i%4!=3 else 'attention' for i in range(64)],'state':[48,128,128],'history':[3,10240],'kv':[4,256],'vocab':248320,'hidden':5120},'payloads':{}}
def save(name,array):
    assert array.dtype in [np.dtype('uint16'),np.dtype('float32')]
    ref['payloads'][name] = {'dtype':'bfloat16' if array.dtype == np.uint16 else 'float32','shape':list(array.shape),'sha256':hashlib.sha256(array.tobytes()).hexdigest()}
for name, source in [('prefill-logits.bin','control-logits-8.bin'),('hidden-output.bin','hidden-output.bin'),('logits.bin','selected-logits.bin')]:
    save(name,load(source).reshape(-1))
for phase,n in [('before',200),('after',201)]:
    layers = json.loads((capture/f'{phase}-layers.json').read_text());assert [x['layer'] for x in layers] == list(range(64))
    for layer in layers:
        i = layer['layer'];assert layer['kind'] == ref['architecture']['layers'][i]
        if layer['kind'] == 'gdn':
            for name in ['state','history']: save(f'{phase}-l{i}-{name}.bin',load(f'{phase}-l{i}-{name}.bin')[0])
        else:
            raw = load(f'{phase}-l{i}-cache-blocks.bin'); lookup = {v:k for k,v in enumerate(layer['blocks'])};page=raw.shape[2]
            rows = np.stack([raw[lookup[layer['block_table'][0][t//page]],:,t%page,:] for t in range(n)])
            save(f'{phase}-l{i}-key.bin',rows[:,:,:256]);save(f'{phase}-l{i}-value.bin',rows[:,:,256:512])
assert len(ref['payloads']) == 259
Path(sys.argv[2]).write_text(json.dumps(ref,indent=2)+'\n')
