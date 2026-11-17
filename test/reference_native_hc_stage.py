"""Original-kernel HC/Engram chain; synthetic branches, not full-model logits.

Run in the pinned reference environment with output directory as argv[1].
No engine code is imported. Existing verified source captures supply immutable
weights, original kernels and the explicit INT4 BF16 -> F16 Engram adapter.
"""
from pathlib import Path
import ctypes as C
import hashlib
import json
import sys

# Reuse the independently preserved original hash/row/math setup, before its
# fixture loop. Digest it before execution; candidate output is never an oracle.
parent = Path(__file__).with_name('reference_native_engram_stage.py')
assert hashlib.sha256(parent.read_bytes()).hexdigest() == 'e2a815dd5f342b8ed79e2f65ee80fc8947cfa7d0ae22cb8b85037f55f7697670'
exec(compile(parent.read_text().split('def read_tokens(length):')[0], str(parent), 'exec'))
hcroot = Path('/tmp/bitnet-goal-native-hc-input')
assert sha((hcroot / 'reference.json').read_bytes()) == '99fa7740241a4c930d502fc1a2491e664b6f569a937a246f3a1da65a25e1827b'
assert sha((hcroot / 'reference.so').read_bytes()) == 'c4faa5ee9dc82cb33ba435d1b7ba2e96ae4368b12d50d76fee62fdf45f62eeec'
hcmanifest = json.loads((hcroot / 'reference.json').read_text())
for relative, digest in hcmanifest['source']['source_sha256'].items():
    assert sha((src / 'exllamav3_ext' / relative).read_bytes()) == digest
hc = C.CDLL(str(hcroot / 'reference.so'))
hc.chunks.argtypes = [it, it]
hc.chunks.restype = it
hc.mix.argtypes = [ptr] * 10 + [it] * 3 + [ptr]
hc.mix.restype = it
hc.apply.argtypes = [ptr] * 4 + [it] * 3 + [ptr]
hc.apply.restype = it
pr = Path('/tmp/bitnet-goal-native-params')
assert sha((pr / 'reference.json').read_bytes()) == 'bf3d2de17798df35b83ef4f1378761bb86ef84241ed12132ba8a4fd17cb44f20'
pm = json.loads((pr / 'reference.json').read_text())
parameters = {v['name']: v for v in pm['parameters']}

def parameter(name):
    v = parameters[name]
    b = (pr / (v['tag'] + '.expected')).read_bytes()
    assert sha(b) == v['expected_sha256']
    dtype = torch.float32 if 'hc_' in name else torch.bfloat16
    return torch.frombuffer(bytearray(b), dtype=dtype).reshape(v['shape']).cuda()

def normalize(x, gamma):
    y = torch.empty_like(x, dtype=torch.float16)
    assert not norm.reference(x.data_ptr(), gamma.data_ptr(), y.data_ptr(),
                              x.shape[0], 5120, 32, 16, 0, 1, 1e-20, 0., 1., stream)
    return y

def engram(layer, x, history, R):
    h, w, norms = models[layer]
    ids = h.compute_ngram_ids(history, R).reshape(-1)
    unique, inverse = ids.unique(sorted=True, return_inverse=True)
    planes = []
    for suffix in ['weight', 'scale']:
        key = f'layers.{layer}.engram.embed.{suffix}'
        with safe_open(str(root / index[key]), framework='pt', device='cpu') as reader:
            sl = reader.get_slice(key)
            planes.append(torch.cat([sl[int(i):int(i)+1] for i in unique], dim=0).cuda())
    decoded = torch.empty((R * 24, 256), device='cuda', dtype=torch.bfloat16)
    local = inverse.cuda().reshape(R, 24)
    module._engram_lookup_int4_kernel[(32,)](
        *planes, local, decoded, 0, len(unique), R*24, 1, 1, 0, 1, 1, 256, 32, 8, 32)
    result = ns['execute'](x, decoded.reshape(R, 6144).half(), w,
                           norms['k_weight'], norms['q_weight'])
    return result['final']

with safe_open(str(root / index['embed.weight']), framework='pt', device='cpu') as reader:
    embedding = reader.get_tensor('embed.weight')
raw = torch.frombuffer(bytearray((hashed / 'hash-1-1-513-text-tokens.i64le').read_bytes()), dtype=torch.int64)
groups = []
cases = []
for name, R, pos, reset in [('r1',1,0,True), ('r7',7,0,True), ('resume7',7,7,False),
                             ('decode1',1,14,False), ('r33',33,0,True)]:
    tokens = raw[pos:pos+R]
    history = torch.cat((torch.full((3,), 2, dtype=torch.int64), mapping[raw]))[pos:pos+R+3].reshape(1,-1)
    x = embedding[tokens].float().unsqueeze(1).expand(R,4,5120).contiguous().cuda()
    initial = save(name+'.embedding', x)
    pre = None
    group = dict(name=name, rows=R, position=pos, reset=reset,
                 tokens=save(name+'.tokens', tokens), embedding=initial, cases=[])
    for layer in range(40):
        if layer in models:
            x = engram(layer, x, history, R)
            group.setdefault('engram', {})[str(layer)] = save(f'{name}.l{layer}.engram', x)
        for kind in ['attn', 'ffn']:
            tag = f'{name}.l{layer}.{kind}'
            base_name = f'layers.{layer}.hc_{kind}_'
            fn, base, scale = [parameter(base_name+s) for s in ['fn','base','scale']]
            part = torch.empty((R,hc.chunks(R,5120),25), device='cuda')
            post = torch.empty((R,4), device='cuda'); comb = torch.empty((R,4,4), device='cuda')
            collapsed = torch.empty((R,5120), device='cuda'); own = torch.empty((R,4), device='cuda')
            f = fn.half() if R <= 32 else fn
            assert not hc.mix(x.data_ptr(),f.data_ptr(),base.data_ptr(),scale.data_ptr(),
                              part.data_ptr(),post.data_ptr(),comb.data_ptr(),collapsed.data_ptr(),
                              pre.data_ptr() if pre is not None else None,own.data_ptr(),R,5120,int(R<=32),stream)
            normalized = normalize(collapsed, parameter(f'layers.{layer}.{kind}_norm.weight'))
            # Source-independent branch values exercise the controller's lifetime
            # and residual ordering. They do NOT stand in for model attention/FFN.
            y = (((torch.arange(R*5120, device='cuda', dtype=torch.int64)*31+layer*17+(kind=='ffn'))%1021-510).float()/1024).reshape(R,5120)
            if kind == 'attn': y = y.half()
            record = dict(group=name,layer=layer,ffn=int(kind=='ffn'),rows=R,
                          outputs={k:save(tag+'.'+k,t) for k,t in [
                              ('partials',part),('post',post),('comb',comb),('collapsed',collapsed),
                              ('pre',own),('normalized',normalized)]}, branch=save(tag+'.branch',y))
            assert not hc.apply(x.data_ptr(),y.data_ptr(),post.data_ptr(),comb.data_ptr(),R,5120,int(y.dtype==torch.float16),stream)
            record['outputs']['streams'] = save(tag+'.streams',x)
            cases.append(record); group['cases'].append(len(cases)-1); pre=own
    final = torch.matmul(pre.unsqueeze(-2),x).squeeze(-2)
    group['collapsed'] = save(name+'.head-collapse',final)
    group['normalized'] = save(name+'.head-norm',normalize(final,parameter('norm.weight')))
    groups.append(group)
    print('CAPTURED',name,flush=True)
libraries = {}
for line in Path('/proc/self/maps').read_text().splitlines():
    f=line.split()[-1]
    if 'libcublas' in f and f.startswith('/') and f not in libraries: libraries[f]=sha(Path(f).read_bytes())
manifest = dict(scope='Connected embedding/Engram/HC/normalization/residual/head-collapse; SYNTHETIC branch values, not full-model execution',
                groups=groups,cases=cases,files=files,libraries=libraries,
                source_math=old['sources'],source_binaries=old['binaries'],
                hc_reference_sha256=sha((hcroot/'reference.json').read_bytes()),
                parameter_reference_sha256=sha((pr/'reference.json').read_bytes()))
emit('reference.json',(json.dumps(manifest,indent=2)+'\n').encode())
print('CAPTURED',len(cases),'sites',flush=True)
