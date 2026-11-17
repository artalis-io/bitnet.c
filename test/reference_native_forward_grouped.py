"""Whole-model grouped prefill oracle from preserved upstream CUDA components.

Run in the pinned source environment with an immutable output directory argv[1].
Candidate engine outputs are never used to produce expected values.
"""
from pathlib import Path
import hashlib

forward_source = Path(__file__).with_name('reference_native_forward.py')
assert hashlib.sha256(forward_source.read_bytes()).hexdigest() == '11bf37fe182b26453cec9698f762803348c2f1042235325e3f529242501a7f15'
text = forward_source.read_text()
marker = "\nwith safe_open(str(root/index['embed.weight'])"
assert text.count(marker) == 1
prefix, tail = text.split(marker)
exec(compile(prefix, str(forward_source), 'exec'))

# This is the independently preserved whole-layer oracle's initialization:
# original router, fused expert kernel, fixed-K gather, reconstruction kernels,
# unchanged upstream group planner, and vendor GEMM policy.
group_source = Path('/tmp/bitnet-goal-native-layer/capture.py')
assert sha(group_source.read_bytes()) == 'c4ad9bdb9b293f0eff8eb9e84152773622f86615cd091a30fe5d3f6c46cdf324'
setup = group_source.read_text().split('# The independently preserved original router')[0]
old = "p=Path(__file__).parent;tag=os.environ.get('LAYER_REFERENCE_TAG','original');outdir=p/tag;assert not outdir.exists();outdir.mkdir()"
assert setup.count(old) == 1
setup = setup.replace(old, 'p=Path(__file__).parent;outdir=out')
gn = {'__file__': str(group_source), 'out': out}
exec(compile(setup, str(group_source), 'exec'), gn)
assert gn['batch_cap'] == 5
group_manifest = json.loads(group_source.with_name('paired').joinpath('reference.json').read_text())
for name, digest in group_manifest['component_reference_sha256'].items():
    assert sha(Path(gn['libs'][name]._name).read_bytes()) == digest, name
for name, digest in group_manifest['source_sha256'].items():
    assert sha((gn['source_root']/name).read_bytes()) == digest, name
small_ffn = ffn
group_cases = []


def ffn(layer, x):
    R, D = x.shape
    if R <= 8:
        return small_ffn(layer, x)
    H, E, K = 2304, 384, 6
    libs = gn['libs']
    gate = dn['oldload'](f'l{layer:02}.gate', torch.float16).reshape(E, D)
    center = dn['oldload'](f'l{layer:02}.center', torch.float16)
    wi8 = torch.empty((2, E, D), dtype=torch.int8, device='cuda')
    sb = torch.empty(E, device='cuda')
    assert not libs['router'].prepare(gate.data_ptr(), wi8.data_ptr(), sb.data_ptr(), E, D, stream)
    S = libs['router'].slices(R, E, D)
    ahi = torch.empty((R, D), dtype=torch.int8, device='cuda'); alo = torch.empty_like(ahi)
    sa = torch.empty((R, D//128), device='cuda'); part = torch.empty((S, R, E), device='cuda')
    logits = torch.empty((R, E), dtype=torch.float16, device='cuda')
    ids = torch.empty((R, K), dtype=torch.int64, device='cuda')
    rw = torch.empty((R, K), dtype=torch.float16, device='cuda')
    assert not libs['router'].run(*(t.data_ptr() for t in [x, gate, wi8, sb, center, logits, ids, rw, ahi, alo, sa, part]), R, D, E, K, 1.5, stream)
    selected = sorted(set(ids.flatten().cpu().tolist()))
    experts = {}
    for e in selected:
        triples = []
        for role, k, n in [('w1', D, H), ('w3', D, H), ('w2', H, D)]:
            prefix = f'layers.{layer}.ffn.experts.{e}.{role}'; pieces = []
            for suffix, dt, tag, shape in [('trellis', torch.uint16, 'I16', [k//16, n//16, 48]), ('suh', torch.float16, 'F16', [k]), ('svh', torch.float16, 'F16', [n])]:
                meta, b = dn['read_tensor'](prefix+'.'+suffix)
                assert meta['dtype'] == tag and meta['shape'] == shape
                expert_provenance.append(dict(tensor=prefix+'.'+suffix, bytes=len(b), sha256=sha(b)))
                pieces.append(torch.frombuffer(bytearray(b), dtype=dt).cuda())
            meta, b = dn['read_tensor'](prefix+'.mul1')
            assert meta['dtype'] == 'I32' and meta['shape'] == [] and dn['struct'].unpack('<I', b)[0] == 0x83dcd12d
            expert_provenance.append(dict(tensor=prefix+'.mul1', bytes=len(b), sha256=sha(b)))
            triples.append(pieces)
        experts[e] = triples
    tables = [torch.tensor([experts[e][pr][pi].data_ptr() if e in experts else 0 for e in range(E)], dtype=torch.int64, device='cuda') for pr in range(3) for pi in range(3)]
    flat = ids.flatten(); order = flat.argsort(stable=False); tokens = order//K
    sw = rw.flatten()[order]
    inv = torch.empty_like(order).scatter_(0, order, torch.arange(R*K, device='cuda'))
    counts = torch.bincount(flat, minlength=E+1); cnt = counts.cpu().tolist()
    assert cnt[-1] == 0 and sum(cnt) == R*K
    starts = counts.cumsum(0)-counts
    base = [0]*E; kind = [0]*E; slots = 0
    for e, c in enumerate(cnt[:E]):
        if 0 < c <= 256:
            base[e] = slots; kind[e] = 1; slots += c
    groups = []
    for ee in gn['ns']['plan_groups']([e for e in range(E) if cnt[e] > 256], lambda e: cnt[e], gn['batch_cap']):
        M = max(cnt[e] for e in ee); groups.append(dict(experts=ee, rows=M, slot=slots))
        for b, e in enumerate(ee):
            base[e] = slots+b*M; kind[e] = 2
        slots += len(ee)*M
    bases = torch.tensor(base, dtype=torch.int64, device='cuda')
    kinds = torch.tensor(kind, dtype=torch.int64, device='cuda')
    tiers = []
    for tile, lo, hi in [(64, 33, 256), (32, 17, 32), (16, 1, 16)]:
        active = sum(lo <= c <= hi for c in cnt[:E])
        if active:
            sms = gn['sms']; W = min(sms//min(sms//8, active), 32); G = min(active, sms//W)
            tiers.append(dict(tile_m=tile, tile_n=256 if tile == 16 else 128, lo=lo, hi=hi, active=active, width=W, groups=G))
    scratch = torch.empty((slots, D), device='cuda')
    scratch.view(torch.uint8).fill_(0xa5)
    routed = torch.zeros((R, D), device='cuda')
    locks = torch.zeros(gn['locks_n'], dtype=torch.int32, device='cuda')
    for tier in tiers:
        G = tier['groups']
        state = [torch.full((G, 256, D), -1., dtype=torch.float16, device='cuda') for _ in range(2)]+[torch.full((G, 256, H), -1., dtype=torch.float16, device='cuda') for _ in range(2)]
        args = [x, *state, routed, *tables, counts, tokens, sw, locks, scratch, bases]
        arr = (ptr*len(args))(*(v.data_ptr() for v in args))
        assert not libs['fused'].ref_launch(arr, D, H, E, K, 256, tier['lo'], tier['hi'], 10., tier['tile_m'], tier['tile_n'], tier['width'], G, 1, stream)
    # The first threshold fixtures have no expert with >256 assigned rows.
    # Reject rather than replace the original heavy reconstruction path.
    assert not groups, 'extend original grouped reconstruction before using >256 rows/expert'
    assert not libs['gather'].ref_gather(*(t.data_ptr() for t in [routed, scratch, flat, inv, starts, bases, kinds, sw]), R, D, K, E, stream)
    weights = [matrix(f'layers.{layer}.ffn.shared_experts.{r}') for r in ['w1', 'w3', 'w2']]
    sg = x@weights[0]; su = x@weights[1]; sharedact = torch.empty_like(sg)
    assert not libs['fold'].ref_act(sg.data_ptr(), su.data_ptr(), sharedact.data_ptr(), sharedact.numel(), 10., stream)
    shared = torch.empty((R, D), device='cuda')
    assert not libs['dense'].ref_dense(gn['handle'], sharedact.data_ptr(), weights[2].data_ptr(), shared.data_ptr(), R, H, D, 1)
    final = routed+shared
    torch.cuda.synchronize()
    group_cases.append(dict(layer=layer, rows=R, position=position, selected=len(selected), max_expert_rows=max(cnt), slots=slots, tiers=tiers, heavy_groups=groups))
    return final, ids, rw


# Keep the original whole-model graph, but cross both execution thresholds:
# R>8 selects fused grouped prefill; R>32 selects F32 HC weight accumulation.
tail = marker+tail
old_specs = "specs=[('single',1,0,True),('prefill',7,0,True),('decode',1,7,False)]\nif extended:specs+=[('chunk',7,8,False)]+[(f'greedy-{i}',1,15+i,False)for i in range(5)]"
new_specs = "specs=[('prefill9',9,0,True),('greedy-small',1,9,False),('prefill33',33,0,True),('chunk17',17,33,False)]+[(f'greedy-{i}',1,50+i,False)for i in range(5)]"
assert tail.count(old_specs) == 1
tail = tail.replace(old_specs, new_specs)
old_hc = "own=torch.empty((R,4),device='cuda');fn=fn.half()"
assert tail.count(old_hc) == 1
tail = tail.replace(old_hc, "own=torch.empty((R,4),device='cuda');fn=fn.half()if R<=32 else fn")
old_mix = 'own.data_ptr(),R,5120,1,stream)'
assert tail.count(old_mix) == 1
tail = tail.replace(old_mix, 'own.data_ptr(),R,5120,int(R<=32),stream)')
needle = "emit('reference.json',"
assert tail.count(needle) == 1
tail = tail.replace(needle, "scope='Actual 40-layer grouped/fused prefill9/33, resumed17 and six autoregressive greedy steps; context55; no heavy reconstruction or long indexed selection'\n"+needle)
# This fixture has at most55 tokens, so original 768-row rings and1024-row
# compressed pools suffice. It does not claim long indexed-selection coverage.
exec(compile(tail, str(forward_source), 'exec'))
assert len(group_cases) == 120 and {c['rows'] for c in group_cases} == {9, 17, 33}
emit('grouped-provenance.json', (json.dumps(dict(scope='Actual 40-layer grouped/fused prefill9/33, resumed17 and six source self-fed greedy steps; context55; no heavy reconstruction or long indexed selection', grouped_cases=group_cases, producer_sha256=sha(Path(__file__).read_bytes()), parent_sha256=sha(forward_source.read_bytes()), layer_producer_sha256=sha(group_source.read_bytes()), source_sha256={s:sha((gn['source_root']/s).read_bytes()) for s in ['modules/block_sparse_mlp.py', 'modules/moe_batch_recon.py']}, binaries={str(Path(v._name)):sha(Path(v._name).read_bytes()) for v in gn['libs'].values()}), indent=2)+'\n').encode())
assert not gn['libs']['fold'].ref_destroy(gn['handle'])
print('CAPTURED grouped whole-model forward', len(group_cases), flush=True)
