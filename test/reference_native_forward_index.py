"""Whole-model original-source index selection across the top512 boundary.

Live checkpoint projections and key history feed the preserved original indexer
AST, RoPE, score kernels and top-k packer. No engine outputs compute gold.
"""
from pathlib import Path
import hashlib

index_parent = Path(__file__).with_name('reference_native_forward_heavy.py')
assert hashlib.sha256(index_parent.read_bytes()).hexdigest() == 'b2ec65e2c1fa78fb45cfe80019f0eff5d2e2bc20971ebdd08bf7dd3ab4eb2254'
index_marker = "exec(compile(heavy_code, str(heavy_parent), 'exec'))"
index_parent_text = index_parent.read_text()
assert index_parent_text.count(index_marker) == 1
# Build the pinned heavy source program before running its fixture loop.
exec(compile(index_parent_text.split(index_marker)[0], str(index_parent), 'exec'))


def index_source_text(text):
    needle = "    grouped=ans['g']['dsa_attn'](q,pc,pr_,ans['bt'],"
    assert text.count(needle) == 1
    text = text.replace(needle, "    selection,k_len=index_select_source(layer,x,qres,T,owner)\n"+needle)
    needle = 'indices=None,k_len=0,pool_len=T'
    assert text.count(needle) == 1
    text = text.replace(needle, 'indices=selection,k_len=k_len,pool_len=T')
    # Extra raw slots are capacity for generated IDs, overwritten by source
    # argmax before use. The original 513-token prompt remains unchanged.
    needle = 'raw=raw.clone();extended='
    assert text.count(needle) == 1
    text = text.replace(needle, 'raw=torch.cat((raw.clone(),torch.full((5,),2,dtype=torch.int64)));extended=')
    needle = 'position=pos;tokens=raw[pos:pos+R]'
    assert text.count(needle) == 1
    return text.replace(needle, 'position=pos;index_cache.clear();index_shared.clear();tokens=raw[pos:pos+R]')


def index_source_init():
    global index_ns, index_cache, index_shared, index_steps, index_active
    capture = Path('/tmp/bitnet-goal-native-index-session/capture.py')
    assert hashlib.sha256(capture.read_bytes()).hexdigest() == 'd28de46631b6eb3ae4a7e9f65de26c8a4af3085fa87e0907f6893dfe7fb4eadb'
    setup = capture.read_text().split("parent=Path('/tmp/bitnet-goal-native-index-query/paired-reference')")[0]
    needle = "p=Path(__file__).parent;d=p/'reference';assert not d.exists();d.mkdir()"
    assert setup.count(needle) == 1
    setup = setup.replace(needle, 'p=Path(__file__).parent;d=out')
    index_ns = {'__file__': str(capture), 'out': out}
    exec(compile(setup, str(capture), 'exec'), index_ns)
    proof = json.loads(capture.with_name('reference').joinpath('reference.json').read_text())
    assert sha(index_ns['src'].read_bytes()) == proof['source_sha256']
    assert sha(index_ns['scorefile'].read_bytes()) == proof['score_sha256']
    assert sha(Path(index_ns['lib']._name).read_bytes()) == proof['topk_library_sha256']
    assert sha(Path(index_ns['rope']._name).read_bytes()) == proof['rope_library_sha256']
    index_cache, index_shared, index_steps, index_active = {}, {}, [], None
    # Capture only original scores; the wrapper changes no arithmetic or order.
    original_score = index_ns['score']

    def capture_score(q, w, keys, pos, rate, bound, block_table=None, epp=0):
        scores = original_score(q, w, keys, pos, rate, bound, block_table=block_table, epp=epp)
        record = dict(position=pos, rows=q.shape[0], entries=bound,
                      scores=save(index_active['tag']+f'.p{pos}.scores', scores))
        index_active['slabs'].append(record)
        return scores

    index_ns['ns']['dsa_indexer_scores'] = capture_score


class IndexSourceLinear:
    def __init__(self, weight):
        self.weight = weight
        self.out_features = weight.shape[-1]
        self.out_dtype = self.bias = self._pinned_store = None
        self.last = None

    def forward(self, x, params):
        self.last = ans['linear'](self, x, params)
        return self.last


def index_select_source(layer, x, qres, entries, owner):
    global index_active
    if owner < 0 or entries <= 512:
        return None, 0
    R = x.shape[0]
    assert position+R <= 768 and entries <= 1024
    source = max(l for l in cfg['index_source_layer_ids'] if l <= layer)
    tag = f'index.p{position}.r{R}.l{layer}'
    if source != layer:
        assert index_cache.get('source') == source
        indices, k = index_cache['selection']
        index_steps.append(dict(tag=tag, layer=layer, source=source, owner=owner,
                                position=position, rows=R, entries=entries, borrowed=True))
        return indices, k
    qlinear = IndexSourceLinear(ans['matrix'](layer, 'indexer.wq_b'))
    wlinear = IndexSourceLinear(ans['param'](layer, 'indexer.weights_proj').reshape(5120, 32))
    candidate = cfg['candidate_source_layer_id']
    obj = ans['types'].SimpleNamespace(
        idx_wq_b=qlinear, idx_weights=wlinear, index_n_heads=32, index_head_dim=128,
        rope_head_dim=64, inv_freq_compress=ans['freqs']['compress'],
        candidate_block_size=8, candidate_topk_blocks=2048,
        candidate_mode='source' if layer == candidate else 'use' if layer > candidate else None,
        index_topk=512, compress_rate=ratios[layer], SCORE_SLAB=256,
    )
    index_active = dict(tag=tag, layer=layer, source=source, owner=owner,
                        position=position, rows=R, entries=entries, borrowed=False, slabs=[])
    indices, k = index_ns['forward'](
        obj, x.unsqueeze(0), {}, qres.unsqueeze(0), pools[owner][2], entries, position,
        block_table=ans['bt'], epp=256//ratios[layer], shared=index_shared,
    )
    assert k == 512 and indices.shape == (R, 512)
    index_active.update(indices=save(tag+'.indices', indices),
                        queries=save(tag+'.queries', qlinear.last),
                        weights=save(tag+'.weights', wlinear.last))
    index_steps.append(index_active)
    index_cache.update(source=source, selection=(indices, k))
    return indices, k


needle = 'text = forward_source.read_text()'
assert heavy_code.count(needle) == 1
heavy_code = heavy_code.replace(needle, 'text = index_source_text(forward_source.read_text())')
needle = "exec(compile(prefix, str(forward_source), 'exec'))"
assert heavy_code.count(needle) == 1
heavy_code = heavy_code.replace(needle, needle+'\nindex_source_init()')
needle = "new_specs = \"specs=[('prefill385',385,0,True)]+[(f'greedy-{i}',1,385+i,False)for i in range(5)]\""
assert heavy_code.count(needle) == 1
heavy_code = heavy_code.replace(needle, "new_specs = \"specs=[('prefill513',513,0,True)]+[(f'greedy-{i}',1,513+i,False)for i in range(5)]\"")
needle = "assert len(group_cases) == 40 and {c['rows'] for c in group_cases} == {385}"
assert heavy_code.count(needle) == 1
heavy_code = heavy_code.replace(needle, "assert len(group_cases) == 40 and {c['rows'] for c in group_cases} == {513}")
index_scope = 'Actual 40-layer prefill513 and five self-fed greedy steps with original live index projections, scoring, top512 packing and shared readers; context518; heavy experts/f16acc0; no candidate two-level selection or physical SWA ring wrap'
assert heavy_code.count(heavy_scope) == 2
heavy_code = heavy_code.replace(heavy_scope, index_scope)
exec(compile(heavy_code, str(index_parent), 'exec'))
assert index_steps and any(not step['borrowed'] for step in index_steps)
emit('indexed-provenance.json', (json.dumps(dict(
    scope=index_scope, heavy_parent_sha256=sha(index_parent.read_bytes()),
    producer_sha256=sha(Path(__file__).read_bytes()), context=518, steps=index_steps,
    index_capture_sha256=sha(Path(index_ns['__file__']).read_bytes()),
    sources={str(f):sha(f.read_bytes()) for f in [index_ns['src'], index_ns['scorefile']]},
    binaries={str(Path(v._name)):sha(Path(v._name).read_bytes()) for v in [index_ns['lib'], index_ns['rope']]},
    heavy_layers=sum(bool(case['heavy_groups']) for case in group_cases),
    source_swa_window=128, source_ring_rows=768, source_pool_tokens=1024,
), indent=2)+'\n').encode())
