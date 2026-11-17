"""Original-source resumed graph with physical SWA rebase and pool growth.

The preserved upstream update block runs after attention. Live index selection,
heavy expert B/M and all earlier numerical profiles remain unchanged.
"""
from pathlib import Path
import ast
import hashlib
import textwrap

resume_parent = Path(__file__).with_name('reference_native_forward_index.py')
assert hashlib.sha256(resume_parent.read_bytes()).hexdigest() == '1f3101297f420318ba7cc2da48a7d88f43f309a867dc5bca8f5f11ce9c1d5cad'
resume_text = resume_parent.read_text()
resume_marker = "exec(compile(heavy_code, str(index_parent), 'exec'))"
assert resume_text.count(resume_marker) == 1
resume_prefix, resume_footer = resume_text.split(resume_marker)
exec(compile(resume_prefix, str(resume_parent), 'exec'))
resume_index_text = index_source_text
resume_index_init = index_source_init


def index_source_text(text):
    text = resume_index_text(text)
    replacements = {
        'rings={};pools={};crings={};position=0':
            'rings={};pools={};crings={};ring_begins={};ring_transitions=[];position=0',
        'rings.clear();pools.clear();crings.clear()':
            'rings.clear();pools.clear();crings.clear();ring_begins.clear()',
        "for l in range(40):rings[l]=torch.zeros((768,512),dtype=torch.float16,device='cuda')":
            "for l in range(40):rings[l]=torch.zeros((768,512),dtype=torch.float16,device='cuda');ring_begins[l]=0",
        'torch.zeros((1024//rate,n)': 'torch.zeros((2048//rate,n)',
        'win_floor=max(0,pos-127),ring_beg=0':
            'win_floor=pos-min(127,pos-ring_begins[layer],pos),ring_beg=ring_begins[layer]',
        'rings[layer][pos:pos+R]=kv': 'resume_ring_update(layer,kv,pos,R)',
        'raw=torch.cat((raw.clone(),torch.full((5,),2,dtype=torch.int64)))':
            'raw=torch.cat((raw.repeat(2),torch.full((5,),2,dtype=torch.int64)))',
    }
    for old, new in replacements.items():
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    return text


def index_source_init():
    global resume_ring_code, resume_ring_source, resume_ring_digest
    resume_index_init()
    source = index_ns['src']
    body = source.read_text()
    start = body.index('        offset = pos0 - rs.window_beg', body.index('    def _forward_cached_one('))
    end = body.index('\n        if return_o:', start)
    branch = textwrap.dedent(body[start:end])
    resume_ring_digest = hashlib.sha256(branch.encode()).hexdigest()
    assert resume_ring_digest == '723e10e5af6b0931825b93e7af283749bd1dac4ed23b13cf52e8074777a5acd0'
    resume_ring_source = source
    resume_ring_code = compile(branch, '<unchanged-original-post-attention-ring-update>', 'exec')
    ans['bt'] = torch.arange(8, dtype=torch.int32, device='cuda').reshape(1, 8)


def resume_ring_update(layer, kv, pos, rows):
    before = ring_begins[layer]
    state = ans['types'].SimpleNamespace(window_beg=before, wshift=0)
    env = dict(pos0=pos, seq=rows, rs=state,
               rsl=ans['types'].SimpleNamespace(ring_rows=768), ring=rings[layer],
               kv=kv.unsqueeze(0), w=128, ring_beg=before, PAGE_SIZE=256)
    exec(resume_ring_code, env)
    ring_begins[layer] = before+state.wshift
    ring_transitions.append(dict(frame=name, layer=layer, rows=rows, position=pos,
                                 before=before, after=ring_begins[layer], shift=state.wshift))


# Keep the complete selection method; only raise the explicit fixture-capacity
# guard now that the original ring update and larger paged pools are present.
resume_select_node = next(node for node in ast.parse(resume_text).body
                          if isinstance(node, ast.FunctionDef) and node.name == 'index_select_source')
resume_select = ast.get_source_segment(resume_text, resume_select_node)
resume_old = 'assert position+R <= 768 and entries <= 1024'
assert resume_select.count(resume_old) == 1
resume_select = resume_select.replace(resume_old, 'assert position+R <= 2048 and entries <= 2048')
exec(compile(resume_select, str(resume_parent), 'exec'))

resume_old = "new_specs = \"specs=[('prefill513',513,0,True)]+[(f'greedy-{i}',1,513+i,False)for i in range(5)]\""
resume_new = "new_specs = \"specs=[('prefill513',513,0,True),('resume513',513,513,False)]+[(f'greedy-{i}',1,1026+i,False)for i in range(5)]\""
assert heavy_code.count(resume_old) == 1
heavy_code = heavy_code.replace(resume_old, resume_new)
resume_old = "assert len(group_cases) == 40 and {c['rows'] for c in group_cases} == {513}"
assert heavy_code.count(resume_old) == 1
heavy_code = heavy_code.replace(resume_old, "assert len(group_cases) == 80 and {c['rows'] for c in group_cases} == {513}")
resume_scope = 'Actual 40-layer prefill513 plus resumed513 and five self-fed greedy steps; context1031; unchanged original post-attention physical SWA rebase, pool2048/BT8, live original index selection and heavy experts/f16acc0; no two-level candidates'
assert heavy_code.count(index_scope) == 2
heavy_code = heavy_code.replace(index_scope, resume_scope)
index_scope = resume_scope
exec(compile(heavy_code, str(resume_parent), 'exec'))
resume_footer = resume_footer.replace('context=518', 'context=1031').replace('source_pool_tokens=1024', 'source_pool_tokens=2048')
exec(compile(resume_footer, str(resume_parent), 'exec'))
assert len(ring_transitions) == 280
assert sum(record['shift'] != 0 for record in ring_transitions) == 40
assert all(record['before'] == 0 and record['after'] == 768
           for record in ring_transitions if record['shift'])
emit('resumed-provenance.json', (json.dumps(dict(
    scope=resume_scope, parent_sha256=sha(resume_parent.read_bytes()),
    producer_sha256=sha(Path(__file__).read_bytes()), context=1031,
    ring_source=str(resume_ring_source), ring_source_sha256=sha(resume_ring_source.read_bytes()),
    original_update_sha256=resume_ring_digest, ring_transitions=ring_transitions,
    source_pool_tokens=2048, source_pages=8,
    prompt='Original valid 513-token ID sequence repeated twice; five generated slots overwritten autoregressively',
), indent=2)+'\n').encode())
