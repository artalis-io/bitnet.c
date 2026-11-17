"""Independent native CLI default512 batch geometry, original source arithmetic.

The qualified parent fixes raw multilingual text and independently tokenizes it.
This fixture uses512+512+2 prompt rows and five self-fed source steps.
"""
from pathlib import Path
import hashlib,json
cli_parent=Path(__file__).with_name('reference_native_forward_text.py')
assert hashlib.sha256(cli_parent.read_bytes()).hexdigest()=='5f11684c9b76d9976af536b2718fec3fc7634d4e1f3810d49511c9557e2e3a08'
cli_marker="exec(compile(heavy_code,str(text_parent),'exec'))"
cli_source=cli_parent.read_text();assert cli_source.count(cli_marker)==1
cli_setup,cli_tail=cli_source.split(cli_marker)
exec(compile(cli_setup,str(cli_parent),'exec'))
old="specs=[('prefill513',513,0,True),('resume513',513,513,False)]+[(f'greedy-{i}',1,1026+i,False)for i in range(5)]"
new="specs=[('prefill512',512,0,True),('resume512',512,512,False),('tail2',2,1024,False)]+[(f'greedy-{i}',1,1026+i,False)for i in range(5)]"
assert heavy_code.count(old)==1;heavy_code=heavy_code.replace(old,new)
old="assert len(group_cases) == 80 and {c['rows'] for c in group_cases} == {513}"
assert heavy_code.count(old)==1
heavy_code=heavy_code.replace(old,"assert len(group_cases) == 80 and {c['rows'] for c in group_cases} == {512}")
old_scope=resume_scope
resume_scope='Actual raw-text source prefill512/resume512/tail2 and five self-fed greedy steps; context1031; unchanged original SWA update, live index selection and heavy experts/f16acc0'
assert heavy_code.count(old_scope)==2;heavy_code=heavy_code.replace(old_scope,resume_scope);index_scope=resume_scope
old='assert len(ring_transitions) == 280';assert text_footer.count(old)==1
text_footer=text_footer.replace(old,'assert len(ring_transitions) == 320')
old='source 40-layer fresh513/resumed513';assert cli_tail.count(old)==1
cli_tail=cli_tail.replace(old,'source 40-layer prefill512/resume512/tail2')
exec(compile(heavy_code,str(cli_parent),'exec'))
exec(compile(cli_tail,str(cli_parent),'exec'))
emit('cli-provenance.json',(json.dumps(dict(parent_sha256=sha(cli_parent.read_bytes()),producer_sha256=sha(Path(__file__).read_bytes()),prefill_rows=[512,512,2],context=1031,source_math_unchanged=True),indent=2)+'\n').encode())
