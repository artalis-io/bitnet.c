#!/usr/bin/env python3
"""Validate and freeze an original B4 prefill/decode capture, independently of native execution."""
import argparse,hashlib,json,struct,sys
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser(description="Freeze only internally stable original four-request CUDA reference data; no native expected outputs.")
parser.add_argument('reference',type=Path)
parser.add_argument('model',type=Path)
parser.add_argument('output',type=Path)
args=parser.parse_args()
p=args.reference
def read(name):return json.loads((p/name).read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
assert read('exit.json')['exit']==0
check=read('reference-check.json');assert check['tokens_identical'] and check['all_raw_logits_identical'] and check['captured_decode']
control=read('control-tokens.json')['batch4'];assert len(control)==4 and all(len(row)==64 for row in control)
assert control==read('observed-tokens.json')['batch4']
status=read('capture-status.json');assert len(status)==1
logits=status[0]['logits'];assert len(logits['control'])==len(logits['observed'])==64
row_hashes=[[]for _ in range(4)]
for step,(a,b) in enumerate(zip(logits['control'],logits['observed'])):
 assert a['shape']==b['shape']==[4,248320] and a['dtype']==b['dtype']=='torch.bfloat16'
 assert a['sha256']==b['sha256']
 for row in [a,b]:assert digest(p/'capture'/row['file'])==row['sha256']
 data=(p/'capture'/a['file']).read_bytes();arr=np.frombuffer(data,dtype='<u2').reshape(4,248320)
 for i in range(4):
  f32=(arr[i].astype(np.uint32)<<16).view(np.float32)
  assert int(np.argmax(f32))==control[i][step],(i,step)
  row_hashes[i].append(hashlib.sha256(arr[i].tobytes()).hexdigest())
stages=read('stage-manifest.json');assert len(stages)==257
for row in stages:assert row['shape']==[4,5120] and digest(p/'stages'/row['file'])==row['sha256']
selected=read('capture/selected-call.json');assert selected['positions']==[129]*4 and len(selected['input_ids'])==4
assert selected['input_ids']==[r[0]for r in control]
assert read('capture/selected-logits-index.json')['index']==1
for row in read('capture/manifest.json'):assert digest(p/'capture'/row['file'])==row['sha256']
report=dict(original_only=True,internally_stable=True,batch_size=4,prompt_tokens_each=129,decode_before_position=129,decode_after_position=130,matching_control_observed_vectors=256,matching_greedy_tokens=256,stage_count=257,stage_values=sum(np.prod(row['shape']).item() for row in stages),selected_decode_call=selected,scope='Independent original batch4 oracle, not native acceptance; native scheduler/prefill shape differ and must be matched before comparison',row_logit_hashes=row_hashes)
trace=read('forward-trace.json');assert len(trace)==64
prompts=read('prompts.json')['batch4'];assert len(prompts)==4 and all(len(x['prompt_token_ids'])==129 for x in prompts)
assert trace[0]['input_ids']==[i for x in prompts for i in x['prompt_token_ids']]
assert trace[0]['positions']==list(range(129))*4
assert trace[0]['gdn']['has_initial_state']['values']==[False]*4
assert trace[0]['gdn']['num_prefills']==4
for step,row in enumerate(trace[1:],1):
 assert row['positions']==[128+step]*4 and row['gdn']['num_decodes']==4
assert read('process-mode.json')['VLLM_ENABLE_V1_MULTIPROCESSING']==0
h=hashlib.sha256()
with args.model.open('rb') as f:
 for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
fixture=dict(schema=1,scope=report['scope'],model_sha256=h.hexdigest(),
 engine_request=read('engine-request.json'),process_mode=read('process-mode.json'),
 positions=dict(prefill_tokens_each=129,first_decode=129,after_first_decode=130),
 streams=[dict(input_ids=prompts[i]['prompt_token_ids'],tokens=control[i],logit_sha256=row_hashes[i])for i in range(4)],
 full_batch_logits=logits['control'],first_decode_stages=stages,
 captured_payloads=read('capture/manifest.json'),
 capture_provenance={name:digest(p/name)for name in ['reference.py','command.json','reference-check.json','capture-status.json','stage-manifest.json','forward-trace.json','capture/manifest.json','engine-effective-config.json']})
args.output.write_text(json.dumps(fixture,indent=2)+'\n')
print(str(args.output),digest(args.output))
