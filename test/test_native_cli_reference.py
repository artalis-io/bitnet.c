"""Real CLI checks against independent native text sources; never generates gold.

Requires a CUDA build/model and the preserved source directories. Not make test.
"""
import argparse,hashlib,json,math,os,re,struct,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--binary',default='./bitnet');p.add_argument('--model',required=True);p.add_argument('--reference',required=True);p.add_argument('--parent-reference',required=True);p.add_argument('--output',required=True);p.add_argument('--short',action='store_true');a=p.parse_args()
binary=str(Path(a.binary).resolve());model=str(Path(a.model).resolve());out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
refs=[Path(a.reference),Path(a.parent_reference)];data=[json.loads((r/'text-reference.json').read_text())for r in refs]
assert data[0]['prompt_bytes']==4948 and data[0]['token_count']==1026 and data[0]['generated_ids']==[25744,14,223,21,16]
base=[binary,'--native-text',model,'--profile','exl3-int4-f16'];results={}
def run(name,flags,*,input=None,code=0,sink=False,extra=None):
 args=base+flags;env=dict(os.environ);env.update(extra or {})
 if sink:
  with open('/dev/full','wb')as target:r=subprocess.run(args,input=input,stdout=target,stderr=subprocess.PIPE,env=env)
  stdout=b''
 else:r=subprocess.run(args,input=input,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env);stdout=r.stdout
 (out/(name+'.stdout')).write_bytes(stdout);(out/(name+'.stderr')).write_bytes(r.stderr)
 results[name]=dict(exit=r.returncode,args=args);(out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
 assert r.returncode==code,(name,r.returncode,r.stderr[-2000:]);return stdout,r.stderr
prompt=(refs[0]/data[0]['prompt']).read_bytes();assert hashlib.sha256(prompt).hexdigest()==data[0]['files'][data[0]['prompt']]['sha256']
def check_trace(name,ref_index,flags,skip,input=None,*,predict=False,count=5):
 r=refs[ref_index];text=data[ref_index];ref=json.loads((r/'reference.json').read_text());host=json.loads((r/'host-reference.json').read_text())
 trace=out/(name+'-trace');trace.mkdir()
 stdout,stderr=run(name,flags+(['--prediction-first']if predict else[])+['-n',str(count),'--token-ids','--trace-dir',str(trace)],input=input)
 events=[json.loads(x)for x in(trace/'events.jsonl').read_text().splitlines()]
 forwards=[e for e in events if e['kind']=='forward'];tokens=[e for e in events if e['kind']=='token'];assert len(events)==len(forwards)+len(tokens)
 prefill=sum(not f.get('generated',False)for f in ref['frames'])
 selected=ref['frames'][:prefill+count-int(predict and count>0)]
 assert len(forwards)==len(selected)
 for i,(got,frame)in enumerate(zip(forwards,selected)):
  assert got==dict(kind='forward',index=i,position=frame['position'],rows=frame['rows'],vocabulary=129280)
  raw=(r/frame['tokens']).read_bytes();assert hashlib.sha256(raw).hexdigest()==ref['files'][frame['tokens']]['sha256']
  ids=[v[0]for v in struct.iter_unpack('<Q',raw)];assert(trace/f'forward-{i}.i32le').read_bytes()==struct.pack('<'+'i'*len(ids),*ids)
  name_=host['frames'][frame['name']]['last_logits'];gold=(r/name_).read_bytes();assert hashlib.sha256(gold).hexdigest()==host['files'][name_]['sha256']
  assert(trace/f'forward-{i}.f32le').read_bytes()==gold,(name,i)
 expected=text['outputs'][int(skip)]
 seen_f=seen_t=0
 for event in events:
  if event['kind']=='forward':seen_f+=1
  elif event['id']>=0:
   seen_t+=1;assert seen_f==prefill+seen_t-int(predict),'incorrect prediction/commit callback order'
  else:assert seen_t==count and seen_f==len(selected)
 assert [t['id']for t in tokens]==text['generated_ids'][:count]+[-1]
 if count==5:
  chunks=[c['bytes']for c in expected['chunks']];full=(r/expected['full']).read_bytes()
 else:
  # The first independently selected token is exactly ASCII "789" in the
  # pinned checkpoint vocabulary, so a zero/one-token flush has no UTF8 tail.
  assert count in (0,1)
  tokenizer=Path(text['tokenizer']['file']);raw=tokenizer.read_bytes()
  assert hashlib.sha256(raw).hexdigest()==text['tokenizer']['sha256']
  assert json.loads(raw)['model']['vocab']['789']==text['generated_ids'][0]
  first=(r/expected['chunks'][0]['file']).read_bytes();assert first==b'789'
  chunks=([len(first)]if count else[])+[0];full=first if count else b''
 assert [t['bytes']for t in tokens]==chunks and stdout==full
 assert [int(x)for x in re.findall(rb'^token_id=(\d+)$',stderr,re.M)]==text['generated_ids'][:count]
 position=text['token_count']+count-int(predict and count>0)
 assert f'native_text_result generated={count} position={position} output_bytes={len(full)}'.encode()in stderr
check_trace('default-file',0,['--prompt-file',str(refs[0]/data[0]['prompt'])],True)
if not a.short:
 check_trace('stdin-pinned-admit',0,['--prompt-file','-','--keep-special','--admit-decode','--pinned-staging'],False,input=prompt)
 check_trace('chunk513-inline',1,['-p',prompt.decode(),'--prefill-tokens','513','--maxseq','1280'],True)
 common=['--prompt-file',str(refs[0]/data[0]['prompt']),'-n','5']
 stdout,_=run('prepare',common+['--prepare-only']);plan=json.loads(stdout)
 assert plan['prepared'] and plan['prompt_tokens']==1026 and plan['prefill_tokens']==512 and plan['maxseq']==4096
 host_mb=math.ceil(plan['host_payload_peak_bytes']/1048576);device_mb=math.ceil(plan['device_payload_peak_bytes']/1048576)
 check_trace('bounded',0,common[:-2]+['--host-budget-mb',str(host_mb),'--device-budget-mb',str(device_mb)],True)
 _,err=run('weight-budget',common+['--device-budget-mb','1'],code=1);assert b'resident weight payload cap'in err and b'native_text_weights'not in err
 _,err=run('request-budget',common+['--device-budget-mb',str(device_mb-1)],code=1);assert b'aggregate payload cap'in err
 _,err=run('host-budget',common+['--host-budget-mb','1'],code=1);assert b'host payload cap'in err and b'native_text_weights'not in err
 _,err=run('context',common+['--maxseq','1024'],code=1);assert b'exceeds maxseq'in err and b'native_text_weights'not in err
 bad=out/'bad.utf8';bad.write_bytes(b'\xc0\x80')
 _,err=run('invalid-utf8',['--prompt-file',str(bad)],code=1);assert b'prompt encoding'in err and b'native_text_weights'not in err
 empty=out/'empty';empty.write_bytes(b'')
 _,err=run('empty',['--prompt-file',str(empty)],code=1);assert b'prompt file'in err and b'native_text_weights'not in err
 trace=out/'protected-trace';trace.mkdir();protected=trace/'events.jsonl';payload=b'preserve existing file';protected.write_bytes(payload)
 _,err=run('trace-protection',common+['--trace-dir',str(trace)],code=1);assert protected.read_bytes()==payload and b'native_text_weights'not in err
 run('stdout-error',common,code=1,sink=True)
 _,err=run('device-error',common,code=1,extra={'BN_CUDA_DEVICE':'999'});assert b'CUDA device'in err
 for i,flags in enumerate([['--profile','wrong'],['--chat'],['--temp','1'],['--kv16'],['--prefill-tokens','0'],['--maxseq','257'],['-n','2147483648'],['--gpu-cache-mb','0','--admit-decode'],['--token-ids','--token-ids']]):
  _,err=run('reject-'+str(i),common+flags,code=2);assert b'native_text_weights'not in err
 # Preserve all 23 committed-mode cases above; add explicit prediction mode.
 check_trace('predict-file',0,common[:-2],True,predict=True)
 check_trace('predict-stdin-pinned',0,['--prompt-file','-','--keep-special','--admit-decode','--pinned-staging'],False,input=prompt,predict=True)
 check_trace('predict-chunk513',1,['-p',prompt.decode(),'--prefill-tokens','513','--maxseq','1280'],True,predict=True)
 check_trace('predict-zero',0,common[:-2],True,predict=True,count=0)
 check_trace('predict-one',0,common[:-2],True,predict=True,count=1)
 check_trace('predict-bounded',0,common[:-2]+['--host-budget-mb',str(host_mb),'--device-budget-mb',str(device_mb)],True,predict=True)
 boundary=common[:-2]+['--maxseq','1280','--prepare-only']
 stdout,_=run('predict-boundary',boundary+['-n','255','--prediction-first']);boundary_plan=json.loads(stdout)
 assert boundary_plan['prepared']and boundary_plan['prompt_tokens']==1026 and boundary_plan['maxseq']==1280
 for name,flags in [('committed-boundary',['-n','255']),('predict-context',['-n','256','--prediction-first'])]:
  _,err=run(name,boundary+flags,code=1);assert b'exceeds maxseq'in err and b'native_text_weights'not in err
 trace=out/'predict-stdout-trace';trace.mkdir()
 _,err=run('predict-stdout-error',common+['--prediction-first','--trace-dir',str(trace)],code=1,sink=True)
 events=[json.loads(x)for x in(trace/'events.jsonl').read_text().splitlines()]
 assert len(events)==3 and all(e['kind']=='forward'for e in events)
 assert not(trace/'forward-3.i32le').exists()and b'generation/output'in err
 _,err=run('predict-duplicate',common+['--prediction-first','--prediction-first'],code=2);assert b'native_text_weights'not in err
 trace=out/'predict-collision-trace';trace.mkdir();protected=trace/'forward-0.i32le';protected.write_bytes(b'preserve input trace')
 stdout,err=run('predict-trace-collision',common+['--prediction-first','--trace-dir',str(trace)],code=1)
 assert not stdout and protected.read_bytes()==b'preserve input trace'and b'generation/output'in err
 check_trace('ram-admission-pinned',0,common[:-2]+['--ram-admit-misses','--ram-cache-mb','64','--pinned-staging'],True,predict=True)
 _,err=run('ram-admission-disabled',common+['--ram-admit-misses','--ram-cache-mb','0'],code=2)
 assert b'native_text_weights'not in err
print('CLI SOURCE CHECKS PASSED',len(results),flush=True)
