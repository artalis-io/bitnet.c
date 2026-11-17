from pathlib import Path
import json,subprocess,sys,shutil,importlib.util,time,resource,statistics
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
def inspect_model(model):
 import torch
 return dict(worker_pid=__import__('os').getpid(),named_parameter_pointers={name:int(t.data_ptr())for name,t in model.named_parameters()},parameter_tensors=sum(1 for _ in model.named_parameters()),parameter_storage_bytes=sum(t.numel()*t.element_size()for _,t in model.named_parameters()),tf32_allowed=torch.backends.cuda.matmul.allow_tf32,float32_matmul_precision=torch.get_float32_matmul_precision(),model_class=type(model).__module__+'.'+type(model).__name__,nonresident_parameters=[n for n,t in model.named_parameters()if t.numel()>4 and t.device.type!='cuda'],gpu=dict(name=torch.cuda.get_device_properties(0).name,total_memory=torch.cuda.get_device_properties(0).total_memory,compute_capability=list(torch.cuda.get_device_capability(0))),torch_version=torch.__version__,torch_cuda=torch.version.cuda)
def install_replay_counter(model):
 import torch
 cls=torch.cuda.CUDAGraph;assert not hasattr(cls,'_bn_probe_original');original=cls.replay
 cls._bn_probe_original=original;cls._bn_probe_count=0
 def replay(graph,*args,**kwargs):
  cls._bn_probe_count+=1
  return original(graph,*args,**kwargs)
 cls.replay=replay
 return dict(installed=True,metadata_only=True)
def stop_replay_counter(model):
 import torch
 cls=torch.cuda.CUDAGraph
 if not hasattr(cls,'_bn_probe_original'):return dict(installed=False)
 n=cls._bn_probe_count;cls.replay=cls._bn_probe_original;del cls._bn_probe_original;del cls._bn_probe_count
 return dict(replay_calls=n,restored=True)
def install_head_capture(model,capture_name):
 import torch,json,hashlib
 from pathlib import Path
 from vllm.v1.worker.gpu.model_runner import GPUModelRunner
 p=Path('/evidence')/capture_name;p.mkdir();(p/'capture').mkdir();state=dict(prompts={},scheduled=[],heads=[]);original_prepare=GPUModelRunner.prepare_inputs;original_logits=model.compute_logits
 def prepare(runner,schedule,*args,**kwargs):
  for request in schedule.scheduled_new_reqs:state['prompts'][request.req_id]=list(request.prompt_token_ids)
  batch=original_prepare(runner,schedule,*args,**kwargs)
  state['scheduled'].append(dict(idx_mapping=batch.idx_mapping_np.tolist(),seq_lens_cpu_upper_bound=batch.seq_lens_cpu_upper_bound.tolist(),num_tokens=batch.num_tokens,num_tokens_after_padding=batch.num_tokens_after_padding,num_reqs_after_padding=batch.num_reqs_after_padding,query_start_loc=batch.query_start_loc_np.tolist(),req_ids=list(batch.req_ids),computed=runner.req_states.num_computed_tokens_np[batch.idx_mapping_np].tolist(),scheduled=[int(schedule.num_scheduled_tokens[r])for r in batch.req_ids]))
  return batch
 def logits(*args,**kwargs):
  out=original_logits(*args,**kwargs)
  if out is not None:
   index=len(state['heads']);geometry=state['scheduled'][index];assert out.dtype==torch.bfloat16 and list(out.shape)==[len(geometry['req_ids']),248320]
   data=out.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes();name=f'logits-{index}.bin';(p/'capture'/name).write_bytes(data)
   state['heads'].append(dict(file=name,sha256=hashlib.sha256(data).hexdigest(),argmax=out.detach().float().argmax(-1).cpu().tolist(),geometry=geometry))
  return out
 GPUModelRunner.prepare_inputs=prepare;model.compute_logits=logits
 def stop():
  assert GPUModelRunner.prepare_inputs is prepare and model.compute_logits is logits
  GPUModelRunner.prepare_inputs=original_prepare;model.compute_logits=original_logits
  assert len(state['heads'])==len(state['scheduled'])>0
  (p/'raw-logits.json').write_text(json.dumps(state,indent=2));return dict(head_batches=len(state['heads']),observers_restored=True)
 model._bn_stop_heads=stop
 return dict(passive_cpu_geometry=True,logits_readback=True,forward_hooks=False,arithmetic_changes=False)
def stop_head_capture(model):return model._bn_stop_heads()

def main():
 p=Path('/evidence')
 subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps','/wheels/vllm_gguf_plugin-0.0.5+e454affe-cp310-abi3-linux_x86_64.whl'],check=True)
 root=Path(importlib.util.find_spec('vllm_gguf_plugin').origin).parent;shutil.copy2('/experts/plugin-512/vllm_gguf_plugin/_C_gguf.abi3.so',root/'_C_gguf.abi3.so')
 adapter=root/'weights_adapter/qwen3_5.py';text=adapter.read_text();old='"qwen3_5_text": "Qwen3_5ForConditionalGeneration"';assert text.count(old)==1;adapter.write_text(text.replace(old,'"qwen3_5_text": "Qwen3_5ForCausalLM"'))
 from vllm import LLM,SamplingParams
 fixture=json.loads(Path('/repo/test/bf16_cuda_resident46_reference.json').read_text());args=dict(fixture['engine_request']);args.pop('compilation_config',None);args.pop('async_scheduling',None)
 (p/'engine-request.json').write_text(json.dumps(args,indent=2));llm=LLM(**args);engine=llm.llm_engine
 config=engine.vllm_config
 resolved=dict(core_client=type(engine.engine_core).__name__,cudagraph_mode=str(config.compilation_config.cudagraph_mode),compilation_mode=str(config.compilation_config.mode),async_scheduling=config.scheduler_config.async_scheduling,full_config=str(config))
 (p/'resolved-config.json').write_text(json.dumps(resolved,indent=2));(p/'model-inspection.json').write_text(json.dumps(llm.apply_model(inspect_model),indent=2));print('DEFAULT_MODEL_READY',resolved,flush=True)
 params=SamplingParams(temperature=0,max_tokens=64,ignore_eos=True,detokenize=False)
 def batch(label,max_outputs=64):
  params=SamplingParams(temperature=0,max_tokens=max_outputs,ignore_eos=True,detokenize=False)
  assert not engine.has_unfinished_requests();client=engine.engine_core;mp=hasattr(client,'call_utility');assigned=[];tokens=[[]for _ in range(4)];delivery=[[]for _ in range(4)];calls=[];metrics={}
  if mp:client.call_utility('pause_scheduler','keep',False)
  for i in range(4):assigned.append(engine.add_request(label+'-'+str(i),{'prompt_token_ids':fixture['groups']['static'][str(i)]['input_ids']},params))
  if mp:assert client.call_utility('is_scheduler_paused')
  released=time.perf_counter_ns()
  if mp:client.call_utility('resume_scheduler')
  for call in range(512):
   if not engine.has_unfinished_requests():break
   before=time.perf_counter_ns();outputs=engine.step();after=time.perf_counter_ns();records=[]
   for output in outputs:
    rid=output.request_id;i=int(rid[len(label)+1:])if rid.startswith(label+'-')else assigned.index(rid);ids=list(output.outputs[0].token_ids);prior=len(tokens[i]);assert ids[:prior]==tokens[i]and len(ids)<=max_outputs;at=time.perf_counter_ns()
    delivery[i].extend([at]*(len(ids)-prior));tokens[i]=ids;records.append(dict(index=i,generated=len(ids),new_tokens=len(ids)-prior,finished=output.finished))
    stats=output.metrics
    if stats is not None:metrics[str(i)]={k:getattr(stats,k,None)for k in ['queued_ts','scheduled_ts','first_token_ts','last_token_ts','num_generation_tokens']}
   calls.append(dict(start_ns=before,end_ns=after,outputs=records))
  else:raise AssertionError('step bound')
  assert all(len(v)==max_outputs for v in tokens)and all(len(v)==max_outputs for v in delivery)
  first=next(v for v in calls if v['outputs']);assert {v['index']for v in first['outputs']}==set(range(4)),'Four-request first wave required'
  return dict(tokens=tokens,delivery=delivery,calls=calls,metrics=metrics,barrier='pause_scheduler keep, no cache clear, bulk add, resume'if mp else'in-process adds before first step',prefill_delivery_ms=(max(v[0]for v in delivery)-released)/1e6,decode_delivery_mean_ms=statistics.mean((v[-1]-v[0])/((max_outputs-1)*1e6) for v in delivery))
 try:
  warm=batch('warmup');(p/'warmup.json').write_text(json.dumps(warm,indent=2));print('DEFAULT_WARMUP_COMPLETE',flush=True);runs=[]
  for i in range(3):
   v=batch('timed-'+str(i));assert v['tokens']==warm['tokens'];runs.append(v);(p/'timing-runs.json').write_text(json.dumps(runs,indent=2));print('DEFAULT_TIMING',i,v['prefill_delivery_ms'],v['decode_delivery_mean_ms'],flush=True)
  from functools import partial
  for capture_name,length in [('repeat64-A',64),('repeat64-B',64),('long96-A',96),('repeat64-C',64),('long96-B',96)]:
   (p/(capture_name+'-installed.json')).write_text(json.dumps(llm.apply_model(partial(install_head_capture,capture_name=capture_name)),indent=2))
   try:
    captured=batch(capture_name,length);assert all(a[:64]==b for a,b in zip(captured['tokens'],warm['tokens']));(p/(capture_name+'-output.json')).write_text(json.dumps(captured,indent=2));print('REPEAT_CAPTURE_COMPLETE',capture_name,length,flush=True)
   finally:(p/(capture_name+'-status.json')).write_text(json.dumps(llm.apply_model(stop_head_capture),indent=2))
  (p/'counter-installed.json').write_text(json.dumps(llm.apply_model(install_replay_counter),indent=2))
  try:
   diagnostic=batch('replay-diagnostic');assert diagnostic['tokens']==warm['tokens'];(p/'replay-diagnostic.json').write_text(json.dumps(diagnostic,indent=2))
  finally:(p/'replay-counter.json').write_text(json.dumps(llm.apply_model(stop_replay_counter),indent=2))
  summary=dict(resolved=resolved,prefill_delivery_median_ms=statistics.median(v['prefill_delivery_ms']for v in runs),decode_delivery_median_ms=statistics.median(v['decode_delivery_mean_ms']for v in runs),timing_observers=False,diagnostic_counter_after_timings=True,token_repeatability=True,cross_eager_profile_tokens_identical=warm['tokens']==[fixture['groups']['static'][str(i)]['tokens']for i in range(4)])
  (p/'summary.json').write_text(json.dumps(summary,indent=2));print('DEFAULT_REFERENCE_COMPLETE',flush=True)
 finally:engine.engine_core.shutdown()
if __name__=='__main__':main()
