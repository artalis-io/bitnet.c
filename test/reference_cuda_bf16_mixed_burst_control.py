from pathlib import Path
import json,time,subprocess,sys,shutil,importlib.util,hashlib


def install_capture(model):
    import torch
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    p=Path('/evidence');(p/'kernels').mkdir(exist_ok=True);(p/'capture').mkdir(exist_ok=True)
    weights={t.data_ptr():n for n,t in model.named_parameters()}
    state={'call':-1,'forwards':[],'norms':[],'logits':[],'request_prompts':{},'gdn_dispatch':[]}
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    original_prepare=GPUModelRunner.prepare_inputs
    def prepare(runner,scheduler_output,*args,**kwargs):
        for req in scheduler_output.scheduled_new_reqs:state['request_prompts'][req.req_id]=list(req.prompt_token_ids)
        batch=original_prepare(runner,scheduler_output,*args,**kwargs)
        state['runner']=runner;state['batch']=batch;state['schedule']=scheduler_output
        state['computed']=runner.req_states.num_computed_tokens_np[batch.idx_mapping_np].tolist()
        return batch
    GPUModelRunner.prepare_inputs=prepare
    original_logits=model.compute_logits;original_run=CachingAutotuner.run
    import flashinfer.gdn_prefill as fi
    original_dispatch={name:getattr(fi,name)for name in ['cp_delta_rule_dsl_sm120','chunk_gated_delta_rule_sm120']}
    for name,fn in original_dispatch.items():
        def dispatch(*args,_name=name,_fn=fn,**kwargs):
            cp=_name.startswith('cp_');cu=args[7 if cp else 8]
            state['gdn_dispatch'].append(dict(call=state['call'],operator=_name,tokens=int(args[2].shape[0]),sequences=int(cu.numel()-1),heads=int(args[2].shape[1])))
            return _fn(*args,**kwargs)
        setattr(fi,name,dispatch)
    def before(module,args,kwargs):
        ids=kwargs.get('input_ids',args[0]if args else None);pos=kwargs.get('positions',args[1]if len(args)>1 else None)
        state['call']+=1
        req_ids=list(state['batch'].req_ids);computed=state['computed'];scheduled=[int(state['schedule'].num_scheduled_tokens[r])for r in req_ids]
        trace=dict(input_ids=ids.detach().cpu().tolist(),positions=pos.detach().cpu().tolist(),req_ids=req_ids,computed=computed,scheduled=scheduled)
        assert sum(scheduled)==len(trace['input_ids'])
        assert trace['positions']==sum([list(range(int(c),int(c)+n))for c,n in zip(computed,scheduled)],[])
        state['forwards'].append(trace)
    handle=model.register_forward_pre_hook(before,with_kwargs=True)
    def record_kernel(self,*args,**kwargs):
        result=original_run(self,*args,**kwargs)
        if 'rms_norm' in self.fn.__name__:
            source=self.fn.src;key=hashlib.sha256(source.encode()).hexdigest();f=p/'kernels'/(key+'.py')
            if not f.exists():f.write_text(source)
            launcher=self.launchers[0]
            named={str(i):weights[t.data_ptr()]for i,t in enumerate(args)if isinstance(t,torch.Tensor)and t.data_ptr()in weights}
            state['norms'].append(dict(call=state['call'],kernel=self.fn.__name__,source=key,config=launcher.config.kwargs,num_warps=launcher.config.num_warps,weights=named,cache_hash=launcher.cache_hash))
        return result
    CachingAutotuner.run=record_kernel
    def logits(*args,**kwargs):
        out=original_logits(*args,**kwargs)
        if out is not None:
            assert out.dtype==torch.bfloat16 and list(out.shape)==[len(state['forwards'][-1]['req_ids']),248320],(out.dtype,out.shape)
            data=out.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes();i=len(state['logits']);(p/'capture'/f'logits-{i}.bin').write_bytes(data)
            state['logits'].append(dict(file=f'logits-{i}.bin',sha256=hashlib.sha256(data).hexdigest(),tokens=out.detach().float().argmax(-1).cpu().tolist(),request_rows=state['forwards'][-1]))
        return out
    model.compute_logits=logits
    def stop():
        handle.remove();model.compute_logits=original_logits;CachingAutotuner.run=original_run;GPUModelRunner.prepare_inputs=original_prepare
        (p/'scheduler-request-prompts.json').write_text(json.dumps(state['request_prompts'],indent=2));(p/'forward-trace.json').write_text(json.dumps(state['forwards'],indent=2));(p/'norm-launches.jsonl').write_text(''.join(json.dumps(x)+'\n'for x in state['norms']));(p/'raw-logits.json').write_text(json.dumps(state['logits'],indent=2))
        for name,fn in original_dispatch.items():setattr(fi,name,fn)
        (p/'gdn-dispatch.json').write_text(json.dumps(state['gdn_dispatch'],indent=2))
        assert len(state['forwards'])==len(state['logits'])>0
        assert len(state['norms'])==129*len(state['forwards'])
        return dict(forwards=len(state['forwards']),norms=len(state['norms']),raw_logits=len(state['logits']),hooks_removed=True)
    model._bitnet_perf_stop=stop
    nonresident=[n for n,t in model.named_parameters()if t.device.type!='cuda'and t.numel()>4]
    assert not nonresident,nonresident
    return dict(model_class=type(model).__module__+'.'+type(model).__name__,nonresident_parameters=nonresident)


def stop_capture(model):return model._bitnet_perf_stop()



def install_schedule_capture(model):
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    original=GPUModelRunner.prepare_inputs;state=dict(prompts={},trace=[])
    def prepare(runner,scheduled,*args,**kwargs):
        for req in scheduled.scheduled_new_reqs:state['prompts'][req.req_id]=list(req.prompt_token_ids)
        batch=original(runner,scheduled,*args,**kwargs)
        state['trace'].append(dict(req_ids=list(batch.req_ids),computed=runner.req_states.num_computed_tokens_np[batch.idx_mapping_np].tolist(),scheduled=[int(scheduled.num_scheduled_tokens[r])for r in batch.req_ids]))
        return batch
    GPUModelRunner.prepare_inputs=prepare
    def stop():
        assert GPUModelRunner.prepare_inputs is prepare
        GPUModelRunner.prepare_inputs=original
        return state
    model._bitnet_schedule_stop=stop
    return dict(cpu_only_geometry_observer=True,gpu_tensor_readbacks=False)


def stop_schedule_capture(model):return model._bitnet_schedule_stop()

def fixed_native_share(bulk_budget, mixed_budget, max_live, max_prefills, max_chunk,
                       decodes, pending):
    budget = mixed_budget if decodes and mixed_budget else bulk_budget
    lanes = min(pending, max_prefills, max_live - decodes, budget - decodes)
    return min(max_chunk, (budget - decodes) // lanes) if lanes > 0 else 0


def main():
    p=Path('/evidence')
    subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps','/wheels/vllm_gguf_plugin-0.0.5+e454affe-cp310-abi3-linux_x86_64.whl'],check=True)
    root=Path(importlib.util.find_spec('vllm_gguf_plugin').origin).parent
    shutil.copy2('/experts/plugin-512/vllm_gguf_plugin/_C_gguf.abi3.so',root/'_C_gguf.abi3.so')
    adapter=root/'weights_adapter/qwen3_5.py';text=adapter.read_text();old='"qwen3_5_text": "Qwen3_5ForConditionalGeneration"'
    assert text.count(old)==1;adapter.write_text(text.replace(old,'"qwen3_5_text": "Qwen3_5ForCausalLM"'))
    from vllm import LLM,SamplingParams
    # Numerical control only: FIFO lane selection reproduces the native
    # cohort contract. Model/worker math, sampling and KV management are original.
    from vllm.v1.core.sched.prefill_interleave import PrefillInterleaveController
    import inspect
    import types
    control_limits=json.loads((p/'input-plan.json').read_text())
    original_begin=PrefillInterleaveController.begin_step
    (p/'original-prefill-interleave.py').write_text(inspect.getsource(original_begin))
    def fifo_begin(self,*args,**kwargs):
        saved=self.last_scheduled_step
        self.last_scheduled_step={}
        kwargs['policy']='round-robin'
        try:step=original_begin(self,*args,**kwargs)
        finally:self.last_scheduled_step=saved
        if step is None:return None
        decodes=kwargs['num_runnable_decodes']
        assert 0<=decodes<=control_limits['max_live']
        step._control_fixed_share=fixed_native_share(
            control_limits['token_budget'],control_limits.get('mixed_token_budget',0),
            control_limits['max_live'],kwargs['max_parallel_prefills'],
            control_limits['engine_request']['long_prefill_token_threshold'],
            decodes,len(step.selected_ids))
        def frozen_budget(this,total_token_budget,scheduled_prefill_ids):
            return min(this._control_fixed_share,total_token_budget)
        step.token_budget=types.MethodType(frozen_budget,step)
        return step
    PrefillInterleaveController.begin_step=fifo_begin
    (p/'reference-control-policy.json').write_text(json.dumps(dict(scope='Numerical model-math control only; not original scheduler or timing',adapter='FIFO lanes with fixed floor shares, one token reserved per runnable decode and no share redistribution; public C planner contract, independently calibrated without model outputs',original_begin_sha256=hashlib.sha256(inspect.getsource(original_begin).encode()).hexdigest()),indent=2))
    import torch
    assert torch.cuda.device_count()==1
    fixture=json.loads((p/'input-plan.json').read_text());args=fixture['engine_request'];streams=fixture['streams']
    (p/'engine-request.json').write_text(json.dumps(args,indent=2));llm=LLM(**args)
    params=SamplingParams(temperature=0,max_tokens=fixture['outputs_each'],ignore_eos=True,detokenize=False)
    def run(label):
        engine=llm.llm_engine;assert not engine.has_unfinished_requests()
        result=[[]for _ in streams];delivery=[[]for _ in streams];arrival=[None]*len(streams)
        added=set();finished=set();steps=[];events=[];start=time.perf_counter()
        for step in range(4096):
            for i,stream in enumerate(streams):
                if stream['arrival_step']!=step:continue
                public=f'{label}-{i}';arrival[i]=(time.perf_counter()-start)*1000
                assigned=engine.add_request(public,{'prompt_token_ids':stream['input_ids']},params)
                added.add(i);events.append(dict(index=i,arrival_step=step,arrival_ms=arrival[i],public_id=public,assigned_id=assigned))
            if not engine.has_unfinished_requests():
                if len(finished)==len(streams):break
                raise AssertionError(('idle gap',step,finished,added))
            begin=time.perf_counter();outputs=engine.step();end=time.perf_counter()
            steps.append(dict(step=step,start_ms=(begin-start)*1000,end_ms=(end-start)*1000,duration_ms=(end-begin)*1000))
            for output in outputs:
                matches=[i for i,s in enumerate(streams)if s['input_ids']==output.prompt_token_ids]
                assert len(matches)==1,(output.request_id,matches);i=matches[0];assert i in added
                ids=list(output.outputs[0].token_ids);assert ids[:len(result[i])]==result[i]
                new=len(ids)-len(result[i]);assert new in (0,1),(step,i,new)
                if new:delivery[i].append((time.perf_counter()-start)*1000)
                result[i]=ids
                if output.finished:assert len(ids)==fixture['outputs_each'];finished.add(i)
            if len(finished)==len(streams):break
        else:raise AssertionError('step bound')
        assert len(added)==len(finished)==len(streams) and all(len(x)==fixture['outputs_each'] for x in result)
        return dict(label=label,tokens=result,arrival_events=events,delivery_ms=delivery,engine_steps=steps,elapsed_ms=(time.perf_counter()-start)*1000)
    try:
        control=run('control');(p/'control.json').write_text(json.dumps(control,indent=2));print('STAGGER_CONTROL_READY',len(control['engine_steps']),flush=True)
        (p/'model-inspection.json').write_text(json.dumps(llm.apply_model(install_capture),indent=2))
        observed=run('observed');assert observed['tokens']==control['tokens']
        (p/'capture-status.json').write_text(json.dumps(llm.apply_model(stop_capture),indent=2));(p/'observed.json').write_text(json.dumps(observed,indent=2))
        actual=json.loads((p/'scheduler-request-prompts.json').read_text());mapping={}
        for req,ids in actual.items():
            matches=[i for i,s in enumerate(streams)if s['input_ids']==ids];assert len(matches)==1;mapping[req]=matches[0]
        assert len(mapping)==len(streams) and set(mapping.values())==set(range(len(streams)))
        raw=json.loads((p/'raw-logits.json').read_text());trace=json.loads((p/'forward-trace.json').read_text())
        assert len(raw)==len(trace)==len(observed['engine_steps'])==len(control['engine_steps'])
        collected=[{}for _ in streams]
        for call,matrix in enumerate(raw):
            meta=matrix['request_rows'];offset=0
            for row,req in enumerate(meta['req_ids']):
                i=mapping[req];computed=int(meta['computed'][row]);n=meta['scheduled'][row]
                assert streams[i]['arrival_step']<=call
                assert meta['input_ids'][offset:offset+n]==(streams[i]['input_ids']+control['tokens'][i])[computed:computed+n]
                offset+=n;output_step=computed+n-len(streams[i]['input_ids'])
                if output_step>=0:
                    assert output_step<fixture['outputs_each'] and output_step not in collected[i]
                    collected[i][output_step]=matrix['tokens'][row]
        assert [[x[i]for i in range(fixture['outputs_each'])]for x in collected]==control['tokens']
        (p/'request-row-map.json').write_text(json.dumps(mapping,indent=2))
        (p/'identity-proof.json').write_text(json.dumps(dict(full_scheduler_prompt_binding=True,every_actual_input_token_bound=True,all_raw_sampled_argmax_match_api=True,sampled_heads=len(streams)*fixture['outputs_each'],model_norm_dispatch_hooks_removed_before_timing=True,cpu_geometry_observer_in_timed_runs=True),indent=2))
        timings=[]
        for i in range(0):
            llm.apply_model(install_schedule_capture)
            timed=run(f'timed-{i}');assert timed['tokens']==control['tokens']
            metadata=llm.apply_model(stop_schedule_capture);assert len(metadata)==1;metadata=metadata[0]
            timed_map={}
            for req,ids in metadata['prompts'].items():
                matches=[j for j,s in enumerate(streams)if s['input_ids']==ids];assert len(matches)==1;timed_map[req]=matches[0]
            assert len(metadata['trace'])==len(trace)
            geometry=[]
            for captured,actual in zip(trace,metadata['trace']):
                ids=[timed_map[req]for req in actual['req_ids']]
                assert ids==[mapping[req]for req in captured['req_ids']] and actual['computed']==captured['computed'] and actual['scheduled']==captured['scheduled']
                geometry.append(dict(request_ids=ids,computed=actual['computed'],scheduled=actual['scheduled']))
            timed['step_geometry']=geometry;timed['cpu_geometry_observer']=True;timed['gpu_model_or_tensor_observer']=False
            assert len(timed['engine_steps'])==len(trace)
            timings.append(timed);(p/'timings.json').write_text(json.dumps(timings,indent=2))
            print('STAGGER_TIMING',i,round(timed['elapsed_ms'],3),flush=True)
        props=torch.cuda.get_device_properties(0)
        (p/'hardware.json').write_text(json.dumps(dict(name=props.name,multiprocessors=props.multi_processor_count,total_memory=props.total_memory,compute_capability=[props.major,props.minor],torch_cuda=torch.version.cuda),indent=2))
        print('STAGGERED_REFERENCE_CAPTURE_QUALIFIED',flush=True)
    finally:llm.llm_engine.engine_core.shutdown()


if __name__=='__main__':main()
