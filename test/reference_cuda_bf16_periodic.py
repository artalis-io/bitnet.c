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

def main():
    p=Path('/evidence')
    subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps','/wheels/vllm_gguf_plugin-0.0.5+e454affe-cp310-abi3-linux_x86_64.whl'],check=True)
    root=Path(importlib.util.find_spec('vllm_gguf_plugin').origin).parent
    shutil.copy2('/experts/plugin-512/vllm_gguf_plugin/_C_gguf.abi3.so',root/'_C_gguf.abi3.so')
    adapter=root/'weights_adapter/qwen3_5.py';text=adapter.read_text();old='"qwen3_5_text": "Qwen3_5ForConditionalGeneration"'
    assert text.count(old)==1;adapter.write_text(text.replace(old,'"qwen3_5_text": "Qwen3_5ForCausalLM"'))
    from vllm import LLM,SamplingParams
    import torch
    assert torch.cuda.device_count()==1
    fixture=json.loads((p/'input-plan.json').read_text());args=fixture['engine_request'];streams=fixture['streams']
    (p/'engine-request.json').write_text(json.dumps(args,indent=2));llm=LLM(**args)
    params=SamplingParams(temperature=0,max_tokens=fixture['outputs_each'],ignore_eos=True,detokenize=False)
    def run(label):
        engine=llm.llm_engine;assert not engine.has_unfinished_requests()
        result=[[]for _ in streams];delivery=[[]for _ in streams]
        added=set();finished=set();steps=[];events=[];start=time.perf_counter()
        offers=fixture['offered_arrival_ms']
        while len(finished)<len(streams):
            elapsed=(time.perf_counter()-start)*1000
            for i,stream in enumerate(streams):
                if i in added or offers[i]>elapsed:continue
                public=f'{label}-{i}';enqueue=(time.perf_counter()-start)*1000
                assigned=engine.add_request(public,{'prompt_token_ids':stream['input_ids']},params)
                added.add(i);events.append(dict(index=i,offered_ms=offers[i],worker_enqueue_ms=enqueue,enqueue_step=len(steps),public_id=public,assigned_id=assigned))
            if not engine.has_unfinished_requests():
                pending=[offers[i]for i in range(len(streams))if i not in added]
                assert pending
                time.sleep(min(.01,max(0,(min(pending)-(time.perf_counter()-start)*1000)/1000)))
                continue
            assert len(steps)<4096
            begin=time.perf_counter();outputs=engine.step();end=time.perf_counter()
            steps.append(dict(step=len(steps),start_ms=(begin-start)*1000,end_ms=(end-start)*1000,duration_ms=(end-begin)*1000))
            for output in outputs:
                matches=[i for i,s in enumerate(streams)if s['input_ids']==output.prompt_token_ids]
                assert len(matches)==1;i=matches[0];assert i in added
                ids=list(output.outputs[0].token_ids);assert ids[:len(result[i])]==result[i]
                new=len(ids)-len(result[i]);assert new in(0,1)
                if new:delivery[i].append((time.perf_counter()-start)*1000)
                result[i]=ids
                if output.finished:assert len(ids)==fixture['outputs_each'];finished.add(i)
        assert len(added)==len(finished)==len(streams) and all(len(x)==fixture['outputs_each'] for x in result)
        return dict(label=label,tokens=result,arrival_events=events,delivery_ms=delivery,engine_steps=steps,elapsed_ms=(time.perf_counter()-start)*1000)
    try:
        warm=run('warmup');(p/'warmup.json').write_text(json.dumps(warm,indent=2));print('WALL_CLOCK_WARMUP',len(warm['engine_steps']),flush=True)
        timings=[]
        for i in range(3):
            llm.apply_model(install_schedule_capture)
            timed=run(f'wall-{i}')
            metadata=llm.apply_model(stop_schedule_capture);assert len(metadata)==1;metadata=metadata[0]
            mapping={}
            for req,ids in metadata['prompts'].items():
                matches=[j for j,s in enumerate(streams)if s['input_ids']==ids];assert len(matches)==1;mapping[req]=matches[0]
            assert len(mapping)==len(streams) and len(metadata['trace'])==len(timed['engine_steps'])
            geometry=[]
            for frame in metadata['trace']:
                ids=[mapping[req]for req in frame['req_ids']]
                assert len(ids)<=8 and sum(frame['scheduled'])<=256
                geometry.append(dict(request_ids=ids,computed=frame['computed'],scheduled=frame['scheduled']))
            timed['step_geometry']=geometry;timed['cpu_geometry_observer']=True;timed['gpu_model_or_tensor_observer']=False
            timings.append(timed);(p/'timings.json').write_text(json.dumps(timings,indent=2))
            print('FIXED_WALL_CLOCK_TIMING',i,round(timed['elapsed_ms'],3),len(geometry),flush=True)
        props=torch.cuda.get_device_properties(0)
        (p/'hardware.json').write_text(json.dumps(dict(name=props.name,multiprocessors=props.multi_processor_count,total_memory=props.total_memory,compute_capability=[props.major,props.minor],torch_cuda=torch.version.cuda),indent=2))
        print('REFERENCE_WALL_CLOCK_TIMINGS_COMPLETE',flush=True)
    finally:llm.llm_engine.engine_core.shutdown()

if __name__=='__main__':main()
