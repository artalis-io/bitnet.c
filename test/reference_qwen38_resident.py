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
    import resource
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps','/wheels/vllm_gguf_plugin-0.0.5+e454affe-cp310-abi3-linux_x86_64.whl'],check=True)
    root=Path(importlib.util.find_spec('vllm_gguf_plugin').origin).parent
    shutil.copy2('/experts/plugin-512/vllm_gguf_plugin/_C_gguf.abi3.so',root/'_C_gguf.abi3.so')
    adapter=root/'weights_adapter/qwen3_5.py';text=adapter.read_text();old='"qwen3_5_text": "Qwen3_5ForConditionalGeneration"'
    assert text.count(old)==1;adapter.write_text(text.replace(old,'"qwen3_5_text": "Qwen3_5ForCausalLM"'))
    from vllm import LLM,SamplingParams
    import torch
    assert torch.cuda.device_count()==1
    fixture=json.loads(Path('/repo/test/bf16_cuda_packed_traced_reference.json').read_text());args=dict(fixture['engine_request']);args.update(max_num_scheduled_tokens=46,prefill_policy='decode-aware',long_prefill_token_threshold=129,max_parallel_prefills=4)
    (p/'engine-request.json').write_text(json.dumps(args,indent=2));llm=LLM(**args)
    def params(n):return SamplingParams(temperature=0,max_tokens=n,ignore_eos=True,detokenize=False)
    def run(label):
        engine=llm.llm_engine;assert not engine.has_unfinished_requests();result={};calls=[];assigned={}
        def add(name,key,n):
            public=label+'-'+name;assigned[name]=engine.add_request(public,{'prompt_token_ids':fixture['streams'][key]['input_ids']},params(n));return public
        def step():
            outputs=engine.step();record=[]
            for output in outputs:
                public=output.request_id
                if public.startswith(label+'-'):name=public[len(label)+1:]
                else:
                    names=[name for name,id in assigned.items()if id==public];assert len(names)==1,(public,assigned);name=names[0]
                tokens=list(output.outputs[0].token_ids);assert tokens[:len(result.get(name,[]))]==result.get(name,[])
                result[name]=tokens;record.append(dict(name=name,tokens=tokens,finished=output.finished))
            calls.append(record)
        add('A',0,8);add('B',1,8);add('C',2,8);step();assert all(len(result[name])==1 for name in ['A','B','C'])
        add('D',3,3)
        for _ in range(32):
            if not engine.has_unfinished_requests():break
            step()
        else:raise AssertionError('drain bound')
        assert all(len(result[name])==8 for name in ['A','B','C'])and len(result['D'])==3
        return dict(tokens=result,steps=calls,assigned=assigned)
    def batch(label):
        engine=llm.llm_engine;assert not engine.has_unfinished_requests();assigned=[];tokens=[[]for _ in range(4)];delivery=[[]for _ in range(4)];steps=[]
        started=time.perf_counter_ns()
        for i in range(4):assigned.append(engine.add_request(label+'-'+str(i),{'prompt_token_ids':fixture['streams'][i]['input_ids']},params(64)))
        for step in range(128):
            if not engine.has_unfinished_requests():break
            before=time.perf_counter_ns();outputs=engine.step();after=time.perf_counter_ns();assert len(outputs)==4
            record=[]
            for output in outputs:
                public=output.request_id
                i=int(public[len(label)+1:])if public.startswith(label+'-')else assigned.index(public)
                ids=list(output.outputs[0].token_ids);assert len(ids)==step+1 and ids[:-1]==tokens[i]
                tokens[i]=ids;delivery[i].append(time.perf_counter_ns());record.append(dict(index=i,tokens=ids,finished=output.finished))
            steps.append(dict(start_ns=before,end_ns=after,outputs=record))
        else:raise AssertionError('batch drain bound')
        assert len(steps)==64 and all(len(v)==64 for v in tokens)
        return dict(tokens=tokens,assigned=assigned,steps=steps,delivery=delivery,admission_inclusive_prefill_ms=(steps[0]['end_ns']-started)/1e6,engine_prefill_ms=(steps[0]['end_ns']-steps[0]['start_ns'])/1e6,decode_engine_ms=[(v['end_ns']-v['start_ns'])/1e6 for v in steps[1:]],decode_delivery_intervals_ms=[(b-a)/1e6 for ds in delivery for a,b in zip(ds,ds[1:])])
    try:
        warm=llm.generate([{'prompt_token_ids':s['input_ids']}for s in fixture['streams']],params(5),use_tqdm=False)
        (p/'warm-tokens.json').write_text(json.dumps([list(v.outputs[0].token_ids)for v in warm],indent=2))
        control=run('control');(p/'control.json').write_text(json.dumps(control,indent=2));print('DYNAMIC_CONTROL_READY',flush=True)
        static_control=batch('control-static');(p/'static-control.json').write_text(json.dumps(static_control,indent=2))
        (p/'model-inspection.json').write_text(json.dumps(llm.apply_model(install_capture),indent=2))
        observed=run('observed');(p/'observed.json').write_text(json.dumps(observed,indent=2));assert observed['tokens']==control['tokens']
        static_observed=batch('observed-static');assert static_observed['tokens']==static_control['tokens'];(p/'static-observed.json').write_text(json.dumps(static_observed,indent=2))
        (p/'capture-status.json').write_text(json.dumps(llm.apply_model(stop_capture),indent=2))
        timing_warm=batch('timing-warmup');assert timing_warm['tokens']==static_control['tokens'];(p/'timing-warmup.json').write_text(json.dumps(timing_warm,indent=2))
        timings=[]
        for i in range(3):
            timed=batch('timed-'+str(i));assert timed['tokens']==static_control['tokens'];timings.append(timed);(p/'timing-runs.json').write_text(json.dumps(timings,indent=2));print('RESIDENT_BATCH_TIMING',i,flush=True)
        props=torch.cuda.get_device_properties(0)
        (p/'hardware.json').write_text(json.dumps(dict(name=props.name,multiprocessors=props.multi_processor_count,total_memory=props.total_memory,compute_capability=[props.major,props.minor],torch_cuda=torch.version.cuda,torch_version=torch.__version__),indent=2))
        print('ORIGINAL_DYNAMIC_CAPTURE_COMPLETE',flush=True)
    finally:llm.llm_engine.engine_core.shutdown()

if __name__=='__main__':main()
