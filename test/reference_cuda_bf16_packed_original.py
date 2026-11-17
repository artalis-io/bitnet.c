from pathlib import Path
import json,time,subprocess,sys,shutil,importlib.util,hashlib

def install_capture(model):
    import torch,inspect
    from pathlib import Path
    from vllm.forward_context import get_forward_context
    p=Path('/evidence');trace=[];records=[];state={'run':'control','calls':0,'captured':False,'current':None,'selected':False,'logits':{'control':[],'observed':[]}}
    def save(name,t):
        data=t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes();f=p/'capture'/name;f.parent.mkdir(exist_ok=True)
        if f.exists():assert f.read_bytes()==data,name
        else:f.write_bytes(data)
        row=dict(file=name,dtype=str(t.dtype),shape=list(t.shape),bytes=len(data),sha256=hashlib.sha256(data).hexdigest());records.append(row)
        (p/'capture/manifest.json').write_text(json.dumps(records,indent=2));return row
    def md_summary(md):
        d={'class':type(md).__module__+'.'+type(md).__name__}
        for k,v in vars(md).items():
            if isinstance(v,(str,int,float,bool,type(None))):d[k]=v
            elif isinstance(v,torch.Tensor):
                d[k]={'dtype':str(v.dtype),'shape':list(v.shape)}
                if v.numel()<=2048:d[k]['values']=v.detach().cpu().tolist()
        return d
    def metadata_for(mds,layer):
        matches=[(k,v)for k,v in mds.items()if f'.layers.{layer}.'in k]
        assert len(matches)==1,[(k,type(v).__name__)for k,v in matches]
        return matches[0]
    def snapshot(root,mds,n):
        layers=[]
        for i,l in enumerate(model.model.layers):
            key,md=metadata_for(mds,i)
            if hasattr(l,'linear_attn'):
                module=l.linear_attn;cache=module.kv_cache
                indices=md.non_spec_state_indices_tensor[:n].reshape(-1).to(torch.int64)
                assert indices.numel()==n and torch.all(indices>=0)
                history=cache[0].index_select(0,indices)
                if history.shape[1:]==(10240,3):history=history.transpose(1,2)
                assert history.shape[1:]==(3,10240),history.shape
                recurrent=cache[1].index_select(0,indices);assert recurrent.shape[1:]==(48,128,128)
                save(f'{root}-l{i}-history.bin',history);save(f'{root}-l{i}-state.bin',recurrent)
                layers.append(dict(layer=i,kind='gdn',metadata_key=key,indices=indices.cpu().tolist(),cache_shapes=[list(t.shape)for t in cache]))
            else:
                module=l.self_attn.attn;cache=module.kv_cache
                assert isinstance(cache,torch.Tensor),type(cache)
                table=md.block_table[:n];indices=torch.unique(table.reshape(-1)).to(torch.int64)
                indices=indices[indices>=0];assert torch.all(indices<cache.shape[0]),(indices,cache.shape)
                save(f'{root}-l{i}-cache-blocks.bin',cache.index_select(0,indices))
                layers.append(dict(layer=i,kind='attention',metadata_key=key,blocks=indices.cpu().tolist(),block_table=table.cpu().tolist(),cache_shape=list(cache.shape),cache_stride=list(cache.stride()),metadata=md_summary(md)))
        (p/'capture'/f'{root}-layers.json').write_text(json.dumps(layers,indent=2))
    from torch.utils._python_dispatch import TorchDispatchMode
    import re
    all_weights={t.data_ptr():n for n,t in model.named_parameters()}
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    original_kernel_run=CachingAutotuner.run
    kernel_dir=p/'kernels';kernel_dir.mkdir(exist_ok=True)
    launches=(p/'norm-launches.jsonl').open('w')
    def record_kernel(self,*args,**kwargs):
        result=original_kernel_run(self,*args,**kwargs)
        if state['run']=='observed'and 'rms_norm' in self.fn.__name__:
            source=self.fn.src;key=hashlib.sha256(source.encode()).hexdigest();f=kernel_dir/(key+'.py')
            if not f.exists():f.write_text(source)
            launcher=self.launchers[0]
            weights={str(i):all_weights.get(t.data_ptr())for i,t in enumerate(args)if isinstance(t,torch.Tensor)and t.data_ptr()in all_weights}
            scalars={str(i):t for i,t in enumerate(args)if isinstance(t,(int,float,bool))}
            launches.write(json.dumps(dict(call=state['current']['call'],positions=state['current']['positions'],kernel=self.fn.__name__,source=key,config=launcher.config.kwargs,num_warps=launcher.config.num_warps,weights=weights,scalars=scalars,cache_hash=launcher.cache_hash))+'\n');launches.flush()
        return result
    CachingAutotuner.run=record_kernel
    stages=[];directory=p/'stages';directory.mkdir(exist_ok=True)
    def save_stage(layer,role,tensor):
        name=f"{state.get('stage_prefix','')}l{layer}-{role}.bin";data=tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes();f=directory/name
        assert not f.exists(),name
        f.write_bytes(data);stages.append({'file':name,'shape':list(tensor.shape),'dtype':str(tensor.dtype),'sha256':hashlib.sha256(data).hexdigest()});(p/'stage-manifest.json').write_text(json.dumps(stages,indent=2))
    class CaptureDispatch(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            kwargs=kwargs or {};linear='_fused_mul_mat_gguf' in str(func)and len(args)>1 and isinstance(args[0],torch.Tensor)
            name=all_weights.get(args[1].data_ptr()) if linear and isinstance(args[1],torch.Tensor)else None
            match=re.search(r'\.layers\.(\d+)\.(.*)',name) if name else None
            layer,role=(int(match.group(1)),match.group(2))if match else(-1,'')
            if linear and args[0].ndim==2 and args[0].shape[1]==5120:
                if state.get('pending_input')is not None:save_stage(state.pop('pending_input'),'input-norm',args[0])
                if state.get('pending_ffn')is not None:save_stage(state.pop('pending_ffn'),'ffn-norm',args[0])
            result=func(*args,**kwargs)
            if role in ['linear_attn.out_proj.weight','self_attn.o_proj.weight']:
                save_stage(layer,'block-output',result);state['pending_ffn']=layer
            if role=='mlp.down_proj.weight':
                save_stage(layer,'ffn-output',result)
                if layer<63:state['pending_input']=layer+1
            return result
    def pre(module,args,kwargs):
        from vllm.forward_context import get_forward_context
        ctx=get_forward_context();mds=ctx.attn_metadata
        if not isinstance(mds,dict)or not mds:return
        inp=kwargs.get('input_ids',args[0]if args else None);pos=kwargs.get('positions',args[1]if len(args)>1 else None)
        if inp is None:return
        gkey,gmd=metadata_for(mds,0)
        record=dict(run=state['run'],call=state['calls'],input_ids=inp.detach().cpu().tolist(),positions=pos.detach().cpu().tolist(),gdn=md_summary(gmd),metadata_keys=list(mds),context={k:str(v)for k,v in vars(ctx).items()if k in ['batch_descriptor','cudagraph_runtime_mode','num_tokens','num_actual_tokens']})
        trace.append(record);state['calls']+=1;state['current']=record;state['selected']=False
        (p/'forward-trace.json').write_text(json.dumps(trace,indent=2))
        if state['run']=='observed'and inp.numel()==516 and getattr(gmd,'num_prefills',0)==4:
            state['stage_prefix']='prefill-'
            state['pending_input']=0;state['prefill_dispatch']=CaptureDispatch();state['prefill_dispatch'].__enter__()
        if state['run']=='observed'and getattr(gmd,'num_prefills',-1)==0 and getattr(gmd,'num_decodes',0)==4 and pos.numel()==4:
            n=gmd.num_actual_tokens;assert n==4 and inp.numel()==4,(n,inp.shape)
            state['stage_prefix']=f"decode{int(pos[0])-128}-"
            state['pending_input']=0;state['prefill_dispatch']=CaptureDispatch();state['prefill_dispatch'].__enter__()
            if not state['captured']and bool(torch.all(pos==177)):
                save('input-ids.bin',inp);save('positions.bin',pos);snapshot('before',mds,n)
                state['selected']=True;state['captured']=True;state['selected_metadata']=mds
                (p/'capture/selected-call.json').write_text(json.dumps(record,indent=2))
    def post(module,args,kwargs,output):
        mode=state.pop('prefill_dispatch',None)
        if mode is not None:
            save_stage(64,'input-norm',output if isinstance(output,torch.Tensor)else output[0]);mode.__exit__(None,None,None)
        if state['selected']:
            save('hidden-output.bin',output if isinstance(output,torch.Tensor)else output[0]);snapshot('after',state['selected_metadata'],4)
    original_logits=model.compute_logits
    def logits(*args,**kwargs):
        output=original_logits(*args,**kwargs)
        if output is not None:
            run=state['run'];index=len(state['logits'][run]);row=save(f'{run}-logits-{index}.bin',output)
            state['logits'][run].append(row)
            if state['selected']:
                save('selected-logits.bin',output);(p/'capture/selected-logits-index.json').write_text(json.dumps({'index':index,'shape':list(output.shape)}));state['selected']=False
        return output
    model.compute_logits=logits
    def command(action):
        if action=='observe':
            state['run']='observed';state['calls']=0
            model.register_forward_pre_hook(pre,with_kwargs=True);model.register_forward_hook(post,with_kwargs=True)
        return {'captured':state['captured'],'logits':state['logits'],'calls':state['calls']}
    model._bitnet_capture_command=command
    attn=model.model.layers[3].self_attn.attn
    (p/'reference-attention-class.py').write_text(inspect.getsource(type(attn.impl)))
    (p/'model-inspection.json').write_text(json.dumps({'model_class':type(model).__module__+'.'+type(model).__name__,'gdn_classes':[type(model.model.layers[i].linear_attn).__module__+'.'+type(model.model.layers[i].linear_attn).__name__ for i in [0]],'gdn_backend':{k:str(getattr(model.model.layers[0].linear_attn,k,None))for k in ['gdn_decode_kernel','gdn_prefill_backend','enable_packed_recurrent_decode']},'attention_class':type(attn.impl).__module__+'.'+type(attn.impl).__name__,'parameter_bytes':sum(t.numel()*t.element_size()for t in model.parameters()),'nonresident_parameters':[n for n,t in model.named_parameters()if t.device.type!='cuda'and t.numel()>4]},indent=2))
    return {'installed_logits_capture':True}

def capture_command(model,action):return model._bitnet_capture_command(action)

def main():
    p=Path('/evidence')
    subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps','/wheels/vllm_gguf_plugin-0.0.5+e454affe-cp310-abi3-linux_x86_64.whl'],check=True)
    root=Path(importlib.util.find_spec('vllm_gguf_plugin').origin).parent;shutil.copy2('/experts/plugin-512/vllm_gguf_plugin/_C_gguf.abi3.so',root/'_C_gguf.abi3.so')
    adapter=root/'weights_adapter/qwen3_5.py';original=adapter.read_text();old='"qwen3_5_text": "Qwen3_5ForConditionalGeneration"';new='"qwen3_5_text": "Qwen3_5ForCausalLM"';assert original.count(old)==1;adapter.write_text(original.replace(old,new))
    (p/'adapter-change.json').write_text(json.dumps({'before_sha256':hashlib.sha256(original.encode()).hexdigest(),'after_sha256':hashlib.sha256(adapter.read_bytes()).hexdigest(),'scope':'Text-only architecture mapping, no model math changes'}))
    from vllm import LLM,SamplingParams
    import torch,numpy as np
    args=dict(model='/data/models/gguf/qwen3_8/27b/q4_k_xl/Qwen3.8-27B-UD-Q4_K_XL.gguf',hf_config_path='/config',skip_tokenizer_init=True,dtype='bfloat16',quantization='gguf',load_format='gguf',tensor_parallel_size=1,max_model_len=8192,max_num_seqs=4,max_num_batched_tokens=8192,enable_prefix_caching=False,enable_chunked_prefill=True,kv_cache_memory_bytes=2*1024**3,mamba_ssm_cache_dtype='float32',disable_log_stats=True,async_scheduling=False,seed=0,compilation_config={'cudagraph_mode':'NONE'},additional_config={'gdn_prefill_backend':'flashinfer','gdn_decode_kernel':'cuda'})
    (p/'engine-request.json').write_text(json.dumps(args,indent=2));llm=LLM(**args)
    try:
        config=llm.llm_engine.vllm_config;(p/'engine-effective-config.json').write_text(json.dumps({key:str(getattr(config,key))for key in ['cache_config','scheduler_config','compilation_config','mamba_config','quant_config']},indent=2))
        print('READY',flush=True);print(llm.apply_model(install_capture),flush=True)
        cases=['batch4'];params=SamplingParams(temperature=0,max_tokens=64,ignore_eos=True,detokenize=False)
        original=np.fromfile('/native-prompts/medium.i32',dtype=np.int32).tolist()
        batch=[]
        for i in range(4):
            ids=original.copy();j=1+i%(len(ids)-1);ids[1],ids[j]=ids[j],ids[1]
            batch.append({'prompt_token_ids':ids})
        prompts={'batch4':batch}
        (p/'prompts.json').write_text(json.dumps(prompts,indent=2));control={};observed={}
        for case in cases:
            result=llm.generate(prompts[case],params,use_tqdm=False);control[case]=[list(r.outputs[0].token_ids)for r in result];print('CONTROL',case,control[case],flush=True)
        (p/'control-tokens.json').write_text(json.dumps(control,indent=2));llm.apply_model(lambda m:capture_command(m,'observe'))
        for case in cases:
            result=llm.generate(prompts[case],params,use_tqdm=False);observed[case]=[list(r.outputs[0].token_ids)for r in result];print('OBSERVED',case,observed[case],flush=True)
        (p/'observed-tokens.json').write_text(json.dumps(observed,indent=2));status=llm.apply_model(lambda m:capture_command(m,'status'));(p/'capture-status.json').write_text(json.dumps(status,indent=2))
        assert observed==control,'observations changed streams'
        for item in status:
            assert item['captured'],'no decode captured';a=item['logits']['control'];b=item['logits']['observed'];assert len(a)==len(b),(len(a),len(b))
            assert all(x['shape']==y['shape']and x['sha256']==y['sha256']for x,y in zip(a,b)),'observations changed raw logits'
        (p/'reference-check.json').write_text(json.dumps({'tokens_identical':True,'all_raw_logits_identical':True,'captured_decode':True,'cases':cases,'scope':'Original compiled full model; FlashInfer prefill/CUDA fused-norm selection; graphs disabled; synchronous scheduler; no tensor/kernel math patches'},indent=2));print('REFERENCE_CAPTURE_PASSED',flush=True)
    finally:llm.llm_engine.engine_core.shutdown()

if __name__=='__main__':main()
