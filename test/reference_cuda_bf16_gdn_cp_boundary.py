"""Generate original FlashInfer CP boundary fixtures into argv[1].
Pinned image 745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29.
"""
from pathlib import Path
import torch,json,ctypes,subprocess,importlib,hashlib,math
cp=importlib.import_module('flashinfer.gdn_kernels.delta_rule_dsl.delta_rule_cp_sm120');p=Path(__import__('sys').argv[1]);p.mkdir(parents=True,exist_ok=True);rows=[]
c=p/'hash.c';c.write_text('#include <stdint.h>\n#include <stddef.h>\nuint64_t hash(const unsigned char*p,size_t n){uint64_t h=UINT64_C(14695981039346656037);for(size_t i=0;i<n;i++)h=(h^p[i])*UINT64_C(1099511628211);return h;}\n');subprocess.run(['cc','-shared','-O3','-fPIC',str(c),'-o',str(p/'hash.so')],check=True);lib=ctypes.CDLL(str(p/'hash.so'));lib.hash.argtypes=[ctypes.c_void_p,ctypes.c_size_t];lib.hash.restype=ctypes.c_uint64
assert torch.cuda.device_count()==1
def raw(t):return t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
def describe(t):
 data=raw(t);return dict(words=t.numel(),fnv=lib.hash(data,len(data)),sha256=hashlib.sha256(data).hexdigest(),dtype=str(t.dtype),shape=list(t.shape))
for seed in [0,1]:
 for nt in [511,512,513]:
  i=torch.arange(nt*16*128,device='cuda',dtype=torch.int64);k=(((i*37+i//13+seed*97)%31-15).float()/(1024 if seed else 256)).to(torch.bfloat16).reshape(nt,16,128)
  q=(((i*19+i//23+seed*67)%31-15).float()/256).to(torch.bfloat16).reshape(nt,16,128)
  j=torch.arange(nt*48*128,device='cuda',dtype=torch.int64);v=(((j*11+j//17+seed*73)%127-63).float()/32).to(torch.bfloat16).reshape(nt,48,128)
  h=torch.arange(nt*48,device='cuda',dtype=torch.int64);beta=((h//48*11+h%48*17+seed)%129).float().reshape(nt,48)/128;alpha=.5+((h*13+h//7+seed*17)%65).float().reshape(nt,48)/128;cu=torch.tensor([0,nt],device='cuda',dtype=torch.int64)
  z=torch.arange(48*128*128,device='cuda',dtype=torch.int64);initial=torch.zeros((1,48,128,128),device='cuda') if seed==0 else (((z*17+z//31+seed*5)%127-63).float()/1024).reshape(1,48,128,128)
  t=cp.cp_delta_rule_t_precompute_dsl_sm120(k,beta,cu,nt,max_seqlen=nt)
  for length in [512]:
   m,n=cp.cp_delta_rule_mn_precompute_dsl_sm120(k,v,t,alpha,cu,nt,cp_chunk_len=length,max_seqlen=nt)
   fixed=cp.cp_delta_rule_fixup_dsl_sm120(m,n,cu,nt,cp_chunk_len=length,initial_state=initial)
   values=[describe(x)for x in [m,n,fixed]]
   m2,n2=cp.cp_delta_rule_mn_precompute_dsl_sm120(k,v,t,alpha,cu,nt,cp_chunk_len=length,max_seqlen=nt);f2=cp.cp_delta_rule_fixup_dsl_sm120(m2,n2,cu,nt,cp_chunk_len=length,initial_state=initial)
   assert values==[describe(x)for x in [m2,n2,f2]]
   output=torch.empty_like(v);state=torch.empty_like(initial)
   def run():
    cp.cp_delta_rule_prefill_dsl_sm120(output,state,q,k,v,t,fixed,alpha,1/math.sqrt(128),cu,nt,cp_chunk_len=length,max_seqlen=nt,initial_state=initial)
    return [describe(output),describe(state)]
   result=run();assert result==run()
   row=dict(seed=seed,tokens=nt,chunk=length,values=result);rows.append(row);(p/'reference.json').write_text(json.dumps(rows,indent=2));print(seed,nt,length,result[0]['sha256'],flush=True)
(p/'reference.h').write_text('/* Original FlashInfer CP boundary fixtures; seed0zero/seed1nonzero initial state. */\nstatic const BnGDNCPPrefillReference gdn_cp_boundary_cases[]={\n'+''.join('{%d,%d,%d,UINT64_C(%d),UINT64_C(%d)},\n'%(r['seed'],r['tokens'],r['chunk'],*[v['fnv']for v in r['values']])for r in rows)+'};\n')
