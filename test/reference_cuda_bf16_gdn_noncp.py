"""Generate independent non-CP prefill fixtures in the pinned FlashInfer image.
Usage: python reference_cuda_bf16_gdn_noncp.py OUTPUT_DIRECTORY
Never run as part of tests or replace fixtures without inspecting differences.
"""
from pathlib import Path
import torch,json,ctypes,subprocess,importlib,hashlib,math
cp=importlib.import_module('flashinfer.gdn_kernels.delta_rule_dsl.delta_rule_cp_sm120');p=Path(__import__('sys').argv[1]);p.mkdir(parents=True,exist_ok=True);rows=[]
c=p/'hash.c';c.write_text('#include <stdint.h>\n#include <stddef.h>\nuint64_t hash(const unsigned char*p,size_t n){uint64_t h=UINT64_C(14695981039346656037);for(size_t i=0;i<n;i++)h=(h^p[i])*UINT64_C(1099511628211);return h;}\n');subprocess.run(['cc','-shared','-O3','-fPIC',str(c),'-o',str(p/'hash.so')],check=True);lib=ctypes.CDLL(str(p/'hash.so'));lib.hash.argtypes=[ctypes.c_void_p,ctypes.c_size_t];lib.hash.restype=ctypes.c_uint64
assert torch.cuda.device_count()==1
def raw(t):return t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
def describe(t):
 data=raw(t);return dict(words=t.numel(),fnv=lib.hash(data,len(data)),sha256=hashlib.sha256(data).hexdigest(),dtype=str(t.dtype),shape=list(t.shape))
from flashinfer.gdn_prefill import chunk_gated_delta_rule
for seed in [0,1]:
 for nt in [1,17,63,64,65,129,257,321,513,1025,4097,8192]:
  i=torch.arange(nt*16*128,device='cuda',dtype=torch.int64);k=(((i*37+i//13+seed*97)%31-15).float()/(1024 if seed else 256)).to(torch.bfloat16).reshape(nt,16,128)
  q=(((i*19+i//23+seed*67)%31-15).float()/256).to(torch.bfloat16).reshape(nt,16,128)
  j=torch.arange(nt*48*128,device='cuda',dtype=torch.int64);v=(((j*11+j//17+seed*73)%127-63).float()/32).to(torch.bfloat16).reshape(nt,48,128)
  h=torch.arange(nt*48,device='cuda',dtype=torch.int64);beta=((h//48*11+h%48*17+seed)%129).float().reshape(nt,48)/128;alpha=.5+((h*13+h//7+seed*17)%65).float().reshape(nt,48)/128;cu=torch.tensor([0,nt],device='cuda',dtype=torch.int64)
  z=torch.arange(48*128*128,device='cuda',dtype=torch.int64);initial=torch.zeros((1,48,128,128),device='cuda') if seed==0 else (((z*17+z//31+seed*5)%127-63).float()/1024).reshape(1,48,128,128)
  for log_input in [0,1]:
   decay=-(h*7+seed).remainder(17).add(1).float().reshape(nt,48)/1024 if log_input else None
   multiplier=decay.exp() if log_input else alpha
   def run():
    output,state=chunk_gated_delta_rule(q=q,k=k,v=v,g=multiplier,beta=beta,initial_state=initial,output_final_state=True,cu_seqlens=cu,use_cp=False,use_qk_l2norm_in_kernel=False)
    return [describe(output),describe(state)]
   result=run();assert result==run()
   rows.append(dict(seed=seed,tokens=nt,log_input=log_input,values=result));(p/'reference.json').write_text(json.dumps(rows,indent=2));print(seed,nt,log_input,flush=True)
(p/'reference.h').write_text('/* Independent pinned FlashInfer SM120 non-CP output/F32-state byte hashes. */\n#include <stdint.h>\ntypedef struct {int seed,tokens,log_input;uint64_t output,state;} BnGDNNonCPPrefillReference;\nstatic const BnGDNNonCPPrefillReference gdn_noncp_prefill_cases[]={\n'+''.join('{%d,%d,%d,UINT64_C(%d),UINT64_C(%d)},\n'%(r['seed'],r['tokens'],r['log_input'],*[v['fnv']for v in r['values']])for r in rows)+'};\n')
