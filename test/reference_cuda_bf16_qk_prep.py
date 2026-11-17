"""Independent pinned vLLM fused QK norm/RoPE/gate oracle.
Use the standard NVIDIA entrypoint of image
745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29.
Existing reference files are immutable; no model files required.
"""
from pathlib import Path
import json,hashlib,torch,argparse
from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate
ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);args=ap.parse_args();p=args.directory;p.mkdir(parents=True,exist_ok=True)
def save(name,b):
 f=p/name
 if f.exists():assert f.read_bytes()==b,name
 else:f.write_bytes(b)
 return hashlib.sha256(b).hexdigest()
def blob(x):return x.cpu().contiguous().view(torch.uint8).numpy().tobytes()
def hashed(x):
 h=14695981039346656037
 for b in blob(x):h=((h^b)*1099511628211)&((1<<64)-1)
 return h
def vals(n,mul,seed,mod,div):
 i=torch.arange(n,dtype=torch.int64)
 x=((i*mul+seed)%mod-mod//2).float()/div
 if mul==17 and seed>=143:x=torch.ldexp(x,(i%9-4).to(torch.int32))
 if (mul==7 and seed>=39) or (mul==11 and seed>=65):x[0]=-0.0
 return x.to(torch.bfloat16)
cases=[]
for seed,(nt,h,hk,rot,pos,beta) in enumerate([(1,24,4,64,0,1),(5,24,4,64,0,1),(1,24,4,64,5,1),(17,24,4,64,17,1),(129,24,4,64,129,1),(1047,24,4,64,4095,1),(33,3,1,256,31,0),(17,4,4,128,100,1),(1,1,1,2,7,0),(65,32,4,32,999,0),(9,24,8,64,129,1),(1,24,4,64,8191,1),(1,24,4,64,37,1),(3,24,4,64,1,0),(17,24,4,256,41,1),(129,24,4,64,129,1),(1047,24,4,64,4095,1)]):
 seed+=1;cols=(2*h+2*hk)*256;length=pos+nt
 x=vals(nt*cols,17,seed*11,509,32).reshape(nt,cols).cuda()
 qw=vals(256,7,seed*3,127,64).cuda();kw=vals(256,11,seed*5,127,64).cuda()
 cs=vals(length*rot,13,seed*7,257,128).reshape(length,rot).cuda()
 positions=torch.arange(pos,pos+nt,dtype=torch.int64).cuda()
 q,k,g=fused_qk_rmsnorm_rope_gate(x[:,:h*512],x[:,h*512:h*512+hk*256],qw,kw,cs,positions,1e-6,h,hk,256,rot,norm_beta=beta)
 v=x[:,h*512+hk*256:].contiguous()
 row=dict(seed=seed,tokens=nt,heads=h,kv_heads=hk,rotary=rot,position=pos,beta=beta,q_hash=hashed(q),k_hash=hashed(k),v_hash=hashed(v),gate_hash=hashed(g))
 for name,t in [('q',q),('k',k),('v',v),('gate',g)]:row[name+'_sha']=save(f'{seed}-{name}.bin',blob(t))
 cases.append(row);print('case',seed,nt,h,hk,rot,pos,beta,flush=True)
save('cases.json',(json.dumps(cases,indent=2)+'\n').encode())
lines=['#ifndef BN_CUDA_BF16_QK_PREP_REFERENCE_H','#define BN_CUDA_BF16_QK_PREP_REFERENCE_H','typedef struct {int seed,tokens,heads,kv_heads,rotary,position,beta;uint64_t q_hash,k_hash,v_hash,gate_hash;} BnBF16QKPrepReference;','static const BnBF16QKPrepReference cuda_bf16_qk_prep_cases[] = {']
for r in cases:lines.append(' {%d,%d,%d,%d,%d,%d,%d,UINT64_C(0x%016x),UINT64_C(0x%016x),UINT64_C(0x%016x),UINT64_C(0x%016x)},'%tuple(r[k] for k in ['seed','tokens','heads','kv_heads','rotary','position','beta','q_hash','k_hash','v_hash','gate_hash']))
lines+=['};','#endif'];save('reference.h',('\n'.join(lines)+'\n').encode())
