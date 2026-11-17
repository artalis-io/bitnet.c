"""Generate independent BF16 GDN prefill-preparation references.

Pinned vLLM image 745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29,
Torch 2.14.0a0+4fdf77b940. Unchanged fused_post_conv_prep: default L2 epsilon
1e-6, BF16 Q/K/V, F32 log-decay and beta. Additional cases call its unchanged
kernel with varied epsilon/softplus-threshold arguments. Inputs use packed [beta,alpha] rows.
The reference's separate Q/K/V results are packed only for serialization.
No native BitNet arithmetic generates expected outputs. BF16 hashes consume
bytes; F32 hashes consume complete uint32 bit patterns, as in the C helpers.
"""
import argparse,json,numpy as np
from pathlib import Path
import torch
from vllm.third_party.flash_linear_attention.ops import fused_post_conv_prep
from vllm.third_party.flash_linear_attention.ops.fused_gdn_prefill_post_conv import _fused_post_conv_kernel
parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);parser.add_argument('--header',type=Path);parser.add_argument('--model',type=Path);parser.add_argument('--captures',type=Path);parser.add_argument('--convolution-dir',type=Path);args=parser.parse_args();p=args.directory;cases=[]
def save(name,t):
 flat=torch.empty(t.numel(),dtype=t.dtype);flat.copy_(t.detach().cpu().reshape(-1));data=flat.view(torch.uint8).numpy().tobytes();(p/name).write_bytes(data);h=14695981039346656037
 for value in (data if t.dtype==torch.bfloat16 else np.frombuffer(data,dtype='<u4')):h=((h^int(value))*1099511628211)&((1<<64)-1)
 return h
geometries=[(1,k,1,(1 if k==1 else max((1<<(k-1).bit_length())//2+1,k-3))) for k in [1,7,16,31,32,33,63,64,65,127,128,129,255,256]]+[(2,k,6,(k-1 if k!=128 else 128)) for k in [32,64,128,256]]+[(16,128,48,128)]
shapes=[(*g,n,s,1e-6,20.) for g in geometries for n in [1,5,17,33] for s in [0,1]]+[(1,32,1,31,65537,s,1e-6,20.) for s in [0,1]]
shapes += [(*g,17,s,e,th) for g in [(1,7,1,5),(16,128,48,128)] for s in [0,1] for e in [1e-5,1e-3] for th in [0.,8.,20.]]
for hk,dk,hv,dv,nt,seed,eps,threshold in shapes:
 dim=2*hk*dk+hv*dv;i=torch.arange(nt*dim);v=(((i*37+i//13+seed*11)%509-254).float()/128).reshape(nt,dim);scales=torch.tensor([0.,2**-20,1.,256.]);v=v*scales[(torch.arange(nt)+seed)%4,None]
 if seed:v=v*torch.pow(2.,((torch.arange(dim)%7)-3).float())[None,:]
 x=v.to(torch.bfloat16).cuda();i=torch.arange(nt*hv);a=(((i*17+i//19+seed*3)%509-254).float()/8).to(torch.bfloat16).reshape(nt,hv).cuda();b=(((i*13+seed*11)%251-125).float()/16).to(torch.bfloat16).reshape(nt,hv).cuda();ba=torch.cat([b,a],-1);i=torch.arange(hv);log=(((i*11+seed*5)%29-14).float()/8+2**-17).cuda();bias=(((i*7+seed)%31-15).float()/32).to(torch.bfloat16).cuda()

 if eps==1e-6 and threshold==20.:
  q,k,value,g,beta=fused_post_conv_prep(x,ba[:,hv:],ba[:,:hv],log,bias,hk,dk,dv)
 else:
  q=torch.empty(nt,hk,dk,dtype=x.dtype,device='cuda');k=torch.empty_like(q);value=torch.empty(nt,hv,dv,dtype=x.dtype,device='cuda');g=torch.empty(nt,hv,device='cuda');beta=torch.empty_like(g)
  _fused_post_conv_kernel[((nt+15)//16,hk+hv)](x,ba[:,hv:],ba[:,:hv],log,bias,q,k,value,g,beta,x.stride(0),ba.stride(0),ba.stride(0),q.stride(0),k.stride(0),value.stride(0),nt,H=hk,HV=hv,K=dk,V=dv,APPLY_L2NORM=True,L2NORM_EPS=eps,OUTPUT_G_EXP=False,SOFTPLUS_THRESHOLD=threshold,BLOCK_T=16,BK=1<<(dk-1).bit_length(),BV=1<<(dv-1).bit_length(),num_warps=4,num_stages=2)
 out=torch.cat([q.reshape(nt,hk*dk),k.reshape(nt,hk*dk),value.reshape(nt,hv*dv)],-1);assert all(torch.isfinite(y).all() for y in [out,g,beta]);name=f'prep-{hk}-{dk}-{hv}-{dv}-{nt}-{seed}';name+=('' if eps==1e-6 and threshold==20. else f'-{eps}-{threshold}');h=save(name+'.bin',out);hg=save(name+'-g.bin',g);hb=save(name+'-beta.bin',beta);cases.append(dict(hk=hk,dk=dk,hv=hv,dv=dv,tokens=nt,seed=seed,eps=eps,threshold=threshold,hash=h,g_hash=hg,beta_hash=hb,file=name+'.bin'))
(p/'cases.json').write_text(json.dumps(cases,indent=2));lines=['/* Unchanged pinned vLLM BF16 fused_post_conv_prep. */','#ifndef BN_CUDA_BF16_GDN_PREP_REFERENCE_H','#define BN_CUDA_BF16_GDN_PREP_REFERENCE_H','typedef struct { int hk,dk,hv,dv,tokens,seed; float eps,threshold; uint64_t hash,g_hash,beta_hash; } BnBF16GDNPrepReference;','static const BnBF16GDNPrepReference cuda_bf16_gdn_prep_cases[] = {']
for c in cases:lines.append(' {%d,%d,%d,%d,%d,%d,%.9ef,%.9ef,UINT64_C(0x%016x),UINT64_C(0x%016x),UINT64_C(0x%016x)},'%(c['hk'],c['dk'],c['hv'],c['dv'],c['tokens'],c['seed'],c['eps'],c['threshold'],c['hash'],c['g_hash'],c['beta_hash']))
lines+=['};','#endif'];(args.header or p/'cuda_bf16_gdn_prep_reference.h').write_text('\n'.join(lines)+'\n');print('GDN prep references',len(cases),'qkv words',sum(c['tokens']*(2*c['hk']*c['dk']+c['hv']*c['dv']) for c in cases),'F32 gates per vector',sum(c['tokens']*c['hv'] for c in cases),flush=True)

if args.model:
 assert args.captures is not None and args.convolution_dir is not None
 import gguf
 from vllm_gguf_plugin.quantization.layout import GGUFHeadTilingLayout
 reader=gguf.GGUFReader(args.model);tensors={t.name:t for t in reader.tensors};layout=GGUFHeadTilingLayout(3,128)
 raw=tensors.get('blk.0.ssm_a',tensors.get('blk.0.ssm_a.weight'));assert raw is not None
 # The pinned adapter transforms the CPU iterator's F32 coefficients before
 # model loading copies the natural-log values to its F32 GPU parameter.
 coefficient=torch.from_numpy(np.array(raw.data,copy=True).reshape(48));assert (coefficient<0).all()
 log_cpu=layout.weight_to_vllm(torch.log(-coefficient),dim=0,head_dim=1)
 bias_raw=tensors['blk.0.ssm_dt.bias'];bias_cpu=layout.weight_to_vllm(torch.from_numpy(np.array(bias_raw.data,copy=True).reshape(48)),dim=0,head_dim=1).to(torch.bfloat16)
 log=log_cpu.cuda();bias=bias_cpu.cuda();save('real-log.bin',log);save('real-bias.bin',bias);real=[]
 for phase,nt in [(0,5),(1,1)]:
  x=torch.from_numpy(np.fromfile(args.convolution_dir/f'real-output-{phase}.bin',np.uint16)).view(torch.bfloat16).reshape(nt,10240).cuda();ba=torch.from_numpy(np.fromfile(args.captures/f'model.layers.0.linear_attn.in_proj_ba.{phase}.output.bin',np.uint16)).view(torch.bfloat16).reshape(nt,96).cuda();save(f'real-input-{phase}.bin',x);save(f'real-ba-{phase}.bin',ba)
  q,k,v,g,beta=fused_post_conv_prep(x,ba[:,48:],ba[:,:48],log,bias,16,128,128);out=torch.cat([q.reshape(nt,2048),k.reshape(nt,2048),v.reshape(nt,6144)],-1);assert all(torch.isfinite(y).all() for y in [out,g,beta]);h=save(f'real-output-{phase}.bin',out);hg=save(f'real-g-{phase}.bin',g);hb=save(f'real-beta-{phase}.bin',beta);real.append(dict(phase=phase,tokens=nt,hk=16,dk=128,hv=48,dv=128,eps=1e-6,threshold=20.,hash=h,g_hash=hg,beta_hash=hb))
 (p/'real-cases.json').write_text(json.dumps(real,indent=2));print('Real Qwen preparation references',real,flush=True)
