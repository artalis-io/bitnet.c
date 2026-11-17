#!/usr/bin/env python3
"""Offline CPU verifier. References are immutable; writes only build/results output.
The optional x86 ISA modes must be supported by the machine running this tool.
"""
from pathlib import Path
import argparse,subprocess,ctypes as C,json,hashlib
import numpy as np
parser=argparse.ArgumentParser(description='Compare portable KDA C state/output at every prefix against frozen original references.')
parser.add_argument('--reference-dir',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,required=True)
parser.add_argument('--mode',choices=['scalar','avx2','avx512'],action='append',help='repeat for a matrix; default scalar; SIMD modes require matching hardware')
options=parser.parse_args();p=options.output_dir.resolve();p.mkdir(parents=True,exist_ok=True)
reference_dir=options.reference_dir.resolve();repo=Path(__file__).resolve().parents[1]
class Norm(C.Structure):
 _fields_=[('q',C.POINTER(C.c_float)),('k',C.POINTER(C.c_float)),('eps',C.c_float),('head_dim',C.c_int),('mode',C.c_int)]
class Context(C.Structure):
 _fields_=[(k,C.POINTER(C.c_float))for k in ['state','out','q','k','v','log_decay','beta']]+[(k,C.c_int)for k in ['num_heads','head_k_dim','head_v_dim']]+[('q_scale',C.c_float)]
def ptr(a):return a.ctypes.data_as(C.POINTER(C.c_float))
ref=json.loads((reference_dir/'reference.json').read_text());all_results={}
for mode,flags in [('scalar',['-fno-tree-vectorize']),('avx2',['-mavx2','-mfma']),('avx512',['-mavx512f','-mavx512dq','-mavx512bw','-mavx512vl','-mfma'])]:
 if mode not in (options.mode or ['scalar']):continue
 lib=p/(mode+'.so');cmd=['cc','-O3','-std=c11','-Wall','-Wextra','-Wshadow','-fPIC','-shared','-I'+str(repo/'include'),*flags,str(repo/'src/transformer/ssm_scalar.c'),str(repo/'src/transformer/math_scalar.c'),'-lm','-o',str(lib)];cmd+=([str(repo/'src/transformer/ssm_avx2.c')]if mode!='scalar'else[])
 r=subprocess.run(cmd,capture_output=True,text=True);assert not r.returncode and not r.stderr,r.stderr
 dll=C.CDLL(str(lib));norm=getattr(dll,'bn_transformer_ssm_l2norm_scalar_range'if mode=='scalar'else'bn_transformer_ssm_l2norm_avx2_range');norm.argtypes=[C.POINTER(Norm),C.c_int,C.c_int];norm.restype=None
 api=dll.bn_transformer_ssm_kda_step;api.argtypes=[C.POINTER(Context),C.c_int,C.c_int];api.restype=C.c_int;results=[]
 for case in ref['cases']:
  f=reference_dir/f'case{case["case"]}.npz';assert hashlib.sha256(f.read_bytes()).hexdigest()==case['sha256'];d=dict(np.load(f));h,dim,tokens=case['heads'],case['dim'],case['tokens'];initial_hash={k:hashlib.sha256(d[k].tobytes()).hexdigest()for k in ['q','k','v','g','beta','initial']}
  ms=mo=mn=0;count=0
  for split in [False,True]:
   state=d['initial'].copy();out=np.empty((h,dim),dtype=np.float32)
   for t in range(tokens):
    q=d['q'][t].copy();k=d['k'][t].copy();n=Norm(ptr(q),ptr(k),case['eps'],dim,2)
    ranges=[(i,i+1)for i in reversed(range(h))]if split else[(0,h)]
    for a,b in ranges:norm(C.byref(n),a,b)
    en=max(float(np.max(np.abs(q-d['nq'][t]))),float(np.max(np.abs(k-d['nk'][t]))));assert en<2e-6,(mode,case,t,en);mn=max(mn,en);count+=q.size+k.size
    c=Context(ptr(state),ptr(out),ptr(q),ptr(k),ptr(d['v'][t]),ptr(d['g'][t]),ptr(d['beta'][t]),h,dim,dim,float(d['scale']))
    for a,b in ranges:assert api(C.byref(c),a,b)==0
    es=float(np.max(np.abs(state-d['states'][t])));eo=float(np.max(np.abs(out-d['out'][t])));ms=max(ms,es);mo=max(mo,eo);assert es<2e-6 and eo<2e-6,(mode,case,t,es,eo)
    count+=state.size+out.size
    if t==tokens//2:state=state.copy() # explicit resume into fresh caller-owned storage
  assert initial_hash=={k:hashlib.sha256(d[k].tobytes()).hexdigest()for k in initial_hash}
  results.append(dict(case=case['case'],state_max_abs=ms,out_max_abs=mo,norm_max_abs=mn,compared_values=count));print(mode,results[-1],flush=True)
 all_results[mode]=results
(p/'operator-results.json').write_text(json.dumps(all_results,indent=2)+'\n')
