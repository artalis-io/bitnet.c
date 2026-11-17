#!/usr/bin/env python3
"""Offline verifier; references are immutable. Select ISA modes only on matching hardware."""
from pathlib import Path
import argparse,ctypes as C,json,hashlib,subprocess
import numpy as np
parser=argparse.ArgumentParser(description='Verify F32 mHC coefficients and coupled collapse/post mixing.')
parser.add_argument('--reference-dir',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,required=True)
parser.add_argument('--mode',choices=['scalar','avx2','avx512'],action='append')
options=parser.parse_args();p=options.output_dir.resolve();p.mkdir(parents=True,exist_ok=True);reference_dir=options.reference_dir.resolve();repo=Path(__file__).resolve().parents[1]
class Coeff(C.Structure):
 _fields_=[(n,C.POINTER(C.c_float))for n in ['pre','post','comb','mixes','scale','base']]+[(n,C.c_int)for n in ['streams','iterations']]+[(n,C.c_float)for n in ['pre_eps','sinkhorn_eps','post_mult']]
def ptr(a):return a.ctypes.data_as(C.POINTER(C.c_float))
ref=json.loads((reference_dir/'reference.json').read_text());results={}
for mode,flags in [('scalar',['-fno-tree-vectorize']),('avx2',['-mavx2','-mfma']),('avx512',['-mavx512f','-mavx512dq','-mavx512bw','-mavx512vl','-mfma'])]:
 if mode not in (options.mode or ['scalar']):continue
 lib=p/(mode+'.so');r=subprocess.run(['cc','-O3','-std=c11','-Wall','-Wextra','-Wshadow','-fPIC','-shared','-I'+str(repo/'include'),*flags,str(repo/'src/transformer/math_scalar.c'),'-lm','-o',str(lib)],capture_output=True,text=True);assert not r.returncode and not r.stderr,r.stderr
 api=C.CDLL(str(lib));coeff=api.bn_transformer_mhc_coefficients;coeff.argtypes=[C.POINTER(Coeff)];coeff.restype=C.c_int;collapse=api.bn_transformer_mhc_mix_input;collapse.argtypes=[C.POINTER(C.c_float)]*3+[C.c_int]*2;collapse.restype=C.c_int;postop=api.bn_transformer_mhc_mix_output;postop.argtypes=[C.POINTER(C.c_float)]*5+[C.c_int]*2;postop.restype=C.c_int;rows=[]
 for case in ref['cases']:
  f=reference_dir/f'case{case["case"]}.npz';assert hashlib.sha256(f.read_bytes()).hexdigest()==case['sha256'];d=dict(np.load(f));before={k:hashlib.sha256(v.tobytes()).hexdigest()for k,v in d.items()};h,dim=case['streams'],case['dim'];err={k:0. for k in ['pre','post','comb','collapsed','out']};count=0
  for t in range(case['tokens']):
   pre=np.empty(h,np.float32);post=np.empty(h,np.float32);comb=np.empty((h,h),np.float32);collapsed=np.empty(dim,np.float32);out=np.empty((h,dim),np.float32)
   c=Coeff(ptr(pre),ptr(post),ptr(comb),ptr(d['mixes'][t]),ptr(d['scale']),ptr(d['base']),h,case['iterations'],case['pre_eps'],case['sinkhorn_eps'],case['post_mult']);assert coeff(C.byref(c))==0;assert collapse(ptr(collapsed),ptr(d['residual'][t]),ptr(pre),h,dim)==0;assert postop(ptr(out),ptr(d['residual'][t]),ptr(d['branch'][t]),ptr(post),ptr(comb),h,dim)==0
   for k,a in dict(pre=pre,post=post,comb=comb,collapsed=collapsed,out=out).items():
    e=float(np.max(np.abs(a-d[k][t])));assert np.isfinite(a).all()and e<5e-6,(mode,case,k,e);err[k]=max(err[k],e);count+=a.size
  assert before=={k:hashlib.sha256(v.tobytes()).hexdigest()for k,v in d.items()}
  rows.append(dict(case=case['case'],errors=err,compared_values=count));print(mode,rows[-1],flush=True)
 results[mode]=rows
(p/'results.json').write_text(json.dumps(results,indent=2)+'\n')
