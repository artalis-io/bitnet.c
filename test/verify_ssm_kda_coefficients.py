#!/usr/bin/env python3
"""Check coefficient construction and every fused gate/recurrence prefix.

Reference files are immutable; writes only libraries/results to --output-dir.
Optional x86 ISA modes require matching hardware. No model support is inferred.
"""
from pathlib import Path
import argparse,ctypes as C,hashlib,json,subprocess
import numpy as np
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--reference-dir',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,required=True)
parser.add_argument('--mode',choices=['scalar','avx2','avx512'],action='append')
options=parser.parse_args();source=options.reference_dir.resolve();dest=options.output_dir.resolve();dest.mkdir(parents=True,exist_ok=True);repo=Path(__file__).resolve().parents[1]
class Coeff(C.Structure):
 _fields_=[(k,C.POINTER(C.c_float))for k in ['log_decay','beta','gate','bias','decay_parameter','beta_raw']]+[('num_heads',C.c_int),('head_k_dim',C.c_int),('lower_bound',C.c_float),('parameter_storage',C.c_int)]
class State(C.Structure):
 _fields_=[(k,C.POINTER(C.c_float))for k in ['state','out','q','k','v','log_decay','beta']]+[(k,C.c_int)for k in ['num_heads','head_k_dim','head_v_dim']]+[('q_scale',C.c_float)]
def ptr(a):return a.ctypes.data_as(C.POINTER(C.c_float))
reference=json.loads((source/'reference.json').read_text());results={}
for mode,flags in [('scalar',['-fno-tree-vectorize']),('avx2',['-mavx2','-mfma']),('avx512',['-mavx512f','-mavx512dq','-mavx512bw','-mavx512vl','-mfma'])]:
 if mode not in (options.mode or ['scalar']):continue
 library=dest/(mode+'.so');cmd=['cc','-O3','-std=c11','-Wall','-Wextra','-Wshadow','-fPIC','-shared','-I'+str(repo/'include'),*flags,str(repo/'src/transformer/ssm_scalar.c'),str(repo/'src/transformer/math_scalar.c'),'-lm','-o',str(library)]
 run=subprocess.run(cmd,capture_output=True,text=True);assert run.returncode==0 and not run.stderr,run.stderr
 lib=C.CDLL(str(library));coeff=lib.bn_transformer_ssm_kda_coefficients;coeff.argtypes=[C.POINTER(Coeff),C.c_int,C.c_int];coeff.restype=C.c_int;step=lib.bn_transformer_ssm_kda_step;step.argtypes=[C.POINTER(State),C.c_int,C.c_int];step.restype=C.c_int
 rows=[]
 for case in reference['cases']:
  f=source/f'case{case["case"]}.npz';assert hashlib.sha256(f.read_bytes()).hexdigest()==case['sha256'];d=dict(np.load(f));h,dim,tokens=case['heads'],case['dim'],case['tokens'];hashes={k:hashlib.sha256(v.tobytes()).hexdigest()for k,v in d.items()};mg=mb=ms=mo=0.;values=0
  for storage in [0,1]:
   parameter=d['a_log']if storage==0 else -np.exp(d['a_log'])
   for alias in [False,True]:
    state=d['initial'].copy()if case['coupled']else None
    for t in range(tokens):
     gate=d['gate'][t].copy();raw=d['beta_raw'][t].copy();gout=gate if alias else np.empty_like(gate);bout=raw if alias else np.empty_like(raw)
     c=Coeff(ptr(gout),ptr(bout),ptr(gate),ptr(d['bias']),ptr(parameter),ptr(raw),h,dim,case['lower_bound'],storage);ranges=[(i,i+1)for i in reversed(range(h))]if alias else[(0,h)]
     for a,b in ranges:assert coeff(C.byref(c),a,b)==0
     assert np.isfinite(gout).all()and np.isfinite(bout).all();assert (gout>=case['lower_bound']).all()and(gout<=0).all()and(bout>=0).all()and(bout<=1).all()
     eg=float(np.max(np.abs(gout-d['log_decay'][t])));eb=float(np.max(np.abs(bout-d['beta'][t])));mg=max(mg,eg);mb=max(mb,eb);assert eg<5e-6 and eb<2e-7,(mode,case,t,eg,eb);values+=gout.size+bout.size
     if case['coupled']:
      out=np.empty((h,dim),dtype=np.float32);s=State(ptr(state),ptr(out),ptr(d['q'][t]),ptr(d['k'][t]),ptr(d['v'][t]),ptr(gout),ptr(bout),h,dim,dim,float(d['scale']))
      for a,b in ranges:assert step(C.byref(s),a,b)==0
      es=float(np.max(np.abs(state-d['states'][t])));eo=float(np.max(np.abs(out-d['out'][t])));ms=max(ms,es);mo=max(mo,eo);assert es<5e-6 and eo<5e-6,(mode,case,t,es,eo);values+=state.size+out.size
      if t==tokens//2:state=state.copy()
  assert hashes=={k:hashlib.sha256(v.tobytes()).hexdigest()for k,v in d.items()}
  rows.append(dict(case=case['case'],decay_max_abs=mg,beta_max_abs=mb,state_max_abs=ms,out_max_abs=mo,compared_values=values));print(mode,rows[-1],flush=True)
 results[mode]=rows
(dest/'results.json').write_text(json.dumps(results,indent=2)+'\n')
