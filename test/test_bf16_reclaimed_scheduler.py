#!/usr/bin/env python3
"""Check the numerical reference's host budgets against the public C planner.

No model files, device context, expected model outputs or reference operators
are involved. The adapter controls geometry only; performance uses the original
unmodified scheduler.
"""
from pathlib import Path
import ast
import ctypes
import os
import random
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory() as directory:
        library=Path(directory)/'host-plan.so'
        subprocess.run([os.environ.get('CC','cc'),'-O2','-Wall','-Wextra','-Wshadow','-Werror','-std=c11','-fPIC','-shared','-I'+str(ROOT/'include'),str(ROOT/'src/generate_batch_plan.c'),'-o',str(library)],check=True)
        class Limits(ctypes.Structure):_fields_=[(k,ctypes.c_int)for k in ['max_requests','max_prefills','max_chunk','token_budget','capacity','mixed_token_budget']]
        class Candidate(ctypes.Structure):_fields_=[(k,ctypes.c_int)for k in ['prompt_tokens','prefilled','generated','max_outputs']]
        class Row(ctypes.Structure):_fields_=[(k,ctypes.c_int)for k in ['candidate','n_tokens','position','is_prefill','emit_token','decode_role']]
        class Plan(ctypes.Structure):_fields_=[('count',ctypes.c_int),('total_tokens',ctypes.c_int),('rows',Row*16)]
        lib=ctypes.CDLL(str(library));fn=lib.bn_generate_batch_plan_with_prefill_policy;fn.argtypes=[ctypes.POINTER(Limits),ctypes.POINTER(Candidate),ctypes.c_int,ctypes.c_int,ctypes.POINTER(Plan)];fn.restype=ctypes.c_int
        ns={};tree=ast.parse((ROOT/'test/reference_cuda_bf16_reclaimed_control.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='reclaimed_native_share');exec(compile(ast.Module(body=[node],type_ignores=[]),'original-reference-host-control','exec'),ns);share=ns['reclaimed_native_share'];rng=random.Random(20261111)
        for case in range(20000):
         count=rng.randrange(1,9);items=[]
         for i in range(count):
          prompt=rng.choice([5,129]);prefill=rng.randrange(prompt)if rng.randrange(2)else prompt;generated=0 if prefill<prompt else rng.randrange(1,193);items.append(Candidate(prompt,prefill,generated,192))
         limits=Limits(8,4,64,256,8192,rng.choice([8,16,32,256]));out=Plan();rc=fn(ctypes.byref(limits),(Candidate*count)(*items),count,1,ctypes.byref(out));assert rc>=0
         decodes=[i for i,c in enumerate(items)if c.prefilled==c.prompt_tokens and c.generated<192];pending=[i for i,c in enumerate(items)if c.prefilled<c.prompt_tokens];fixed=share(256,limits.mixed_token_budget,8,4,64,len(decodes),[items[i].prompt_tokens-items[i].prefilled for i in pending]);selected=pending[:min(len(pending),4,8-len(decodes))]
         expected={i:1 for i in decodes};expected.update({i:min(fixed,items[i].prompt_tokens-items[i].prefilled)for i in selected});actual={out.rows[j].candidate:out.rows[j].n_tokens for j in range(out.count)};assert actual==expected,(case,actual,expected)
         ordered=sorted(expected,key=lambda i:(expected[i],i));assert ordered==[out.rows[j].candidate for j in range(out.count)]
         assert out.total_tokens==sum(expected.values())
    print('Reference reclaimed-quota host control: 20000 independent C-planner cases PASSED')


if __name__=='__main__':
    main()
