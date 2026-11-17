"""Independent Engram math capture; no candidate engine imports.

The explicit bridge is Diffbot INT4 -> BF16 -> FP16 lookup, followed by the
pinned Exllama conv-less Engram math. It is not a Diffbot full-forward oracle.
"""
import argparse,ctypes as C,hashlib,importlib.util,json,math,os
from pathlib import Path
import torch
from safetensors import safe_open
P=Path('/tmp/bitnet-goal-native-engram-math')
SRC=Path('/tmp/bitnet-goal-exl3-source-bd1f5191/exllamav3')
ROOT=Path('/data/models/safetensors/deepseek-v4.1-flash-1017-diffbot-exl3-3bpw')
CORE=Path('/tmp/bitnet-goal-native-core-owner')
HASH=Path('/tmp/bitnet-goal-native-engram/reference-final')
sha=lambda b:hashlib.sha256(b).hexdigest()
parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);args=parser.parse_args();out=args.output
# Check original source dependencies before creating any output.
for file,h in json.loads((P/'primary-source.json').read_text()).items():assert sha(Path(file).read_bytes())==h,file
EXPECTED_PRIMARY = {'/tmp/bitnet-goal-exl3-source-bd1f5191/exllamav3/modules/ple.py': 'c22d60b3d7d6765e37685a6383b54a2233f6063d2500dcb638f3c83010e598fc', '/tmp/bitnet-goal-exl3-source-bd1f5191/exllamav3/modules/ngram_embedding.py': '7affaa5ec9c1a31d2d3d1d95eeb6270db70f65e3e4fb33b10447accda5afd65c', '/tmp/bitnet-goal-exl3-source-bd1f5191/exllamav3/modules/linear.py': '4638fea2f6f2fbf897ea6206f06f06ee45a37848f8f605574b169cd4141671df', '/tmp/bitnet-goal-exl3-source-bd1f5191/exllamav3/modules/rmsnorm.py': '2145364078c17de81291a9b786b6517a880d6af79bec9ce6ff2f7686a3b41273', '/tmp/bitnet-goal-exl3-source-bd1f5191/exllamav3/loader/safetensors.py': '25a0354436599760d08fdb54594ea7f384f74008c480874d778ec5bf265f84da', '/tmp/bitnet-goal-exl3-source-bd1f5191/exllamav3/exllamav3_ext/ple.cu': 'b8af7f8b4d672c138721be1928c7a5fdc8458bd07d67487c6deb0068d290fd6c', '/tmp/bitnet-goal-exl3-source-bd1f5191/exllamav3/exllamav3_ext/norm.cu': 'cec4dee97c5e28fda89587fa5ab58419ca437c36168174dde49cb9ac4ee1040f', '/tmp/bitnet-goal-native-engram-math/reference-gate.so': '0ab34525d02a19e624125f1192a1a2550714f4067d1cabf915eb793acefb9aa7', '/tmp/bitnet-goal-native-rmsnorm/reference.so': 'df855447522fa9b18803432573056269d26fb5e976b1d9c5eeb3fd8092f11fcb'}
for file,h in EXPECTED_PRIMARY.items():assert sha(Path(file).read_bytes())==h,file
original=(SRC/'exllamav3_ext/ple.cu').read_text();start=original.index('__global__ __launch_bounds__(NUM_THREADS)');end=original.index('\nvoid ple_gate_gr',start)
assert (P/'original-gate.cuh').read_text()==original[start:end]
core_manifest=json.loads((CORE/'reference.json').read_text());assert core_manifest['source_sha256']==sha((SRC/'modules/linear.py').read_bytes())
assert torch.cuda.device_count()==1;torch.cuda.set_device(0);assert not torch.backends.cuda.matmul.allow_tf32
out.mkdir(parents=True,exist_ok=True);files={};cases=[];weight_meta=[];lookup_meta=[]
def emit(name,data):
 f=out/name
 if f.exists():assert f.read_bytes()==data,'immutable reference conflict: '+str(f)
 else:f.write_bytes(data)
 files[name]={'bytes':len(data),'sha256':sha(data)}
 return name
def save(name,t):return emit(name,t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
ptr=C.c_void_p;it=C.c_int
norm=C.CDLL('/tmp/bitnet-goal-native-rmsnorm/reference.so');norm.reference.argtypes=[ptr]*3+[it]*6+[C.c_float]*3+[ptr];norm.reference.restype=it
gate=C.CDLL(str(P/'reference-gate.so'));gate.reference_gate.argtypes=[ptr]*3+[C.c_size_t,it,it,C.c_float,ptr];gate.reference_gate.restype=it
stream=torch.cuda.current_stream().cuda_stream
sp=importlib.util.spec_from_file_location('original_int4_engram', '/tmp/bitnet-goal-native-engram-rows/original_kernel.py');module=importlib.util.module_from_spec(sp);sp.loader.exec_module(module)
index=json.loads((ROOT/'model.safetensors.index.json').read_text())['weight_map']
def read(name):
 with safe_open(str(ROOT/index[name]),framework='pt',device='cpu')as f:return f.get_tensor(name)
def bf16_rows(layer,rows):
 prefix=f'hash-{layer}-1-{rows}-text';path=HASH/(prefix+'-rows.i64le');blob=path.read_bytes();ids=torch.frombuffer(bytearray(blob),dtype=torch.int64);assert ids.numel()==rows*24
 unique,inverse=ids.unique(sorted=True,return_inverse=True);values=[];raw_meta={}
 for suffix in ['weight','scale']:
  name=f'layers.{layer}.engram.embed.{suffix}'
  with safe_open(str(ROOT/index[name]),framework='pt',device='cpu')as f:
   sl=f.get_slice(name);t=torch.cat([sl[int(i):int(i)+1] for i in unique],dim=0)
  values.append(t.cuda());raw_meta[suffix]=save(prefix+'.'+suffix,t)
 w,s=values;decoded=torch.empty((ids.numel(),256),device='cuda',dtype=torch.bfloat16)
 local=inverse.cuda().view(-1,1)
 module._engram_lookup_int4_kernel[(32,)](w,s,local,decoded,0,len(unique),ids.numel(),1,1,0,1,1,256,32,8,32)
 # The intermediate BF16 rounding is deliberately preserved before FP16.
 packed=w[inverse.cuda()].to(torch.int32);nibbles=torch.stack((packed&15,packed>>4),dim=-1).reshape(-1,256)
 direct=((nibbles.float()-8)*s[inverse.cuda()].float().repeat_interleave(32,dim=1)).half()
 bridged=decoded.half();different=int((direct.view(torch.int16)!=bridged.view(torch.int16)).sum().item())
 lookup_meta.append(dict(layer=layer,rows=rows,original_hash_rows=str(path),hash_rows_sha256=sha(blob),unique_ids=save(prefix+'.uids',unique),inverse=save(prefix+'.inverse',inverse),source_rows=raw_meta,bf16=save(prefix+'.bf16',decoded),f16=save(prefix+'.f16',bridged),direct_f16_different_words=different))
 return bridged.reshape(rows,6144)
def streams(rows,H,D,seed):
 x=(((torch.arange(rows*H*D,dtype=torch.int64)*73+seed)%2047-1023).float()/512).reshape(rows,H,D)
 if rows:x[0]=0
 if rows>1:x[1]*=2**-70
 if rows>2:x[2]*=2**20
 return x.cuda()
def execute(x,emb,w,kw,qw):
 R,H,D=x.shape;key=emb@w[:,:H*D];value=emb@w[:,H*D:];kn=torch.empty_like(x);qn=torch.empty_like(x)
 assert key.dtype==value.dtype==torch.float16
 assert norm.reference(key.data_ptr(),kw.data_ptr(),kn.data_ptr(),R*H,D,16,32,1,H,1e-20,0.,1.,stream)==0
 assert norm.reference(x.data_ptr(),qw.data_ptr(),qn.data_ptr(),R*H,D,32,32,1,H,1e-20,0.,1.,stream)==0
 dots=torch.bmm(qn.reshape(-1,1,D),kn.reshape(-1,D,1)).reshape(R,H)
 delta=torch.empty_like(x);assert gate.reference_gate(dots.data_ptr(),value.data_ptr(),delta.data_ptr(),x.numel(),H,D,1/math.sqrt(D),stream)==0
 final=x+delta
 return dict(key=key,value=value,key_norm=kn,query_norm=qn,dot=dots,delta=delta,final=final)
def capture(label,w,kw,qw,embeddings,H,D,metadata):
 weight_meta.append(dict(label=label,**metadata));assert kw.dtype==qw.dtype==torch.bfloat16
 for R,emb in embeddings:
  x=streams(R,H,D,19);result=execute(x,emb,w,kw,qw);repeat=execute(x,emb,w,kw,qw);torch.cuda.synchronize()
  for name,t in result.items():assert bool(torch.isfinite(t).all())and torch.equal(t.view(torch.uint8),repeat[name].view(torch.uint8)),(label,R,name)
  tag=f'{label}.r{R}';case=dict(label=label,rows=R,streams=H,dim=D,k=emb.shape[1],x=save(tag+'.x',x),emb=save(tag+'.emb',emb))
  case.update({name:save(tag+'.'+name,t)for name,t in result.items()});cases.append(case)
  print(label,R,'captured',flush=True)
for layer in [1,14]:
 record=next(v for v in core_manifest['matrices']if v['name']==f'layers.{layer}.engram.wkv');f=CORE/(record['tag']+'.packed-f16');raw=f.read_bytes();assert sha(raw)==record['packed_f16_sha256'];K,N=record['cols'],record['rows'];assert (K,N)==(6144,25600)
 w=torch.frombuffer(bytearray(raw),dtype=torch.float16).reshape(K,N).cuda();del raw
 kw=read(f'layers.{layer}.engram.k_weight');qw=read(f'layers.{layer}.engram.q_weight');assert kw.shape==qw.shape==(4,5120)
 meta=dict(weight=str(f),weight_sha256=record['packed_f16_sha256'],k_weight=save(f'l{layer}.kw',kw),q_weight=save(f'l{layer}.qw',qw),weight_layout='F16[K,(H+1)*D], key and value are unequal STRIDED column views',source_dequant_manifest=str(CORE/'reference.json'),source_dequant_manifest_sha256=sha((CORE/'reference.json').read_bytes()))
 kw=kw.cuda();qw=qw.cuda()
 capture(f'l{layer}',w,kw,qw,((R,bf16_rows(layer,R))for R in [1,7,17,129,513]),4,5120,meta)
 del w,kw,qw
for H,D,K in [(1,128,96),(4,256,128)]:
 N=(H+1)*D;w=(((torch.arange(K*N,dtype=torch.int64)*37+13)%257-128).float()/4096).half().reshape(K,N).cuda()
 kw=(((torch.arange(H*D,dtype=torch.int64)*11+3)%127-63).float()/64).bfloat16().reshape(H,D).cuda();qw=kw.flip(-1).contiguous()
 label=f's{H}';meta=dict(weight=save(label+'.weight',w),k_weight=save(label+'.kw',kw),q_weight=save(label+'.qw',qw),weight_layout='F16[K,(H+1)*D], strided column views')
 def embeddings():
  for R in [1,2,7,17,33,129]:
   b=(((torch.arange(R*K,dtype=torch.int64)*19+7)%251-125).float()/256).bfloat16().cuda();yield R,b.half().reshape(R,K)
 capture(label,w,kw,qw,embeddings(),H,D,meta)
# Standalone exact gate edge capture covers signed zero and active clamp.
g=torch.tensor([0.,-0.,1e-30,-1e-30,1e-6,-1e-6,1.,-1.,100.,-100.,float.fromhex('0x1.fffffep-1'),-float.fromhex('0x1.fffffep-1')],device='cuda').reshape(3,4)
v=torch.linspace(-1,1,3*128,device='cuda').half().reshape(3,128);delta=torch.empty((3,4,128),device='cuda');assert not gate.reference_gate(g.data_ptr(),v.data_ptr(),delta.data_ptr(),delta.numel(),4,128,1.,stream)
edge=dict(gate=save('edge.gate',g),value=save('edge.value',v),delta=save('edge.delta',delta),rows=3,streams=4,dim=128,scale=1.)
libraries={}
for line in Path('/proc/self/maps').read_text().splitlines():
 f=line.split()[-1]
 if 'libcublas'in f and f.startswith('/')and f not in libraries:libraries[f]=sha(Path(f).read_bytes())
sources={str(SRC/f):sha((SRC/f).read_bytes())for f in ['modules/ple.py','modules/ngram_embedding.py','modules/linear.py','modules/rmsnorm.py','loader/safetensors.py','exllamav3_ext/ple.cu','exllamav3_ext/norm.cu']}
binaries={str(f):sha(f.read_bytes())for f in [P/'reference-gate.so',Path('/tmp/bitnet-goal-native-rmsnorm/reference.so')]}
manifest=dict(profile='Explicit Diffbot INT4 BF16 lookup -> FP16 adapter + pinned Exllama conv-less Engram math; not Diffbot full-model reference',source_commit='bd1f519172406a9560bd5c90e61000691bb69a07',torch=torch.__version__,torch_git=torch.version.git_version,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),tf32=torch.backends.cuda.matmul.allow_tf32,allow_fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,eps=1e-20,norm_bias=0.,sources=sources,binaries=binaries,libraries=libraries,lookup=lookup_meta,weights=weight_meta,cases=cases,edge=edge,files=files)
emit('reference.json',(json.dumps(manifest,indent=2)+'\n').encode());print('CAPTURED',len(cases),sum(x['bytes']for x in files.values()),flush=True)
