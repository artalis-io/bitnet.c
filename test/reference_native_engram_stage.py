"""Same-history two-layer Engram reference from unchanged source math/hash."""
import ast,ctypes as C,hashlib,importlib.util,json,math,sys
from pathlib import Path
import torch
from safetensors import safe_open
p=Path(__file__).parent;out=Path(sys.argv[1]);src=Path('/tmp/bitnet-goal-exl3-source-bd1f5191/exllamav3');parent=Path('/tmp/bitnet-goal-native-engram-math');hashed=Path('/tmp/bitnet-goal-native-engram/reference-final');core=Path('/tmp/bitnet-goal-native-core-owner');root=Path('/data/models/safetensors/deepseek-v4.1-flash-1017-diffbot-exl3-3bpw');sha=lambda b:hashlib.sha256(b).hexdigest()
EXPECTED_PARENTS={'/tmp/bitnet-goal-native-engram-math/paired/reference.json': '97f8d4918f83aed523654ccf215b61b3029acca50bfbb71f29278e3d09ad7d25', '/tmp/bitnet-goal-native-core-owner/reference.json': '2ad88cc42ced85c4c2babed47e87e8072e164ffb1e460c23a2af88bf413340de', '/tmp/bitnet-goal-native-engram/reference-final/manifest.json': '92875f3309c982a04a8d7bbea0221f729454af3c4be387019e610e97df8778ae', '/tmp/bitnet-goal-native-engram-math/capture.py': '779bc11007f87c8e2f4107e06868de9e18645fbc991328b0961e1e54d0fd256b'}
for file,h in EXPECTED_PARENTS.items():assert sha(Path(file).read_bytes())==h,file
source_manifest=json.loads((hashed/'manifest.json').read_text())
for name in ['token-map.i64le','layout.json']+[f'hash-1-1-{R}-text-tokens.i64le'for R in [1,7,17,129,513]]:
 data=(hashed/name).read_bytes();assert len(data)==source_manifest[name]['bytes']and sha(data)==source_manifest[name]['sha256'],name
old=json.loads((parent/'paired/reference.json').read_text())
for file,h in {**old['sources'],**old['binaries']}.items():assert sha(Path(file).read_bytes())==h,file
assert torch.cuda.device_count()==1;torch.cuda.set_device(0);assert not torch.backends.cuda.matmul.allow_tf32
out.mkdir(exist_ok=True,parents=True);files={};groups=[];cases=[];weights=[]
def emit(name,data):
 f=out/name
 if f.exists():assert f.read_bytes()==data,'immutable conflict: '+name
 else:f.write_bytes(data)
 files[name]=dict(bytes=len(data),sha256=sha(data));return name
def save(name,t):return emit(name,t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
ptr=C.c_void_p;it=C.c_int
norm=C.CDLL('/tmp/bitnet-goal-native-rmsnorm/reference.so');norm.reference.argtypes=[ptr]*3+[it]*6+[C.c_float]*3+[ptr];norm.reference.restype=it
gate=C.CDLL(str(parent/'reference-gate.so'));gate.reference_gate.argtypes=[ptr]*3+[C.c_size_t,it,it,C.c_float,ptr];gate.reference_gate.restype=it
stream=torch.cuda.current_stream().cuda_stream
# Exact original producer math and original hash methods; no candidate imports.
ns=dict(torch=torch,norm=norm,gate=gate,stream=stream,math=math)
for node in ast.parse((parent/'capture.py').read_text()).body:
 if isinstance(node,ast.FunctionDef)and node.name in ['execute','streams']:exec(compile(ast.Module(body=[node],type_ignores=[]),str(parent/'capture.py'),'exec'),ns)
cls=next(c for c in ast.parse((src/'modules/ngram_embedding.py').read_text()).body if isinstance(c,ast.ClassDef)and c.name=='NGramEmbedding')
body=[f for f in cls.body if isinstance(f,ast.FunctionDef)and f.name in ['_shift_right_ignore_eos','compute_ngram_ids']]
node=ast.ClassDef(name='OriginalHash',bases=[],keywords=[],body=body,decorator_list=[]);exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(src/'modules/ngram_embedding.py'),'exec'),ns)
sp=importlib.util.spec_from_file_location('original_rows','/tmp/bitnet-goal-native-engram-rows/original_kernel.py');module=importlib.util.module_from_spec(sp);sp.loader.exec_module(module)
index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map'];catalog=json.loads((core/'reference.json').read_text())['matrices'];layouts=json.loads((hashed/'layout.json').read_text());mapblob=(hashed/'token-map.i64le').read_bytes();mapping=torch.frombuffer(bytearray(mapblob),dtype=torch.int64)
models={}
for layer in [1,14]:
 entry=layouts[str(layer)];h=ns['OriginalHash']();h.heads_per_ngram=8;h.ngram_size=4;h.eos_token_id=2;h.layer_multipliers=torch.tensor(entry['multipliers'],dtype=torch.int64);h.head_vocab_sizes=torch.tensor(entry['sizes'],dtype=torch.int64);h.head_offsets=torch.tensor(entry['offsets'],dtype=torch.int64)
 rec=next(w for w in catalog if w['name']==f'layers.{layer}.engram.wkv');f=core/(rec['tag']+'.packed-f16');raw=f.read_bytes();assert sha(raw)==rec['packed_f16_sha256'];w=torch.frombuffer(bytearray(raw),dtype=torch.float16).reshape(6144,25600).cuda();del raw
 nspec={}
 for role in ['k_weight','q_weight']:
  name=f'layers.{layer}.engram.{role}'
  with safe_open(str(root/index[name]),framework='pt',device='cpu')as reader:t=reader.get_tensor(name)
  assert t.dtype==torch.bfloat16 and t.shape==(4,5120);nspec[role]=t.cuda()
 weights.append(dict(layer=layer,weight=str(f),weight_sha256=rec['packed_f16_sha256'],k_weight=save(f'l{layer}.kw',nspec['k_weight']),q_weight=save(f'l{layer}.qw',nspec['q_weight'])))
 models[layer]=(h,w,nspec)
def read_tokens(length):
 f=hashed/f'hash-1-1-{length}-text-tokens.i64le';raw=f.read_bytes();t=torch.frombuffer(bytearray(raw),dtype=torch.int64);assert t.numel()==length;return t
specs=[]
for R in [1,7,17,129,513]:specs.append((f'independent-{R}',read_tokens(R),0,R,'reset',False))
specs.append(('decode-nonzero',read_tokens(1),0,1,'reset',True))
long=read_tokens(513);pos=0
for R in [1,7,17,129,359]:specs.append((f'continue-{pos}-{R}',long,pos,R,'reset'if pos==0 else'continue',False));pos+=R
assert pos==513
specs.append(('rewind-17-17',long,17,17,'rewind',False))
for name,raw,pos,R,mode,nonzero in specs:
 mapped=mapping[raw];prefix=torch.full((3,),2,dtype=torch.int64);history=torch.cat((prefix,mapped))[pos:pos+R+3].reshape(1,-1)
 x=ns['streams'](len(raw),4,5120,19)[pos:pos+R].contiguous()
 if nonzero:x=(((torch.arange(R*4*5120,dtype=torch.int64)*73+19)%2047-1023).float()/512).reshape(R,4,5120).cuda()
 group=dict(name=name,rows=R,position=pos,mode=mode,nonzero=nonzero,tokens=save(name+'.tokens',raw[pos:pos+R]),history=save(name+'.history',history),x=save(name+'.x',x));groups.append(group)
 for layer,(h,w,norms)in models.items():
  ids=h.compute_ngram_ids(history,R).reshape(-1);assert ids.numel()==R*24
  unique,inverse=ids.unique(sorted=True,return_inverse=True);planes=[]
  for suffix in ['weight','scale']:
   key=f'layers.{layer}.engram.embed.{suffix}'
   with safe_open(str(root/index[key]),framework='pt',device='cpu')as reader:
    sl=reader.get_slice(key);t=torch.cat([sl[int(i):int(i)+1]for i in unique],dim=0)
   planes.append(t.cuda())
  decoded=torch.empty((R*24,256),device='cuda',dtype=torch.bfloat16);local=inverse.cuda().reshape(-1,1)
  module._engram_lookup_int4_kernel[(32,)](*planes,local,decoded,0,len(unique),R*24,1,1,0,1,1,256,32,8,32)
  emb=decoded.half().reshape(R,6144);result=ns['execute'](x,emb,w,norms['k_weight'],norms['q_weight']);repeat=ns['execute'](x,emb,w,norms['k_weight'],norms['q_weight']);torch.cuda.synchronize()
  for key,t in result.items():assert torch.isfinite(t).all() and torch.equal(t.view(torch.uint8),repeat[key].view(torch.uint8)),(name,layer,key)
  tag=f'{name}.l{layer}';case=dict(group=name,layer=layer,rows=R,position=pos,ids=save(tag+'.ids',ids),bf16=save(tag+'.bf16',decoded),emb=save(tag+'.emb',emb))
  case.update({key:save(tag+'.'+key,t)for key,t in result.items()});cases.append(case)
  print(name,layer,'captured',flush=True)
libraries={}
for line in Path('/proc/self/maps').read_text().splitlines():
 f=line.split()[-1]
 if 'libcublas'in f and f.startswith('/')and f not in libraries:libraries[f]=sha(Path(f).read_bytes())
manifest=dict(profile='Explicit Diffbot BF16 lookup -> pinned Exllama F16 math; SAME mapped history across layers1/14',groups=groups,cases=cases,weights=weights,files=files,libraries=libraries,sources=old['sources'],binaries=old['binaries'],token_map_sha256=sha(mapblob),hash_layout_sha256=sha((hashed/'layout.json').read_bytes()),source_hash_methods='Unchanged original NGramEmbedding._shift_right_ignore_eos/compute_ngram_ids',math_producer_sha256=sha((parent/'capture.py').read_bytes()),torch=torch.__version__,torch_git=torch.version.git_version)
emit('reference.json',(json.dumps(manifest,indent=2)+'\n').encode());print('CAPTURED',len(groups),len(cases),flush=True)
