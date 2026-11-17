"""Independent real-weight first-two-block reference; no synthetic branches.

Actual embedding, Engram, HC, attention, routing, selected expert decode and
shared FFN. Prefix only: not complete-model logits or generation. argv[1] is
an immutable output directory. Run with the matched source cuBLAS preload.
"""
from pathlib import Path
import hashlib
import sys

hcfile=Path(__file__).with_name('reference_native_hc_stage.py')
assert hashlib.sha256(hcfile.read_bytes()).hexdigest()=='abae8805f8f8f5ef4913cbabacc6a57e4a8510f698a2cbbdde724ed5366f7a4a'
exec(compile(hcfile.read_text().split('with safe_open(str(root / index[\'embed.weight\'])')[0],str(hcfile),'exec'))
attfile=Path('/tmp/bitnet-goal-native-attention-producer/checkpoint-capture.py')
assert sha(attfile.read_bytes())=='161ddb424ba15d6ec9ccbaea33e90153caa69dbdaf8f2c640fa0858b773368b0'
atext=attfile.read_text().split('frames=')[0]
assert atext.count('assert not d.exists();d.mkdir()')==1
atext=atext.replace('assert not d.exists();d.mkdir()','d=out')
ans={'__file__':str(attfile),'out':out}
exec(compile(atext,str(attfile),'exec'),ans)
decfile=Path('/tmp/bitnet-goal-native-decode/capture.py')
assert sha(decfile.read_bytes())=='3960bb78dd5a21230a976ed99db51c56479ab8ea78b102aaa036e4436b8fe52e'
dtext=decfile.read_text().split('gate=oldload(')[0]
assert dtext.count('assert not outdir.exists();outdir.mkdir()')==1
dtext=dtext.replace('assert not outdir.exists();outdir.mkdir()','outdir=out')
dn={'__file__':str(decfile),'out':out}
exec(compile(dtext,str(decfile),'exec'),dn)
expert_provenance=[]

def matrix(name):
    m=next(v for v in catalog if v['name']==name)
    f=core/(m['tag']+'.packed-f16');b=f.read_bytes();assert sha(b)==m['packed_f16_sha256']
    return torch.frombuffer(bytearray(b),dtype=torch.float16).reshape(m['cols'],m['rows']).cuda()

def attention(layer,x):
    R=x.shape[0];freq=ans['freqs']['main']
    qa=x@ans['matrix'](layer,'wq_a');qres=torch.empty_like(qa)
    assert not norm.reference(qa.data_ptr(),ans['param'](layer,'q_norm').data_ptr(),qres.data_ptr(),R,1280,16,16,0,1,1e-20,0.,1.,stream)
    q=(qres@ans['matrix'](layer,'wq_b')).reshape(R,64,512)
    kv=(x@ans['matrix'](layer,'wkv')).reshape(R,1,512)
    rope=ans['rope']
    assert not rope.ref_rope(q.data_ptr(),freq.data_ptr(),None,None,R,64,64,512,448,64,0,0,0,1e-20,stream)
    assert not rope.ref_rope(kv.data_ptr(),freq.data_ptr(),ans['param'](layer,'kv_norm').data_ptr(),None,R,1,512,512,0,64,448,0,1,1e-20,stream)
    kv=kv.reshape(R,512)
    # Both initial layers are sliding-only, with empty pre-update rings.
    ring=torch.zeros((768,512),dtype=torch.float16,device='cuda')
    sink=parameter(f'layers.{layer}.attn.attn_sink')
    pool=ans['pools'][2]
    grouped=ans['g']['dsa_attn'](q,pool[0],pool[1],ans['bt'],sinks=sink,ring=ring,kv_chunk=kv,
        win_len=128,win_floor=0,ring_beg=0,indices=None,k_len=0,pool_len=0,q_pos0=0,
        compress_rate=1,derot_inv_freq=-freq,groups=8,
        out=torch.empty((8,R,4096),dtype=torch.float16,device='cuda'),page_size=256)
    wa=ans['matrix'](layer,'wo_a');Linear=ans['Linear']
    obj=ans['types'].SimpleNamespace(o_groups=8,woa_multi_ready=True,wo_a_multi=None,
        wo_a=[Linear(4096,1024)for _ in range(8)],wo_b=Linear(8192,5120),out_dtype=torch.float16)
    for i in range(8):obj.wo_a[i].weight=wa[:,i*1024:(i+1)*1024].contiguous()
    obj.wo_b.weight=ans['matrix'](layer,'wo_b')
    return ans['project'](obj,grouped.unsqueeze(1),{},torch.float32).reshape(R,5120)

def ffn(layer,x):
    D,H,E,K=5120,2304,384,6;R=x.shape[0];libs=dn['libs'];load=dn['oldload']
    gate=load(f'l{layer:02}.gate',torch.float16).reshape(E,D);center=load(f'l{layer:02}.center',torch.float16)
    wi8=torch.empty((2,E,D),dtype=torch.int8,device='cuda');sb=torch.empty(E,device='cuda')
    assert not libs['router'].prepare(gate.data_ptr(),wi8.data_ptr(),sb.data_ptr(),E,D,stream)
    S=libs['router'].slices(R,E,D);ahi=torch.empty((R,D),dtype=torch.int8,device='cuda');alo=torch.empty_like(ahi)
    sa=torch.empty((R,D//128),device='cuda');part=torch.empty((S,R,E),device='cuda')
    logits=torch.empty((R,E),dtype=torch.float16,device='cuda');ids=torch.empty((R,K),dtype=torch.int64,device='cuda');rw=torch.empty((R,K),dtype=torch.float16,device='cuda')
    assert not libs['router'].run(*(t.data_ptr()for t in [x,gate,wi8,sb,center,logits,ids,rw,ahi,alo,sa,part]),R,D,E,K,1.5,stream)
    selected=sorted(set(ids.flatten().cpu().tolist()));experts={}
    for e in selected:
        triples=[]
        for role,k,n in [('w1',D,H),('w3',D,H),('w2',H,D)]:
            prefix=f'layers.{layer}.ffn.experts.{e}.{role}';pieces=[]
            for suffix,dt,tag,shape in [('trellis',torch.uint16,'I16',[k//16,n//16,48]),('suh',torch.float16,'F16',[k]),('svh',torch.float16,'F16',[n])]:
                meta,b=dn['read_tensor'](prefix+'.'+suffix);assert meta['dtype']==tag and meta['shape']==shape
                expert_provenance.append(dict(tensor=prefix+'.'+suffix,bytes=len(b),sha256=sha(b)))
                pieces.append(torch.frombuffer(bytearray(b),dtype=dt).cuda())
            meta,b=dn['read_tensor'](prefix+'.mul1');assert dn['struct'].unpack('<I',b)[0]==0x83dcd12d
            expert_provenance.append(dict(tensor=prefix+'.mul1',bytes=len(b),sha256=sha(b)))
            triples.append(pieces)
        experts[e]=triples
    tables=[torch.tensor([experts[e][pr][pi].data_ptr()if e in experts else 0 for e in range(E)],dtype=torch.int64,device='cuda')for pr in range(3)for pi in range(3)]
    cap=8*K;ca=cap*(H//128);cb=8*(D//128);ctr=torch.zeros(ca+cb+2+(cap+1)+cap,dtype=torch.int32,device='cuda')
    hadg=torch.empty((cap,D),dtype=torch.float16,device='cuda');hadu=torch.empty_like(hadg)
    gg=torch.empty((cap,H),dtype=torch.float16,device='cuda');gu=torch.empty_like(gg);act=torch.empty_like(gg)
    down=torch.empty((cap,D),device='cuda');routed=torch.empty((8,D),device='cuda')
    args=[x,ids,rw,*tables,hadg,hadu,gg,gu,act,down,ctr,routed]
    arr=(ptr*len(args))(*(t.data_ptr()for t in args))
    assert not libs['decode'].ref_launch(arr,R,D,H,E,K,10.,stream)
    weights=[matrix(f'layers.{layer}.ffn.shared_experts.{r}')for r in ['w1','w3','w2']]
    sg=x@weights[0];su=x@weights[1];sharedact=torch.empty_like(sg)
    assert not libs['fold'].ref_act(sg.data_ptr(),su.data_ptr(),sharedact.data_ptr(),sharedact.numel(),10.,stream)
    shared=torch.empty((R,D),device='cuda')
    assert not libs['dense'].ref_dense(dn['handle'],sharedact.data_ptr(),weights[2].data_ptr(),shared.data_ptr(),R,H,D,1)
    final=routed[:R]+shared;torch.cuda.synchronize()
    return final,ids,rw

# Parameter() above is for HC/raw BF16 tensors. Sinks are raw F32.
original_parameter=parameter
def parameter(name):
    if name.endswith('attn_sink'):
        v=parameters[name];b=(pr/(v['tag']+'.expected')).read_bytes();assert sha(b)==v['expected_sha256']
        return torch.frombuffer(bytearray(b),dtype=torch.float32).cuda()
    return original_parameter(name)

with safe_open(str(root/index['embed.weight']),framework='pt',device='cpu')as reader:embedding=reader.get_tensor('embed.weight')
raw=torch.frombuffer(bytearray((hashed/'hash-1-1-513-text-tokens.i64le').read_bytes()),dtype=torch.int64)
cases=[]
for R in [1,7]:
    tokens=raw[:R];history=torch.cat((torch.full((3,),2,dtype=torch.int64),mapping[tokens])).reshape(1,-1)
    x=embedding[tokens].float().unsqueeze(1).expand(R,4,5120).contiguous().cuda();pre=None
    tokenfile=save(f'r{R}.tokens',tokens)
    for layer in [0,1]:
        tag=f'r{R}.l{layer}';boundaries={}
        if layer in models:x=engram(layer,x,history,R)
        for kind in ['attn','ffn']:
            name=f'layers.{layer}.hc_{kind}_';fn,base,scale=[parameter(name+s)for s in ['fn','base','scale']]
            part=torch.empty((R,hc.chunks(R,5120),25),device='cuda');post=torch.empty((R,4),device='cuda');comb=torch.empty((R,4,4),device='cuda');collapsed=torch.empty((R,5120),device='cuda');own=torch.empty((R,4),device='cuda');f=fn.half()
            assert not hc.mix(x.data_ptr(),f.data_ptr(),base.data_ptr(),scale.data_ptr(),part.data_ptr(),post.data_ptr(),comb.data_ptr(),collapsed.data_ptr(),pre.data_ptr()if pre is not None else None,own.data_ptr(),R,5120,1,stream)
            normalized=normalize(collapsed,parameter(f'layers.{layer}.{kind}_norm.weight'))
            boundaries[kind+'_normalized']=save(tag+'.'+kind+'.normalized',normalized)
            if kind=='attn':y=attention(layer,normalized)
            else:
                y,ids,rw=ffn(layer,normalized);boundaries['ids']=save(tag+'.ids',ids);boundaries['weights']=save(tag+'.weights',rw)
            boundaries[kind+'_delta']=save(tag+'.'+kind+'.delta',y)
            assert not hc.apply(x.data_ptr(),y.data_ptr(),post.data_ptr(),comb.data_ptr(),R,5120,0,stream)
            pre=own
            boundaries[kind+'_streams']=save(tag+'.'+kind+'.streams',x)
        assert torch.isfinite(x).all()
        cases.append(dict(rows=R,layer=layer,tokens=tokenfile,outputs=boundaries));print('CAPTURED actual block',R,layer,flush=True)
libraries={}
for line in Path('/proc/self/maps').read_text().splitlines():
    f=line.split()[-1]
    if 'libcublas'in f and f.startswith('/')and f not in libraries:libraries[f]=sha(Path(f).read_bytes())
emit('reference.json',(json.dumps(dict(scope='Actual two-block prefix with real attention and learned-router FFN; no full-model logits/generation',cases=cases,files=files,libraries=libraries),indent=2)+'\n').encode())
dependencies=[hcfile,attfile,decfile,src/'modules/attention_fn/dsa_triton.py',src/'modules/attention_fn/smem.py',src/'modules/dsv4.py',src/'modules/quant/fp16.py']
binaries=[Path(hcroot/'reference.so'),Path(norm._name),Path(ans['rope']._name)]+[Path(v._name)for v in dn['libs'].values()]
proof=dict(commit='bd1f519172406a9560bd5c90e61000691bb69a07',
    sources={str(f):sha(f.read_bytes())for f in dependencies},binaries={str(f):sha(f.read_bytes())for f in binaries},
    experts=expert_provenance,attention_weights=ans['weights'],libraries=libraries,
    config_sha256=sha((root/'config.json').read_bytes()),index_sha256=sha((root/'model.safetensors.index.json').read_bytes()))
emit('provenance.json',(json.dumps(proof,indent=2)+'\n').encode())
print('CAPTURED actual prefix',len(cases),flush=True)
