"""Independent Unicode/key/map references from unchanged source and Rust0.23.3."""
import argparse,ast,hashlib,json,struct
from functools import lru_cache
from pathlib import Path
import tokenizers
from tokenizers import normalizers,Regex
p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('engram_reference',type=Path);p.add_argument('output',type=Path);p.add_argument('--unit-header',type=Path);p.add_argument('--skip-unicode',action='store_true');a=p.parse_args()
assert tokenizers.__version__=='0.23.3'
raw=a.source.read_bytes();assert hashlib.sha256(raw).hexdigest()=='18e0dc275d300e68ee09e51d890a029907dfd32a2e1c13a53f1b2cc0a3a48aff'
tree=ast.parse(raw);fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='engram_token_map')
assignment=next(x for x in fn.body if isinstance(x,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='normalizer'for t in x.targets))
ns={'normalizers':normalizers,'Regex':Regex};exec(compile(ast.fix_missing_locations(ast.Module(body=[assignment],type_ignores=[])),str(a.source),'exec'),ns);norm=ns['normalizer']
parent_raw=(a.engram_reference/'manifest.json').read_bytes();assert hashlib.sha256(parent_raw).hexdigest()=='92875f3309c982a04a8d7bbea0221f729454af3c4be387019e610e97df8778ae';parent=json.loads(parent_raw)
a.output.mkdir(exist_ok=True,parents=True);files={}
def emit(name,data):
 target=a.output/name
 if target.exists():assert target.read_bytes()==data,'refusing to replace differing reference: '+str(target)
 else:target.write_bytes(data)
 files[name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
def put_strings(prefix,strings):
 blob=bytearray();offsets=[0]
 for s in strings:blob.extend(s.encode());offsets.append(len(blob))
 emit(prefix,bytes(blob));emit(prefix+'.offsets',struct.pack('<'+'Q'*len(offsets),*offsets))
# Source map is already independently captured for every actual vocabulary ID.
for name in ['token-map.i64le','token-keys.jsonl','provenance.json']:
 data=(a.engram_reference/name).read_bytes();assert hashlib.sha256(data).hexdigest()==parent[name]['sha256']
 if name=='token-map.i64le':emit('actual.map',data)
visible=list(range(33,127))+list(range(161,173))+list(range(174,256));alphabet={b:chr(b)for b in visible};nextcp=256
for b in range(256):
 if b not in alphabet:alphabet[b]=chr(nextcp);nextcp+=1
def piece(data):return ''.join(alphabet[b]for b in data)
texts=['A','a','É','e','é',' ','  ','\t\n','\u0301','\u09be',' x ','x','\ue000a',' a','ΟΣ','οσ','ος','\x00','','\x00a','\x00b',' \u0085','\u0085','\ue000','a\ue000 ','a ','\u1c89','\u1c8a','A\u1abf','A\u0301','\U0001ccd6']
raws=[piece(x.encode())for x in texts]+[piece(b'\xc3'),piece(b'\xef\xbf\xbd')];assert len(set(raws))==len(raws)
unit=tokenizers.Tokenizer(tokenizers.models.BPE(vocab={s:i for i,s in enumerate(raws)},merges=[]));unit.decoder=tokenizers.decoders.ByteLevel();emit('unit.json',unit.to_str(pretty=True).encode())
decoded=[unit.decode([i],skip_special_tokens=False)for i in range(len(raws))];keys=[raws[i]if'\ufffd'in text else norm.normalize_str(text)or text for i,text in enumerate(decoded)]
# Execute the unchanged full function itself for the synthetic compressed IDs.
import torch
exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(a.source),'exec'),{'torch':torch,'lru_cache':lru_cache},function_ns:={})
ids=function_ns['engram_token_map'](str(a.output/'unit.json')).tolist();assert len(ids)==len(raws)
put_strings('unit.keys',keys);emit('unit.map',struct.pack('<'+'Q'*len(ids),*ids));emit('unit.json',(a.output/'unit.json').read_bytes())
if a.unit_header:
 lines=['/* Unchanged source Engram map plus Rust0.23.3 key goldens. */','static const char native_tokenmap_fixture_json[] = '+json.dumps((a.output/'unit.json').read_text())+';']
 lines += ['enum {NATIVE_TOKENMAP_VOCAB='+str(len(raws))+',NATIVE_TOKENMAP_COMPRESSED='+str(max(ids)+1)+',NATIVE_TOKENMAP_PAD=5};','static const uint32_t native_tokenmap_ids[] = {'+','.join(map(str,ids))+'};']
 for i,s in enumerate(keys):lines.append('static const unsigned char native_tokenmap_key_'+str(i)+'[] = {'+','.join(map(str,s.encode()+b'\0'))+'};')
 lines+=['static const struct {const unsigned char *data;size_t bytes;} native_tokenmap_keys[] = {']
 lines+=['    {native_tokenmap_key_'+str(i)+','+str(len(s.encode()))+'},'for i,s in enumerate(keys)];lines.append('};');data=('\n'.join(lines)+'\n').encode()
 if a.unit_header.exists():assert a.unit_header.read_bytes()==data,'refusing to replace differing unit reference'
 else:a.unit_header.write_bytes(data)
if not a.skip_unicode:
 source=bytearray();target=bytearray();so=[0];to=[0];count=0
 def row(s):
  global count
  source.extend(s.encode());target.extend(norm.normalize_str(s).encode());so.append(len(source));to.append(len(target));count+=1
 for context in [False,True]:
  for cp in range(0x110000):
   if 0xd800<=cp<=0xdfff:continue
   s=chr(cp);row(' \ue000A'+s+'Σ\u0301\ue000 \t'if context else s)
 for s in ['', ' ', '  ', '\t\n', ' \u0085', '\u0085 ', '\ue000 ', ' \ue000', '\ue000a', 'a\ue000 ', '\x1ca\x1c', '\x00a\x00', 'A\u1abf', 'ΟΣ', 'A\u0301', '\u0130']:
  row(s)
 emit('unicode.source',bytes(source));emit('unicode.normalized',bytes(target));emit('unicode.source.offsets',struct.pack('<'+'Q'*len(so),*so));emit('unicode.normalized.offsets',struct.pack('<'+'Q'*len(to),*to))
else:count=None
metadata={'files':files,'source_sha256':hashlib.sha256(raw).hexdigest(),'parent_sha256':hashlib.sha256(parent_raw).hexdigest(),'unicode_cases':count,'unit_vocab':len(raws),'unit_compressed':max(ids)+1,'tokenizers':tokenizers.__version__,'scope':'Unchanged original Engram function and normalizer AST; no candidate runtime import or Unicode-table generator use.'}
data=(json.dumps(metadata,indent=2)+'\n').encode();dest=a.output/('reference-lite.json'if a.skip_unicode else'reference.json')
if dest.exists():assert dest.read_bytes()==data,'refusing to replace differing reference manifest'
else:dest.write_bytes(data)
print(json.dumps({k:metadata[k]for k in ['unicode_cases','unit_vocab','unit_compressed']}),flush=True)
