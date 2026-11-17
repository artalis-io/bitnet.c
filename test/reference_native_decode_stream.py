"""Independent ByteLevel stream references: Python incremental UTF8, checked
against complete Rust tokenizers0.23.3 decode for every sequence. No candidate imports.
Chunk contract is append-safe UTF8, not Rust DecodeStream's replacement heuristic.
"""
import argparse,codecs,hashlib,json,random,struct,itertools
from pathlib import Path
import tokenizers
from tokenizers.decoders import DecodeStream
p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('output',type=Path);p.add_argument('--unit-header',type=Path);p.add_argument('--error-header',type=Path);a=p.parse_args();assert tokenizers.__version__=='0.23.3'
a.output.mkdir(parents=True,exist_ok=True);files={}
def emit(name,data):
 f=a.output/name
 if f.exists():assert f.read_bytes()==data,'refusing differing reference '+str(f)
 else:f.write_bytes(data)
 files[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
def js(value):return(json.dumps(value,ensure_ascii=False,separators=(',',':'))+'\n').encode()
def packed(values,fmt):return struct.pack('<'+fmt*len(values),*values)
raw=(a.checkpoint/'tokenizer.json').read_bytes();assert hashlib.sha256(raw).hexdigest()=='c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b'
actual=tokenizers.Tokenizer.from_file(str(a.checkpoint/'tokenizer.json'));emit('actual.json',raw)
visible=list(range(33,127))+list(range(161,173))+list(range(174,256));alphabet={b:chr(b)for b in visible};n=256
for b in range(256):
 if b not in alphabet:alphabet[b]=chr(n);n+=1
inverse={c:b for b,c in alphabet.items()}
def token_bytes(text):
 return bytes(inverse[c]for c in text)if all(c in inverse for c in text)else text.encode()
j=json.loads(raw);s=dict(j);s['model']=dict(j['model']);s['model']['vocab']={alphabet[i]:i for i in range(256)};s['model']['vocab']['']=256;s['model']['merges']=[]
s['added_tokens']=[]
for i,(text,special)in enumerate([('<s>',True),('<plain>',False),('\ufffd',False),('😀',False)]):s['added_tokens'].append(dict(id=257+i,content=text,single_word=False,lstrip=False,rstrip=False,normalized=False,special=special))
synthetic_json=js(s);synthetic=tokenizers.Tokenizer.from_str(synthetic_json.decode());emit('synthetic.json',synthetic_json)
# Arbitrary complete UTF8 error-prefix boundaries and valid split characters.
byte_cases=[[],[0xc3,0xa9],[0xe2,0x82,0xac],[0xf0,0x9f,0x98,0x80],[0xe2],[0xf0,0x90],[0xf0,0x90,0x80],[0xef,0xbf,0xbd]]
for lead in [0xc0,0xc1,0xc2,0xdf,0xe0,0xe1,0xed,0xef,0xf0,0xf1,0xf4,0xf5,0xff]:
 for tail in itertools.product([0,0x7f,0x80,0x8f,0x90,0x9f,0xa0,0xbf,0xc0,0xff],repeat=3):byte_cases.append([lead,*tail])
records=[];unit=[];diagnostics=[]
for prefix,t in [('actual',actual),('synthetic',synthetic)]:
 count=t.get_vocab_size();pieces=[token_bytes(t.id_to_token(i))for i in range(count)];special={i for i,v in t.get_added_tokens_decoder().items()if v.special};byteids=[t.token_to_id(alphabet[i])for i in range(256)];assert all(i is not None for i in byteids)
 for i in range(count):assert pieces[i].decode(errors='replace')==t.decode([i],skip_special_tokens=False)
 sequences=[[i]for i in range(count)]+[[byteids[x],byteids[y]]for x in range(256)for y in range(256)]
 sequences += [[byteids[b]for b in v]for v in byte_cases]
 if special:
  ident=min(special);sequences += [[byteids[0xc3],ident,byteids[0xa9]],[byteids[0xf0],ident,byteids[0x90],ident,byteids[0x80],ident,byteids[0x80]]]
 rng=random.Random(20261114);sequences += [[rng.randrange(count)for _ in range(rng.randrange(0,48))]for _ in range(2048)]
 sequences += [[byteids[0xc3],byteids[0xa9]]*4096,[byteids[0x80]]*8192,[byteids[0xf0],byteids[0x9f],byteids[0x98],byteids[0x80]]*2048]
 ids=[];offsets=[0];flags=[];outputs=bytearray();out_offsets=[0];step_offsets=[0];case_steps=[0];pending=bytearray()
 for seq in sequences:
  for skip in [False,True]:
   ids.extend(seq);offsets.append(len(ids));flags.append(skip);decoder=codecs.getincrementaldecoder('utf-8')(errors='replace');whole=bytearray();chunks=[];states=[]
   for ident in seq:
    data=b''if skip and ident in special else pieces[ident];chunk=decoder.decode(data,final=False).encode();state=decoder.getstate()[0];assert len(state)<=3
    outputs.extend(chunk);whole.extend(chunk);step_offsets.append(len(outputs));pending.extend(bytes([len(state)])+state.ljust(3,b'\0'));chunks.append(chunk);states.append(state)
   last=decoder.decode(b'',final=True).encode();outputs.extend(last);whole.extend(last);step_offsets.append(len(outputs));pending.extend(b'\0'*4);chunks.append(last);states.append(b'')
   assert whole==t.decode(seq,skip_special_tokens=skip).encode(),(prefix,seq,skip)
   out_offsets.append(len(outputs));case_steps.append(len(step_offsets)-1)
   if prefix=='synthetic'and len(unit)<64 and seq in [[0xc3,0xa9],[0xe2,0x82,0xac],[0xf0,0x9f,0x98,0x80],[0xe2],[0xef,0xbf,0xbd],[],[0xc3,257,0xa9],[0xf0,257,0x90,257,0x80,257,0x80],[259],[256]]:
    unit.append(dict(ids=seq,skip=skip,chunks=chunks,states=states))
 for name,data in [('ids',packed(ids,'I')),('id-offsets',packed(offsets,'Q')),('skip',bytes(flags)),('outputs',bytes(outputs)),('output-offsets',packed(out_offsets,'Q')),('step-offsets',packed(step_offsets,'Q')),('case-steps',packed(case_steps,'Q')),('pending',bytes(pending))]:emit(prefix+'.'+name,data)
 records.append(dict(prefix=prefix,vocabulary=count,cases=len(flags),ids=len(ids),steps=len(step_offsets)-1,output_bytes=len(outputs)))
 for seq in [[byteids[0xef],byteids[0xbf],byteids[0xbd]],[byteids[0xc3],byteids[0xa9]]]:
  stream=DecodeStream(skip_special_tokens=False);diagnostics.append(dict(prefix=prefix,ids=seq,source_steps=[stream.step(t,i)for i in seq],source_full=t.decode(seq,skip_special_tokens=False)))
 print('CAPTURED',records[-1],flush=True)
manifest=dict(version=1,tokenizers=tokenizers.__version__,checkpoint_sha256=hashlib.sha256(raw).hexdigest(),contract='Python incremental UTF8 replacement, final checked against Rust full decode; no Rust DecodeStream chunk-timing equivalence',records=records,diagnostics=diagnostics,files=files);emit('manifest.json',js(manifest))
if a.unit_header:
 def literal(data):return '"'+''.join('\\x%02x'%b for b in data)+'"'if data else'""'
 header='/* Independent Python UTF8 step reference, full output verified with Rust. */\nstatic const char native_stream_fixture[]=\n'+ '\n'.join(literal(synthetic_json[i:i+48])for i in range(0,len(synthetic_json),48))+';\n'
 header+='static const struct {int ids[8];size_t count;int skip;const unsigned char *chunks[9];size_t lengths[9];unsigned char pending[9][4];} native_stream_cases[]={\n'
 for row in unit:
  assert len(row['ids'])<=8
  header+='{{'+(','.join(map(str,row['ids']))or'0')+'},'+str(len(row['ids']))+','+str(int(row['skip']))+',{'+','.join('(const unsigned char*)'+literal(x)for x in row['chunks'])+'},{'+','.join(str(len(x))for x in row['chunks'])+'},{'+','.join('{'+','.join(map(str,bytes([len(x)])+x.ljust(3,b'\0')))+'}'for x in row['states'])+'}},\n'
 header+='};\n'
 if a.unit_header.exists():assert a.unit_header.read_text()==header
 else:a.unit_header.write_text(header)
 print('UNIT CAPTURED',len(unit),flush=True)

if a.error_header:
 def literal_error(data):return '"'+''.join('\\x%02x'%b for b in data)+'"'if data else'""'
 cases=[[0xed,0xa0],[0xed,0xbf],[0xed,0xa0,0x80],[0xed,0xa0,0],[0xed,0xa0,0xc3,0xa9],[0xed,0xa0,256,0x80],[0xed,0xa0,257,0x80],[0xed,0xa0,259]]
 special={i for i,x in synthetic.get_added_tokens_decoder().items()if x.special}
 header='/* Independent surrogate-prefix timing/error references; full Rust checked. */\nstatic const struct {int ids[8];size_t count;int skip;const unsigned char *chunks[9];size_t lengths[9];unsigned char pending[9][4];} native_stream_error_cases[]={\n'
 for seq in cases:
  for skip in [False,True]:
   decoder=codecs.getincrementaldecoder('utf-8')(errors='replace');chunks=[];states=[]
   for ident in seq:
    data=b''if skip and ident in special else token_bytes(synthetic.id_to_token(ident));chunks.append(decoder.decode(data,final=False).encode());states.append(decoder.getstate()[0])
   chunks.append(decoder.decode(b'',final=True).encode());states.append(b'');assert b''.join(chunks)==synthetic.decode(seq,skip_special_tokens=skip).encode()
   header+='{{'+','.join(map(str,seq))+'},'+str(len(seq))+','+str(int(skip))+',{'+','.join('(const unsigned char*)'+literal_error(x)for x in chunks)+'},{'+','.join(str(len(x))for x in chunks)+'},{'+','.join('{'+','.join(map(str,bytes([len(x)])+x.ljust(3,b'\0')))+'}'for x in states)+'}},\n'
 header+='};\n'
 if a.error_header.exists():assert a.error_header.read_text()==header
 else:a.error_header.write_text(header)
 print('ERROR CAPTURED',len(cases)*2,flush=True)
