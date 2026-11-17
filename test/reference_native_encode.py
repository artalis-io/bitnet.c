"""Independent tokenizers0.23.3 text/BPE and Unicode split references.
No candidate runtime imports; existing differing outputs are never replaced.
"""
import argparse,hashlib,json,struct,random,copy
from pathlib import Path
import tokenizers
p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('output',type=Path);p.add_argument('--unicode-header',type=Path);p.add_argument('--unit-header',type=Path);p.add_argument('--long-cases',action='store_true');p.add_argument('--edge-header',type=Path);p.add_argument('--boundary-only',action='store_true');a=p.parse_args()
assert tokenizers.__version__=='0.23.3'
a.output.mkdir(parents=True,exist_ok=True);files={}
def emit(name,data):
 f=a.output/name
 if f.exists():assert f.read_bytes()==data,'refusing differing reference '+str(f)
 else:f.write_bytes(data)
 files[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
def js(v):return (json.dumps(v,ensure_ascii=False,separators=(',',':'))+'\n').encode()
raw=(a.checkpoint/'tokenizer.json').read_bytes();assert hashlib.sha256(raw).hexdigest()=='c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b'
j=json.loads(raw);actual=tokenizers.Tokenizer.from_file(str(a.checkpoint/'tokenizer.json'));emit('actual.json',raw)
if a.boundary_only:
 import itertools
 alphabet=['a','A','é','\u0301','1','Ⅻ','!','、','$','😀',' ','\t','\r','\n','\u00a0','\u0085','中','・','\u3040','\x7f','\u200d','\u0378','\x00']
 inputs=bytearray();input_offsets=[0];ids=[];id_offsets=[0];ends=[];end_offsets=[0];count=0
 for width in range(5):
  for chars in itertools.product(alphabet,repeat=width):
   text=''.join(chars);inputs.extend(text.encode());input_offsets.append(len(inputs));ids.extend(actual.encode(text,add_special_tokens=False).ids);id_offsets.append(len(ids))
   ends.extend(len(text[:end].encode())for _,(_,end)in actual.pre_tokenizer.pre_tokenize_str(text));end_offsets.append(len(ends));count+=1
 emit('boundary.json',raw);emit('boundary.inputs',bytes(inputs))
 for suffix,values,fmt in [('input-offsets',input_offsets,'Q'),('ids',ids,'I'),('id-offsets',id_offsets,'Q'),('ends',ends,'I'),('end-offsets',end_offsets,'Q')]:emit('boundary.'+suffix,struct.pack('<'+fmt*len(values),*values))
 emit('manifest.json',js(dict(version=1,tokenizers=tokenizers.__version__,checkpoint_sha256=hashlib.sha256(raw).hexdigest(),cases=count,alphabet=alphabet,max_width=4,files=files)))
 print('BOUNDARY CAPTURED',count,len(ids),len(ends),flush=True);raise SystemExit(0)
# Runtime Regex is the authority for category/whitespace membership.
flags=bytearray(0x110000)
classes=[r'\p{L}',r'\p{M}',r'\p{N}',r'\p{P}',r'\p{S}',r'\s']
for bit,pattern in enumerate(classes):
 split=tokenizers.pre_tokenizers.Split(tokenizers.Regex(pattern),behavior='removed')
 for first in range(0,0x110000,4096):
  cps=[c for c in range(first,min(first+4096,0x110000))if not 0xd800<=c<=0xdfff];text=''.join(map(chr,cps));last=0
  for _,(begin,end)in split.pre_tokenize_str(text):
   for i in range(last,begin):flags[cps[i]]|=1<<bit
   last=end
  for i in range(last,len(cps)):flags[cps[i]]|=1<<bit
 print('Captured class',pattern,flush=True)
emit('unicode.flags',bytes(flags));ranges=[(0,flags[0])]
for i in range(1,len(flags)):
 if flags[i]!=flags[i-1]:ranges.append((i,flags[i]))
header='/* Generated from independent Rust tokenizers0.23.3 Regex captures.\n * Source: test/reference_native_encode.py; full scalar flags pinned in manifest. */\nstatic const struct {uint32_t start;unsigned char flags;} tokenizer_encode_unicode[]={\n'+''.join('{0x%x,%d},\n'%r for r in ranges)+'};\n'
if a.unicode_header:
 if a.unicode_header.exists():assert a.unicode_header.read_text()==header
 else:a.unicode_header.write_text(header)
# Synthetic full byte alphabet plus overlapping ranked merges and added tokens.
visible=list(range(33,127))+list(range(161,173))+list(range(174,256));alphabet={b:chr(b)for b in visible};n=256
for b in range(256):
 if b not in alphabet:alphabet[b]=chr(n);n+=1
vocab={alphabet[b]:b for b in range(256)};merges=[];pairs=set()
for word in ['ab','abc','aa','aaa','aaaa',' hello',' world','!abc','!!abc','123','456','789','你好','世界','é','e\u0301','\r\n','  ','\x00a','\u3040あ','カ・ナ']:
 pieces=[alphabet[b]for b in word.encode()];left=pieces[0]
 for right in pieces[1:]:
  value=left+right
  if value not in vocab:vocab[value]=len(vocab)
  if(left,right)not in pairs:merges.append(left+' '+right);pairs.add((left,right))
  left=value
s=copy.deepcopy(j);s['model']['vocab']=vocab;s['model']['merges']=merges;s['added_tokens']=[]
for text,special,normalized in [('<tag>',True,False),('ab',False,True),('<tag-long>',False,False)]:
 ident=vocab.get(text,len(vocab)+sum(x['content']not in vocab for x in s['added_tokens']))
 s['added_tokens'].append(dict(id=ident,content=text,single_word=False,lstrip=False,rstrip=False,normalized=normalized,special=special))
synthetic_json=js(s);synthetic=tokenizers.Tokenizer.from_str(synthetic_json.decode());emit('synthetic.json',synthetic_json)
base=['','hello world','1234567890','abc123def456','!abc !!abc ?hello',' !abc  !abc','你好世界abc123カ・ナ','éA e\u0301A','\x00a\x00','ababc<tag>x<tag-long>ab','a\r\n\t b','\u3040あ\u309f','a'*4096,'e'+'\u0301'*4096,'  '*4096]
base += [chr(c)for c in range(128)]
base += ['x'+chr(c)+'a'for c,_ in ranges if not 0xd800<=c<=0xdfff]
rng=random.Random(20261113);symbols='aA0 !?\n\r\t<>_é中あカ😀\u0301\u200d\u00a0\u2028\x00'
base += [''.join(rng.choice(symbols)for _ in range(rng.randrange(1,100)))for _ in range(1000)]
if a.long_cases:base += ['a'*131072,'e'+'\u0301'*65536,'hello world! '*16384]
for name,t in [('actual',actual),('synthetic',synthetic)]:
 texts=base[:]
 if name=='actual':texts += [actual.decode([i],skip_special_tokens=False)for i in range(actual.get_vocab_size())]
 texts += ['a'+x['content']+'b'for x in json.loads(t.to_str())['added_tokens']]
 # Binary corpus avoids parser node overhead for every vocabulary token.
 inputs=bytearray();input_offsets=[0];ids=[];id_offsets=[0];ends=[];end_offsets=[0]
 for text in texts:
  data=text.encode();inputs.extend(data);input_offsets.append(len(inputs));encoded=t.encode(text,add_special_tokens=False);ids.extend(encoded.ids);id_offsets.append(len(ids))
  # Model pre-tokenizer alone, excluding added-vocabulary isolation.
  for _,(_,end)in t.pre_tokenizer.pre_tokenize_str(text):ends.append(len(text[:end].encode()))
  end_offsets.append(len(ends))
 emit(name+'.inputs',bytes(inputs))
 for suffix,values,fmt in [('input-offsets',input_offsets,'Q'),('ids',ids,'I'),('id-offsets',id_offsets,'Q'),('ends',ends,'I'),('end-offsets',end_offsets,'Q')]:emit(name+'.'+suffix,struct.pack('<'+fmt*len(values),*values))
 print('Captured',name,len(texts),'cases',len(ids),'IDs',flush=True)
# Exhaustive scalar contexts exercise pre-tokenizer category decisions separately.
for context in ['A{}B',' {}a','!{}z','\n{} ']:
 for first in range(0,0x110000,4096):
  text=''.join(context.format(chr(c))for c in range(first,min(first+4096,0x110000))if not 0xd800<=c<=0xdfff)
  data=text.encode();ends=[len(text[:end].encode())for _,(_,end)in actual.pre_tokenizer.pre_tokenize_str(text)]
  # Outputs chunked so native replay can use a bounded buffer.
  prefix='scalars-%d-%06x'%( ['A{}B',' {}a','!{}z','\n{} '].index(context),first)
  emit(prefix+'.input',data);emit(prefix+'.ends',struct.pack('<'+'I'*len(ends),*ends))
manifest=dict(version=1,tokenizers=tokenizers.__version__,checkpoint_sha256=hashlib.sha256(raw).hexdigest(),regex_patterns=classes,unicode_ranges=len(ranges),unicode_header_sha256=hashlib.sha256(header.encode()).hexdigest(),files=files)
emit('manifest.json',js(manifest));print('CAPTURED',len(files),'files',flush=True)

if a.unit_header:
 def literal(data):
  return '\n'.join('"'+''.join('\\x%02x'%b for b in data[i:i+48])+'"'for i in range(0,len(data),48)) if data else '""'
 unit='/* Independent Rust tokenizers0.23.3 capture; do not hand-edit values. */\nstatic const char native_encode_fixture[]=\n'+literal(synthetic_json)+';\n'
 curated=base[:12]+[' a  b','a\r\n b','\u0301','!abc',' abc','é中A','abc<tag>ab','abc<tag-long>abc','1234567',' 123abc','\u3040あ\u309f','\u200d\u200da']
 variant=copy.deepcopy(json.loads(synthetic_json));variant['model']['merges']=[x.split(' ')for x in variant['model']['merges']];array_model=tokenizers.Tokenizer.from_str(json.dumps(variant,ensure_ascii=False))
 unit+='static const struct {const unsigned char *text;size_t bytes;int ids[64];size_t count;uint32_t ends[64];size_t pieces;} native_encode_cases[]={\n'
 for text in curated:
  encoded=synthetic.encode(text,add_special_tokens=False).ids;assert encoded==array_model.encode(text,add_special_tokens=False).ids
  ends=[len(text[:end].encode())for _,(_,end)in synthetic.pre_tokenizer.pre_tokenize_str(text)];assert len(encoded)<=64 and len(ends)<=64
  unit+='{(const unsigned char*)'+literal(text.encode())+','+str(len(text.encode()))+',{'+(','.join(map(str,encoded))or'0')+'},'+str(len(encoded))+',{'+(','.join(map(str,ends))or'0')+'},'+str(len(ends))+'},\n'
 unit+='};\n'
 if a.unit_header.exists():assert a.unit_header.read_text()==unit
 else:a.unit_header.write_text(unit)
 print('UNIT CAPTURED',len(curated),hashlib.sha256(unit.encode()).hexdigest(),flush=True)

if a.edge_header:
 def edge_literal(data):
  return '"'+''.join('\\x%02x'%b for b in data)+'"' if data else '""'
 cases=[chr(c)+'A\u0301'for c in range(128)]+['\x7fa\u0301b','\x7fabc\u0301d','~A\u0301\u0301','{a\u0301','\x7f\u0301A']
 edge='/* Independent Rust ASCII-prefix / mark-tail boundary regressions. */\nstatic const struct {const unsigned char *text;size_t bytes;int ids[64];size_t count;uint32_t ends[64];size_t pieces;} native_encode_edges[]={\n'
 for text in cases:
  ids=synthetic.encode(text,add_special_tokens=False).ids;ends=[len(text[:end].encode())for _,(_,end)in synthetic.pre_tokenizer.pre_tokenize_str(text)]
  edge+='{(const unsigned char*)'+edge_literal(text.encode())+','+str(len(text.encode()))+',{'+(','.join(map(str,ids))or'0')+'},'+str(len(ids))+',{'+(','.join(map(str,ends))or'0')+'},'+str(len(ends))+'},\n'
 edge+='};\n'
 if a.edge_header.exists():assert a.edge_header.read_text()==edge
 else:a.edge_header.write_text(edge)
 print('EDGE CAPTURED',len(cases),hashlib.sha256(edge.encode()).hexdigest(),flush=True)
