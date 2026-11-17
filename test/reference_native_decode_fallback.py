"""Independent Rust ByteLevel whole-token fallback and Python UTF8 step oracle."""
from pathlib import Path
import argparse,codecs,hashlib,json
import tokenizers
p=argparse.ArgumentParser();p.add_argument('fixture',type=Path);p.add_argument('output',type=Path);p.add_argument('header',type=Path);a=p.parse_args()
assert tokenizers.__version__=='0.23.3'
raw=a.fixture.read_bytes();assert hashlib.sha256(raw).hexdigest()=='2b40c6947f1e794294b55dbb57fea20929573cbc8b9ffb4c4b96e08b612a3f73'
j=json.loads(raw);visible=list(range(33,127))+list(range(161,173))+list(range(174,256));alphabet={b:chr(b)for b in visible};n=256
for b in range(256):
 if b not in alphabet:alphabet[b]=chr(n);n+=1
inverse={c:b for b,c in alphabet.items()}
def convert(s):return bytes(inverse[c]for c in s)if all(c in inverse for c in s)else s.encode()
def literal(s):return '"'+''.join('\\x%02x'%b for b in s)+'"'if s else'""'
rows=[]
# Include all 256 Latin-1 characters with a mapped non-ASCII character, in both
# orders, plus unmapped Unicode and valid empty/all-mapped controls.
pieces=['Ġ😀','😀Ġ','Ġ�','é😀','Ā😀','',*['Ġ'+chr(i)for i in range(256)],*[chr(i)+'Ġ'for i in range(256)]]
for piece in pieces:
 d=json.loads(raw);del d['model']['vocab'][''];d['model']['vocab'][piece]=256
 # Avoid duplicate spellings of an existing core token.
 if len(d['model']['vocab'])!=257:continue
 t=tokenizers.Tokenizer.from_str(json.dumps(d,ensure_ascii=True));assert t.get_vocab_size()==261
 for seq in [[256],[0xc3,256,0xa9],[0xed,0xa0,256,0x80]]:
  dec=codecs.getincrementaldecoder('utf-8')(errors='replace');chunks=[];states=[]
  for ident in seq:
   data=convert(t.id_to_token(ident));chunks.append(dec.decode(data,final=False).encode());states.append(dec.getstate()[0])
  chunks.append(dec.decode(b'',final=True).encode());states.append(b'');assert b''.join(chunks)==t.decode(seq,skip_special_tokens=False).encode()
  rows.append(dict(replacement=json.dumps(piece,ensure_ascii=True)+':256',ids=seq,chunks=[c.hex()for c in chunks],pending=[s.hex()for s in states]))
header='/* Independent Rust whole-token fallback; Python incremental UTF8 steps. */\nstatic const struct {const char *replacement;int ids[4];size_t count;const unsigned char *chunks[5];size_t lengths[5];unsigned char pending[5][4];} native_stream_fallback_cases[]={\n'
for row in rows:
 chunks=[bytes.fromhex(c)for c in row['chunks']];states=[bytes.fromhex(s)for s in row['pending']]
 header+='{'+literal(row['replacement'].encode())+',{'+','.join(map(str,row['ids']))+'},'+str(len(row['ids']))+',{'+','.join('(const unsigned char*)'+literal(c)for c in chunks)+'},{'+','.join(str(len(c))for c in chunks)+'},{'+','.join('{'+','.join(map(str,bytes([len(s)])+s.ljust(3,b'\0')))+'}'for s in states)+'}},\n'
header+='};\n'
payload=(json.dumps(dict(tokenizers=tokenizers.__version__,fixture_sha256=hashlib.sha256(raw).hexdigest(),cases=rows),ensure_ascii=True,separators=(',',':'))+'\n').encode()
for path,data in [(a.output,payload),(a.header,header.encode())]:
 if path.exists():assert path.read_bytes()==data,'refusing differing reference '+str(path)
 else:path.write_bytes(data)
print('WHOLE TOKEN FALLBACK CAPTURED',len(rows))
