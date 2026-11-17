"""Independent Rust tokenization/decoding gold for native generation control tests."""
import argparse,hashlib,json
from pathlib import Path
import tokenizers
p=argparse.ArgumentParser();p.add_argument('fixture',type=Path);p.add_argument('header',type=Path);a=p.parse_args()
assert tokenizers.__version__=='0.23.3'
raw=a.fixture.read_bytes();assert hashlib.sha256(raw).hexdigest()=='3cb2995d5587e7ef277154390e758906ba98837e0a0c519550fa136de24c09e0'
t=tokenizers.Tokenizer.from_str(raw.decode());assert t.get_vocab_size()==315
ids=[195,t.token_to_id('<tag>'),169,0,237,160]
assert ids==[195,313,169,0,237,160]
def literal(b):return '"'+''.join('\\x%02x'%v for v in b)+'"'if b else'""'
s='/* Independent Rust full-decode prefixes, including early stop/flush. */\nstatic const int native_text_control_ids[]={'+','.join(map(str,ids))+'};\n'
s+='static const struct {int skip,limit;const unsigned char *bytes;size_t count;} native_text_control_cases[]={\n'
for skip in [False,True]:
 for limit in range(len(ids)+1):
  out=t.decode(ids[:limit],skip_special_tokens=skip).encode();s+='{'+str(int(skip))+','+str(limit)+',(const unsigned char*)'+literal(out)+','+str(len(out))+'},\n'
s+='};\n'
if a.header.exists():assert a.header.read_text()==s,'refusing differing header'
else:a.header.write_text(s)
print('CONTROL PREFIX SOURCE',len(ids),14)
