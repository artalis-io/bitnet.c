"""Capture immutable ByteLevel vocabulary references from tokenizers 0.23.3.

The actual checkpoint outputs are rechecked against the pinned original Engram
capture. Synthetic cases use the Rust decoder directly; no candidate imports.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import tokenizers

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('checkpoint',type=Path)
parser.add_argument('engram_reference',type=Path)
parser.add_argument('output',type=Path)
parser.add_argument('--unit-header',type=Path)
a=parser.parse_args()
assert tokenizers.__version__=='0.23.3'
manifest_raw=(a.engram_reference/'manifest.json').read_bytes()
assert hashlib.sha256(manifest_raw).hexdigest()=='92875f3309c982a04a8d7bbea0221f729454af3c4be387019e610e97df8778ae'
original=json.loads(manifest_raw)
def original_file(name):
    data=(a.engram_reference/name).read_bytes()
    assert hashlib.sha256(data).hexdigest()==original[name]['sha256']
    return data
provenance=json.loads(original_file('provenance.json'))
checkpoint=(a.checkpoint/'tokenizer.json').read_bytes()
assert hashlib.sha256(checkpoint).hexdigest()==provenance['checkpoint_files']['tokenizer.json']
a.output.mkdir(exist_ok=True,parents=True);files={}
def emit(name,data):
    target=a.output/name
    if target.exists():assert target.read_bytes()==data,'refusing to replace differing reference: '+str(target)
    else:target.write_bytes(data)
    files[name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
def node_count(v):
    return 1+sum(node_count(x)for x in (v.values()if isinstance(v,dict)else v))if isinstance(v,(dict,list))else 1
def blob(prefix,field,strings):
    data=bytearray();offsets=[0]
    for s in strings:data.extend(s.encode('utf-8'));offsets.append(len(data))
    emit(prefix+'.'+field,bytes(data));emit(prefix+'.'+field+'.offsets',struct.pack('<'+'Q'*len(offsets),*offsets))
def corpus(prefix,raws,decoded):
    blob(prefix,'raw',raws);blob(prefix,'decoded',decoded)
    flags=bytes(int('\ufffd'in x)for x in decoded);emit(prefix+'.replacement',flags)
    return {'prefix':prefix,'vocabulary':len(raws),'replacement_tokens':sum(flags),'max_decoded_bytes':max(map(lambda s:len(s.encode()),decoded))}
tok=tokenizers.Tokenizer.from_file(str(a.checkpoint/'tokenizer.json'));assert tok.get_vocab_size()==129280
captured=[json.loads(line)for line in original_file('token-keys.jsonl').decode().splitlines()];assert len(captured)==129280
raws=[];decoded=[];keys=[]
for i,row in enumerate(captured):
    assert row[0]==i
    text=tok.decode([i],skip_special_tokens=False);assert text==row[1]and('\ufffd'in text)==row[3]
    raws.append(tok.id_to_token(i));decoded.append(text);keys.append(row[2])
cases=[corpus('actual',raws,decoded)];blob('actual','keys',keys);emit('actual.json',checkpoint)
cases[0]['json_nodes']=node_count(json.loads(checkpoint));assert cases[0]['replacement_tokens']==1484
# Canonical byte-to-Unicode alphabet supplies inputs; all outputs come from Rust.
visible=list(range(33,127))+list(range(161,173))+list(range(174,256));mapping={b:chr(b)for b in visible};n=256
for b in range(256):
    if b not in mapping:mapping[b]=chr(n);n+=1
def piece(data):return ''.join(mapping[b]for b in data)
raws=[piece(b'')]+[piece(bytes([b]))for b in range(256)]+[piece(bytes([x,y]))for x in range(256)for y in range(256)]
raws += [piece(bytes([0xe0,0xa0,b]))for b in range(256)]+[piece(bytes([0xf0,0x9f,0x92,b]))for b in range(256)]
raws += [piece(bytes(x))for x in [(0xed,0xa0,0x80),(0xf0,0x8f,0xbf,0xbf),(0xf4,0x90,0x80,0x80),(0xf4,0x8f,0xbf,0xbf)]]
raws += ['\U0001f600','<｜▁pad▁｜>','\ufffd','\u00a0','\u00ad','\u200d',' ','\n']
assert len(set(raws))==len(raws)
decoder=tokenizers.decoders.ByteLevel();decoded=[decoder.decode([s])for s in raws]
pair_doc={'model':{'type':'BPE','vocab':{s:i for i,s in enumerate(raws)},'merges':[]},'added_tokens':[],'decoder':{'type':'ByteLevel'}}
pair_json=(json.dumps(pair_doc,ensure_ascii=False,separators=(',',':'))+'\n').encode();emit('pairs.json',pair_json)
cases.append(corpus('pairs',raws,decoded));cases[-1]['json_nodes']=node_count(pair_doc)
# Small default unit fixture, including identical core/added ID overlap.
unit_raw=[piece(x)for x in [b'\0',b'\xc3',b'\xc3\xa9',b' hi',b'\xe2\x82',b'\xf0\x9f\x92',b'\xef\xbf\xbd',b'']]+['\U0001f600','<｜▁pad▁｜>','\u00a0','<|text|>']
unit_doc={'model':{'type':'BPE','vocab':{s:i for i,s in enumerate(unit_raw[:10])}},'added_tokens':[{'id':0,'content':unit_raw[0]},{'id':10,'content':unit_raw[10]},{'id':11,'content':unit_raw[11]}],'decoder':{'type':'ByteLevel'}}
if a.unit_header:
    lines=['/* Original Rust tokenizers 0.23.3 ByteLevel outputs; regenerate with reference_native_vocab.py. */','static const char native_vocab_fixture_json[] = '+json.dumps(json.dumps(unit_doc,ensure_ascii=True,separators=(',',':')))+';']
    for i,s in enumerate(unit_raw):
        for field,data in [('raw',s.encode()),('decoded',decoder.decode([s]).encode())]:
            lines.append('static const unsigned char native_vocab_'+field+'_'+str(i)+'[] = {'+','.join(str(b)for b in data+b'\0')+'};')
    lines += ['typedef struct {const unsigned char *raw,*decoded;size_t raw_bytes,decoded_bytes;int replacement;} NativeVocabReference;','static const NativeVocabReference native_vocab_reference[] = {']
    for i,s in enumerate(unit_raw):
        text=decoder.decode([s]);lines.append('    {native_vocab_raw_'+str(i)+',native_vocab_decoded_'+str(i)+','+','.join(map(str,[len(s.encode()),len(text.encode()),int('\ufffd'in text)]))+'},')
    lines.append('};');data=('\n'.join(lines)+'\n').encode()
    if a.unit_header.exists():assert a.unit_header.read_bytes()==data,'refusing to replace differing unit reference'
    else:a.unit_header.write_bytes(data)
metadata={'cases':cases,'files':files,'tokenizers':tokenizers.__version__,'checkpoint_tokenizer_sha256':hashlib.sha256(checkpoint).hexdigest(),'original_manifest_sha256':hashlib.sha256(manifest_raw).hexdigest(),'scope':'Rust ByteLevel decoder and raw vocabulary only; actual decoded strings match the pinned original Engram capture. No candidate runtime import.'}
emit_manifest=(json.dumps(metadata,indent=2)+'\n').encode();dest=a.output/'reference.json'
if dest.exists():assert dest.read_bytes()==emit_manifest,'refusing to replace differing reference manifest'
else:dest.write_bytes(emit_manifest)
print(json.dumps(cases),flush=True)
