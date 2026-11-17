"""Independent real-text input and generation through the pinned native source graph.

Only fixture input changes: Rust raw text encoding replaces saved prompt IDs.
All inherited model math, routing, grouped geometry and SWA updates are retained.
The explicit Diffbot INT4 BF16 -> Exllama F16 Engram adapter still applies.
"""
from pathlib import Path
import hashlib,json,codecs,struct
import tokenizers

assert tokenizers.__version__=='0.23.3'
text_checkpoint=Path('/data/models/safetensors/deepseek-v4.1-flash-1017-diffbot-exl3-3bpw/tokenizer.json')
text_tokenizer_bytes=text_checkpoint.read_bytes()
assert hashlib.sha256(text_tokenizer_bytes).hexdigest()=='c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b'
text_tokenizer=tokenizers.Tokenizer.from_file(str(text_checkpoint))
text_paragraph=('Explain how to build a memory-bounded language model runtime. '
 'Separate immutable weights from request state; batch the prompt, preserve routing, '
 'and measure cold and warm expert transfers. Compare throughput and latency only '
 'with matched precision, context and concurrency.\n'
 'Multilingual notes: café and cafe\u0301; 中文模型需要正确处理文本边界。 '
 'A split UTF-8 character must remain intact 😀. Numbers: 123456789, 3.14159.\n')
text_long=''.join(f'Observation {i}: '+text_paragraph for i in range(24))
text_long_encoded=text_tokenizer.encode(text_long,add_special_tokens=False)
assert len(text_long_encoded.ids)>1026
text_prompt=text_long[:text_long_encoded.offsets[1025][1]]
text_prompt_ids=text_tokenizer.encode(text_prompt,add_special_tokens=False).ids
assert len(text_prompt_ids)==1026 and text_prompt_ids==text_long_encoded.ids[:1026]
text_prompt_bytes=text_prompt.encode()

text_parent=Path(__file__).with_name('reference_native_forward_resume.py')
assert hashlib.sha256(text_parent.read_bytes()).hexdigest()=='d6dbcb699a23763471b9469ece6f55fa641466913e58682f2cab343d9d784488'
text_marker="exec(compile(heavy_code, str(resume_parent), 'exec'))"
text_parent_source=text_parent.read_text();assert text_parent_source.count(text_marker)==1
text_setup,text_footer=text_parent_source.split(text_marker)
exec(compile(text_setup,str(text_parent),'exec'))
text_previous_source=index_source_text

def index_source_text(text):
    text=text_previous_source(text)
    old="raw=torch.frombuffer(bytearray((hashed/'hash-1-1-513-text-tokens.i64le').read_bytes()),dtype=torch.int64)"
    assert text.count(old)==1
    text=text.replace(old,'raw=torch.tensor(text_prompt_ids,dtype=torch.int64)')
    old='raw=torch.cat((raw.repeat(2),torch.full((5,),2,dtype=torch.int64)))'
    assert text.count(old)==1
    return text.replace(old,'raw=torch.cat((raw.clone(),torch.full((5,),2,dtype=torch.int64)))')

text_old_prompt='Original valid 513-token ID sequence repeated twice; five generated slots overwritten autoregressively'
assert text_footer.count(text_old_prompt)==1
text_footer=text_footer.replace(text_old_prompt,'Rust raw UTF8 prompt, 1026 independently encoded IDs; five generated slots overwritten autoregressively')
exec(compile(heavy_code,str(text_parent),'exec'))
exec(compile(text_footer,str(text_parent),'exec'))
text_generated=[]
for frame in frames:
    if frame.get('generated'):
        data=(out/frame['tokens']).read_bytes();assert len(data)==8
        text_generated.append(struct.unpack('<Q',data)[0])
assert len(text_generated)==5 and text_generated==raw[1026:1031].tolist()
text_visible=list(range(33,127))+list(range(161,173))+list(range(174,256));text_alphabet={b:chr(b)for b in text_visible};text_n=256
for b in range(256):
    if b not in text_alphabet:text_alphabet[b]=chr(text_n);text_n+=1
text_inverse={c:b for b,c in text_alphabet.items()}
def text_piece_bytes(ident):
    piece=text_tokenizer.id_to_token(ident)
    return bytes(text_inverse[c]for c in piece)if all(c in text_inverse for c in piece)else piece.encode()
text_special={i for i,v in text_tokenizer.get_added_tokens_decoder().items()if v.special}
text_outputs=[]
for skip in [False,True]:
    decoder=codecs.getincrementaldecoder('utf-8')(errors='replace');chunks=[];joined=b''
    for step,ident in enumerate(text_generated):
        chunk=decoder.decode(b''if skip and ident in text_special else text_piece_bytes(ident),final=False).encode()
        joined+=chunk;chunks.append(dict(file=emit(f'text.skip{int(skip)}.step{step}.utf8',chunk),bytes=len(chunk)))
    chunk=decoder.decode(b'',final=True).encode();joined+=chunk
    chunks.append(dict(file=emit(f'text.skip{int(skip)}.finish.utf8',chunk),bytes=len(chunk)))
    assert joined==text_tokenizer.decode(text_generated,skip_special_tokens=skip).encode()
    text_outputs.append(dict(skip_special=skip,chunks=chunks,full=emit(f'text.skip{int(skip)}.full.utf8',joined),bytes=len(joined)))
text_manifest=dict(scope='Independent Rust raw UTF8 encoding -> source 40-layer fresh513/resumed513 -> five self-fed tokens -> source-checked streaming UTF8; no automatic BOS/EOS/chat template or EOS stopping',
    parent_sha256=sha(text_parent.read_bytes()),producer_sha256=sha(Path(__file__).read_bytes()),
    tokenizer=dict(version=tokenizers.__version__,file=str(text_checkpoint),sha256=sha(text_tokenizer_bytes)),
    prompt=emit('text.prompt.utf8',text_prompt_bytes),prompt_bytes=len(text_prompt_bytes),
    ids=emit('text.prompt.ids-u32le',struct.pack('<'+'I'*len(text_prompt_ids),*text_prompt_ids)),token_count=len(text_prompt_ids),
    generated_ids=text_generated,outputs=text_outputs,
    profile_limitation='Explicit Diffbot INT4 BF16 lookup to Exllama F16 Engram adapter; not original Diffbot full-runtime equivalence',
    files=dict(files))
emit('text-reference.json',(json.dumps(text_manifest,indent=2)+'\n').encode())
print('CAPTURED REAL TEXT',len(text_prompt_bytes),len(text_prompt_ids),text_generated,flush=True)
