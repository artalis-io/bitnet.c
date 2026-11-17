"""Independent IEEE F16 -> F32 last-logit gold from immutable source logits."""
import hashlib
import json
from pathlib import Path
import struct
import sys

root=Path(sys.argv[1]);manifest=json.loads((root/'reference.json').read_text());files={};frames={}
for frame in manifest['frames']:
    outputs={}
    for key in ['logits','last_logits']:
        name=frame[key];blob=(root/name).read_bytes()
        assert hashlib.sha256(blob).hexdigest()==manifest['files'][name]['sha256']
        tail=blob[-129280*2:]
        converted=b''.join(struct.pack('<f',value)for(value,)in struct.iter_unpack('<e',tail))
        target=frame['name']+'.'+key+'.host-f32'
        path=root/target
        if path.exists():assert path.read_bytes()==converted,target
        else:path.write_bytes(converted)
        files[target]=dict(bytes=len(converted),sha256=hashlib.sha256(converted).hexdigest());outputs[key]=target
    frames[frame['name']]=outputs
data=(json.dumps(dict(method='Python struct IEEE binary16 decode then binary32 encode; original source logits only, no candidate values',frames=frames,files=files),indent=2)+'\n').encode()
path=root/'host-reference.json'
if path.exists():assert path.read_bytes()==data
else:path.write_bytes(data)
print('HOST SOURCE GOLD',len(frames),flush=True)
