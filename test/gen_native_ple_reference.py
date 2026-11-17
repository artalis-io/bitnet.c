"""Generate immutable model-free fixtures from independent source captures."""
import argparse
import hashlib
import json
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paired',type=Path)
    parser.add_argument('extra_paired',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    lines=['/* Independent pinned original PLE kernels + Torch, paired cuBLAS13.10. No candidate outputs. */',
           'typedef struct {int streams,dim,k,rows,nonzero;uint64_t hashes[9];} NativePLEFixture;',
           'static const NativePLEFixture native_ple_fixtures[]={']
    for root,nonzero in [(args.paired,0),(args.extra_paired,1)]:
        manifest=json.loads((root/'reference.json').read_text())
        for case in manifest['cases']:
            if not case['label'].startswith('s'):
                continue
            hashes=[]
            for key in ['emb','x','key','value','key_norm','query_norm','dot','delta','final']:
                name=case[key]
                data=(root/name).read_bytes()
                expected=manifest['files'][name]
                assert len(data)==expected['bytes'] and hashlib.sha256(data).hexdigest()==expected['sha256'],name
                h=14695981039346656037
                for byte in data:
                    h=((h^byte)*1099511628211)&((1<<64)-1)
                hashes.append(h)
            lines.append('{%d,%d,%d,%d,%d,{%s}},'%(case['streams'],case['dim'],case['k'],case['rows'],nonzero,
                         ','.join('UINT64_C(0x%016x)'%h for h in hashes)))
    lines.append('};')
    result=('\n'.join(lines)+'\n').encode()
    if args.output.exists():
        assert args.output.read_bytes()==result,'immutable fixture conflict'
    else:
        args.output.write_bytes(result)

if __name__=='__main__':
    main()
