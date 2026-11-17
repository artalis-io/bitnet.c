"""Generate generic Unicode tables for the pinned Engram key profile.

Only official Unicode 9 normalization/mark data and Unicode 17 lowercase/white
space data are inputs. No model vocabulary or candidate implementation is used.
"""
import argparse
from functools import lru_cache
import hashlib
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
hashes={'UnicodeData-9.0.0.txt':'68dfc414d28257b9b5d6ddbb8b466c768c00ebdf6cbf7784364a9b6cad55ee8f','UnicodeData-17.0.0.txt':'2e1efc1dcb59c575eedf5ccae60f95229f706ee6d031835247d843c11d96470c','SpecialCasing-17.0.0.txt':'efc25faf19de21b92c1194c111c932e03d2a5eaf18194e33f1156e96de4c9588','PropList-17.0.0.txt':'130dcddcaadaf071008bdfce1e7743e04fdfbc910886f017d9f9ac931d8c64dd','LICENSE-UNICODE.txt':'e7a93b009565cfce55919a381437ac4db883e9da2126fa28b91d12732bc53d96'}
for name,h in hashes.items():assert hashlib.sha256((a.source/name).read_bytes()).hexdigest()==h,name
rows={int(f[0],16):f for f in(line.split(';')for line in(a.source/'UnicodeData-9.0.0.txt').read_text().splitlines())}
marks={cp for cp,f in rows.items()if f[2].startswith('M')}
assert all(not int(f[3])or cp in marks for cp,f in rows.items())
lower={}
for line in(a.source/'UnicodeData-17.0.0.txt').read_text().splitlines():
 f=line.split(';')
 if f[13]:lower[int(f[0],16)]=[int(f[13],16)]
for line in(a.source/'SpecialCasing-17.0.0.txt').read_text().splitlines():
 f=line.split('#')[0].strip().split(';')
 if len(f)>=5 and not f[4].strip():lower[int(f[0],16)]=[int(x,16)for x in f[1].split()]
@lru_cache(None)
def decompose(cp):
 if 0xac00<=cp<0xd7a4:
  n=cp-0xac00;return [0x1100+n//588,0x1161+n%588//28]+([0x11a7+n%28]if n%28 else [])
 f=rows.get(cp);d=f[5].split()if f else []
 if d and d[0].startswith('<'):d=d[1:]
 return [y for x in d for y in decompose(int(x,16))]if d else [cp]
entries=[];values=[]
for cp in sorted(set(rows)|set(lower)):
 if 0xd800<=cp<=0xdfff or 0xac00<=cp<0xd7a4:continue
 folded=[y for x in decompose(cp)if x not in marks for y in lower.get(x,[x])]
 if folded==[cp]:continue
 assert len(folded)<256 and all(y<=0x10ffff and not 0xd800<=y<=0xdfff for y in folded)
 entries.append((cp,len(values),len(folded)));values+=folded
white=[]
for line in(a.source/'PropList-17.0.0.txt').read_text().splitlines():
 f=line.split('#')[0].strip().split(';')
 if len(f)>1 and f[1].strip()=='White_Space':
  ends=[int(x,16)for x in f[0].strip().split('..')];white+=list(range(ends[0],ends[-1]+1))
assert len(white)==25
lines=['/* Generic Unicode9 NFKD/mark removal + Unicode17 per-character lowercase.',
       ' * All retained pre-lowercase characters have canonical class zero:',
       ' * canonical mark ordering cannot affect the retained sequence.',
       ' * Hangul decomposition is algorithmic. No model-specific entries.']
for n,h in hashes.items():lines.append(' * '+n+' SHA256 '+h)
lines+=[' */','/*\n'+(a.source/'LICENSE-UNICODE.txt').read_text().strip()+'\n */','typedef struct {uint32_t cp,offset;unsigned char count;} TokenizerMapFoldEntry;','static const uint32_t tokenizer_map_fold_values[] = {']
for i in range(0,len(values),12):lines.append('    '+','.join(hex(x)for x in values[i:i+12])+',')
lines+=['};','static const TokenizerMapFoldEntry tokenizer_map_folds[] = {']
lines+=['    {'+hex(cp)+','+str(offset)+','+str(n)+'},'for cp,offset,n in entries]
lines+=['};','static const uint32_t tokenizer_map_white_space[] = {'+','.join(hex(x)for x in white)+'};']
data=('\n'.join(lines)+'\n').encode()
if a.output.exists():assert a.output.read_bytes()==data,'refusing to replace differing generated table'
else:a.output.write_bytes(data)
print('entries',len(entries),'values',len(values),'max_expansion',max(x[2]for x in entries),'bytes',len(data),'sha256',hashlib.sha256(data).hexdigest())
