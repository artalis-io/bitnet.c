"""Original GLM gated-pool cache writer and original indexer-cache gather.
This captures the F32-scale/64-state-page profile, including untouched bytes.
"""
from pathlib import Path
import argparse,torch,json,hashlib,inspect
import vllm
from vllm import _custom_ops
from vllm.models.glm5next.nvidia.ops import kpool_compress as ops
ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);ap.add_argument('--header',type=Path,required=True);a=ap.parse_args();a.directory.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);assert not ops.mqa_logits_use_mx_scales()
def save(n,t):
 b=t.contiguous().view(torch.uint8).cpu().numpy().tobytes();f=a.directory/n
 if f.exists():assert f.read_bytes()==b,n
 else:f.write_bytes(b)
 h=14695981039346656037
 for x in b:h=((h^x)*1099511628211)&((1<<64)-1)
 return h
records=[]
for rows in [0,1,2,63,64,65,129,1025,65537]:
 for seed in [0,1]:
  blocks=(rows+63)//64+2;capacity=blocks*64;i=torch.arange(rows*4*128,dtype=torch.int64)
  k=(((i*37+i//13+seed*11)%509-254).float()/128).bfloat16().reshape(rows,4,128).cuda();g=(((i*13+i//17+seed*3)%251-125).float()/(16 if seed==0 else 2)).bfloat16().reshape(rows,4,128).cuda();j=torch.arange(512,dtype=torch.int64);ape=(((j*19+seed*13)%127-63).float()/32+2**-17).reshape(4,128).cuda()
  rid=torch.arange(rows,dtype=torch.int64);slots=(rid*37+seed*11)%capacity;slots=torch.where(rid%11==3,-1,slots);slots=torch.where(rid%17==5,capacity+7,slots);valid=(slots>=0)&(slots<capacity);loc=torch.where(valid,slots,0).cuda();mask=valid.cuda();table=torch.arange(blocks-1,-1,-1,dtype=torch.int32).cuda().reshape(1,blocks);gather_rows=capacity-7
  def run():
   cache=torch.full((blocks,64,132),0xa5,device='cuda',dtype=torch.uint8)
   q,s=ops.kpool_compress_and_write_cache(cache,k,g,ape,loc,4,return_compressed=True,write_cache=False)
   ops.kpool_compress_and_write_cache(cache,k,g,ape,loc,4,write_mask=mask)
   out=torch.empty((gather_rows,128),device='cuda',dtype=torch.float8_e4m3fn);scales=torch.empty((gather_rows,4),device='cuda',dtype=torch.uint8)
   _custom_ops.cp_gather_indexer_k_quant_cache(cache,out,scales,table,torch.tensor([0,gather_rows],device='cuda',dtype=torch.int32));torch.cuda.synchronize();return q,s,cache,out,scales
  first,repeat=run(),run();assert all(torch.equal(x.view(torch.uint8),y.view(torch.uint8))for x,y in zip(first,repeat));prefix=f'cache-{rows}-{seed}';record=dict(rows=rows,seed=seed,blocks=blocks,gather_rows=gather_rows)
  for name,value in zip(['values','scales','cache','gather','gather_scales'],first):record[name]=save(prefix+'-'+name+'.bin',value)
  records.append(record);print(rows,seed,'original writer/gather captured twice',flush=True)
lines=['/* Original GLM F32-scale paged cache writer and indexer gather. */','#ifndef BN_CUDA_GLM_FP8_CACHE_REFERENCE_H','#define BN_CUDA_GLM_FP8_CACHE_REFERENCE_H','typedef struct {int rows,seed,blocks,gather_rows;uint64_t values,scales,cache,gather,gather_scales;} BnGLMFP8CacheReference;','static const BnGLMFP8CacheReference glm_fp8_cache_reference[] = {']
for c in records:lines.append(' {%d,%d,%d,%d,UINT64_C(0x%016x),UINT64_C(0x%016x),UINT64_C(0x%016x),UINT64_C(0x%016x),UINT64_C(0x%016x)},'%tuple(c[k]for k in ['rows','seed','blocks','gather_rows','values','scales','cache','gather','gather_scales']))
a.header.write_text('\n'.join(lines+['};','#endif'])+'\n');source=Path(inspect.getfile(ops.kpool_compress_and_write_cache));extension=Path(vllm.__file__).parent/'_C_stable_libtorch.abi3.so';(a.directory/'reference.json').write_text(json.dumps(dict(records=records,source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),extension_sha256=hashlib.sha256(extension.read_bytes()).hexdigest(),profile='64-state pages; F32 scales; write-mask skips invalid physical slots; reverse physical block mapping; untouched page bytes retained',header_sha256=hashlib.sha256(a.header.read_bytes()).hexdigest()),indent=2)+'\n')
