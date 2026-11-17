#include "native_moe_plan_reference.h"
static uint64_t native_moe_plan_hash(const void*data,size_t bytes){const unsigned char*p=data;uint64_t h=UINT64_C(14695981039346656037);for(size_t i=0;i<bytes;i++)h=(h^p[i])*UINT64_C(1099511628211);return h;}
static void test_native_moe_plan(void){
 for(size_t i=0;i<sizeof(native_moe_plan_fixtures)/sizeof(native_moe_plan_fixtures[0]);i++){
  const NativeMoEPlanFixture*f=&native_moe_plan_fixtures[i];size_t bytes=(size_t)(f->E+1)*sizeof(int);int*order=malloc(bytes),*starts=malloc(bytes),*base=malloc(bytes),*kind=malloc(bytes);BnMoEEXL3Group*groups=calloc((size_t)f->E,sizeof(*groups));assert(order&&starts&&base&&kind&&groups);BnMoEEXL3Plan p={order,starts,base,kind,groups,(size_t)f->E,(size_t)f->E,0,0,0,0};
  assert(!bn_moe_exl3_plan_groups(&p,f->counts,f->E,f->R,f->K,f->D,f->H,f->fused,256u<<20));
  assert(p.group_count==f->groups&&p.slot_count==f->slots&&p.fused_count==f->fused_count&&p.batch_cap==f->cap);
  assert(native_moe_plan_hash(starts,bytes)==f->starts&&native_moe_plan_hash(base,(size_t)f->E*4)==f->base&&native_moe_plan_hash(kind,(size_t)f->E*4)==f->kind&&native_moe_plan_hash(groups,(size_t)p.group_count*sizeof(*groups))==f->group_hash);
  BnMoEEXL3Plan saved=p;uint64_t guard=native_moe_plan_hash(starts,bytes);int*bad=malloc(bytes);assert(bad);memcpy(bad,f->counts,bytes);bad[f->E]=1;assert(bn_moe_exl3_plan_groups(&p,bad,f->E,f->R,f->K,f->D,f->H,f->fused,256u<<20)==-1&&!memcmp(&p,&saved,sizeof(p))&&native_moe_plan_hash(starts,bytes)==guard);bad[f->E]=0;bad[0]=-1;assert(bn_moe_exl3_plan_groups(&p,bad,f->E,f->R,f->K,f->D,f->H,f->fused,256u<<20)==-1);free(bad);
  assert(bn_moe_exl3_plan_groups(&p,f->counts,f->E,f->R,f->K,f->D+1,f->H,f->fused,256u<<20)==-1);assert(bn_moe_exl3_plan_groups(&p,f->counts,f->E,f->R,f->K,f->D,f->H,f->fused,0)==-1);
  free(order);free(starts);free(base);free(kind);free(groups);
 }
 puts("Native EXL3 source fused/batched tier planning, padding, slot layouts and rejection PASSED");
}
