#ifndef BN_GPU_BACKEND_H
#define BN_GPU_BACKEND_H

#include <stddef.h>
#include <stdint.h>
#include "runtime_policy.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    BN_GPU_BACKEND_UNKNOWN = 0,
    BN_GPU_BACKEND_METAL = 1,
    BN_GPU_BACKEND_WEBGPU = 2,
    BN_GPU_BACKEND_CUDA = 3,
} BnGPUBackendKind;

// SSM Q/K normalization semantics, carried in SSM_L2NORM parameter p[4].
typedef enum {
    BN_GPU_SSM_QK_NORM_CLAMPED_L2 = 0,
    BN_GPU_SSM_QK_NORM_RMS_SCALED = 1,
} BnGPUSSMQKNormMode;

// Model-independent layout and math for projected prompt Q/K/V values.
// Frequency offsets refer to the backend's prepared activation frequency table.
typedef struct {
    int n_tokens, n_heads, n_kv_heads, head_size;
    int q_row_stride, q_gated, qk_norm_per_head;
    int reference_rmsnorm_order;
    int pos0, rope_dims;
    int attention_window; // 0 = full causal prompt; positive = visible key count
    size_t rope_freq_offset;
    float norm_eps, attention_scale;
    // For pos0 > 0, read earlier KV from the backend's session cache.
    // Offset identifies the first NEW row, in elements; stride is in elements.
    // Prepared attention returns current rows and does not modify the cache.
    size_t kv_cache_off;
    int kv_cache_stride;
} BnGPUAttentionPrefillPlan;

// Descriptor for one operation in a batched matvec submission.
typedef struct {
    float *out;      // host output pointer
    void  *W_buf;    // GPU weight handle (opaque, from buffer_create)
    int rows, cols, type;
} BnGPUMatvecOp;

// Borrowed backend weight handles for explicit BF16 GGUF linear arithmetic.
// type is the arithmetic input format (BF16 for explicitly staged matrices).
// decoded is required only when the
// backend-quant plan selects decoded GEMM; raw may then be NULL, allowing its
// GPU payload to be released. A8 plans require raw even if decoded is present.
typedef struct {
    void *raw;
    void *decoded;
    int type;
} BnGPUBF16LinearWeight;

typedef struct {
    BnGPUBF16LinearWeight gate, up, down;
} BnGPUBF16FFNWeights;

/* Request-owned mutable storage, distinct from immutable weight handles.
 * offset/count are elements of the declared GGUF scalar type (F32, F16, BF16, I32, I8).
 * Views expire when their request is released/reset or their allocation is
 * explicitly resized. slot is backend-private and may include generation bits.
 * Never mix request owners.
 */
typedef struct BnGPUResidentView {
    void *request;
    size_t offset, count;
    uint32_t slot;
    int type;
} BnGPUResidentView;

/* One projection inside a backend-owned pinned upload bundle. On success a
 * NULL buffer becomes a newly owned handle; an existing buffer must be idle
 * and exclusive. offsets/bytes describe disjoint pinned staging ranges. A NULL
 * source uses bytes already at that offset; otherwise source borrows ordinary
 * CPU memory for bytes until batch returns. It must not overlap any part of
 * the staging allocation. The backend copies it just before enqueueing H2D,
 * allowing the next CPU copy to overlap the preceding projection transfer. */
typedef struct {
    void *buffer;
    size_t offset,bytes;
    int format,k,n;
    uint32_t marker;
    const void *source;
} BnGPUTrellisUploadItem;

/* Dense FP16 input [tokens,K] times immutable packed FP16 [K,N]. Weight
 * must come from buffer_create_f16_rows on this backend. Output is FP16 or
 * FP32, with cuBLAS COMPUTE_32F/default math and full matrix geometry.
 * Views are exact, disjoint, request-owned tensors. Weight/backend must
 * outlive asynchronous consumers. No bias or implicit dtype conversion.
 */
typedef struct {
    void *weight;
    int k, n, n_tokens;
    BnGPUResidentView x, out;
} BnGPUF16LinearResidentPlan;

/* Explicit BF16 lookup -> F16 conv-less n-gram gate profile. This is not
 * eager BF16 PLE. lookup/embedding are [tokens,k], immutable packed weight is
 * F16[k,(streams+1)*dim]; key/value are TWO unequal strided column GEMMs with
 * F32 accumulation and F16 output. Raw BF16 norm weights [streams,dim],
 * vector4 RMS into F32, F32 batched dot, signed-sqrt/sigmoid shared-value gate,
 * and separate rounded F32 residual addition. All ten views are exact and
 * mutually disjoint. All are caller-owned; reserve linear_f16 resources first.
 * Input BF16 values must be finite and representable as finite F16; callers
 * establish payload validity before upload. Metadata rejection queues nothing;
 * execution errors invalidate the request. No allocation, copy or host fence.
 * This primitive does not commit history, position, KV or model execution. */
typedef struct {
    int n_tokens,k,dim,streams;
    float eps,gate_scale;
    void *weight,*norm_key,*norm_query;
    BnGPUResidentView lookup,embedding,stream,key,value,normalized_key,normalized_query,dot,delta,out;
} BnGPUPLEF16FromBF16ResidentPlan;

/* Group-specific F16 projections followed by a final linear. Group weight is
 * one immutable packed [K,G*N] buffer, preserving its leading dimension.
 * Input is F16[G,rows,K]. Each independent FP32-accumulate GEMM stores F16
 * directly into its columns of down.x F16[rows,G*N], then down emits F16/F32.
 * No weight duplication, group-output scratch or concatenation copy/kernel.
 * Reserve linear_f16 workspace first. Views exact, disjoint and request-owned;
 * weights/backend outlive asynchronous consumers. No allocation/transfer/fence. */
typedef struct {int width;size_t input_count,concat_count;} BnGPUF16GroupedOutputWorkspace;
static inline int bn_gpu_f16_grouped_output_workspace(int groups,int k,int n,int rows,
    BnGPUF16GroupedOutputWorkspace*out) {
    if(!out||groups<1||groups>64||k<1||n<1||rows<1||rows>16384||n>INT32_MAX/groups||
       (size_t)groups>SIZE_MAX/(size_t)k/2||(size_t)rows>SIZE_MAX/(size_t)groups/(size_t)k/2||
       (size_t)groups>SIZE_MAX/(size_t)n/(size_t)k/2||
       (size_t)rows>SIZE_MAX/(size_t)groups/(size_t)n/2)return -1;
    BnGPUF16GroupedOutputWorkspace w={groups*n,(size_t)groups*rows*k,(size_t)groups*rows*n};*out=w;return 0;
}
typedef struct {
    void *group_weight;
    int groups,k,n,n_tokens;
    BnGPUResidentView x;
    BnGPUF16LinearResidentPlan down;
} BnGPUF16GroupedOutputResidentPlan;

typedef enum {
    BN_GPU_F16_FFN_SILU_HALF = 0,
} BnGPUF16FFNActivationProfile;
/* Dense F16 gate/up, half-precision SiLU/product boundaries, F32 down output.
 * gate/up borrow identical input; all five live activation tensors are
 * disjoint. Hidden width is even for source half2 arithmetic. limit=0 disables
 * clipping, otherwise finite [0,65504]. No implicit residual addition or bias.
 * Uses linear_f16_reserve resources; validates the entire plan before work. */
typedef struct {
    BnGPUF16LinearResidentPlan gate, up, down;
    BnGPUF16FFNActivationProfile activation_profile;
    float act_limit;
} BnGPUF16FFNResidentPlan;

/* Explicit rotated-basis EXL3 projection: pre-scale RNh -> Hadamard F32 ->
 * RNh -> decoded FP16 GEMM with F32 accumulation -> RNh -> Hadamard F32 ->
 * RNh -> post-scale RNh. This profile does not select upstream adaptive
 * FP16-partial GEMM or fused original-basis reconstruction.
 * packed is I8 raw bytes [K/16,N/16,48*2]. suh[K], svh[N],
 * x/xh[tokens,K], decoded[K,N] are F16. gemm/out[tokens,N] must
 * both be F16 or both F32. F32 output preserves the original F32 Hadamard
 * and post-scale without intermediate FP16 rounding.
 * All views must be disjoint, correctly aligned, and owned by this request.
 * Caller budgets/allocates decoded/xh/gemm scratch outside execution; reuse
 * one projection's scratch between ordered expert calls. No model mutation.
 */
typedef struct {
    int format, k, n, n_tokens;
    uint32_t codebook_marker;
    BnGPUResidentView packed, suh, svh, x, xh, decoded, gemm, out;
    /* Optional immutable backend-owned projection handle. When non-NULL,
     * packed/suh/svh must be empty. Owner backend must outlive every request;
     * retain the handle until all asynchronous consumers complete. Mutable
     * request views remain disjoint and request-local. */
    void *weight;
} BnGPUTrellisProjectionF16ResidentPlan;

/* Explicit DQ expert profile. Gate/up produce F16, down produces F32.
 * Activation is FP16 SiLU (half exp/add/reciprocal/multiply), followed by
 * optional clipping: SiLU gate <= limit; up in [-limit,limit], then FP16
 * product into down.x. limit=0 disables clipping; otherwise finite [0,65504].
 * Gate/up share exactly the same input. Persistent input, gate/up outputs,
 * activation (down.x), and down output must be disjoint. Scratch views may
 * overlap only across projections, for sequential reuse; they must not
 * overlap persistent values or any of the nine weight/scale views.
 * All three projections and lifetime aliases validate before work is queued.
 * Uses trellis_projection_f16_reserve; no additional hidden scratch.
 */
typedef enum {
    BN_GPU_TRELLIS_FFN_SILU_HALF = 0,
} BnGPUTrellisFFNActivationProfile;
typedef struct {
    BnGPUTrellisProjectionF16ResidentPlan gate, up, down;
    BnGPUTrellisFFNActivationProfile activation_profile;
    float act_limit;
} BnGPUTrellisFFNResidentPlan;

/* Direct source fused EXL3 SiLU FFN with F16 nonlinear/clipping boundaries.
 * act_limit=0 disables clipping; otherwise source gate/up clipping applies.
 * Group width is an explicit FULL-tier policy:
 * never recompute it from an offload residency window (rounding changes).
 * At most32 borrowed expert triples per launch, unique IDs in [0,E).
 * Caller retains handles/cache leases until completion. Source row tiers:
 * M16 count1..16 (N128 or256),M32 count17..32,N128,M64 count33..capacity,N128.
 * capacity is64/128/256; all four temporary slabs are distinct F16 arrays.
 * Native index order is original assignment ID (not token ID), counts E+1,
 * sorted F16 route weights and host/GPU fused base table E must match it.
 * Metadata is immutable while queued. Resident experts outside the row tier
 * are skipped. Slots F32[slot_rows,D] are OVERWRITTEN only for eligible expert
 * rows, already weighted through the output Hadamard; use slot-kind1.
 * Cooperative launch enforces block residency; request owns private locks.
 * All structural validation precedes launch. Execution allocates/copies/fences
 * nothing. Setup/temporary-slab/lock budgets are explicit and separate. */
#define BN_GPU_FUSED_EXL3_MAX_EXPERTS 32
typedef struct {
    size_t state_count,intermediate_count,temporary_bytes;
} BnGPUFusedEXL3Workspace;
static inline int bn_gpu_fused_exl3_workspace(int groups,int capacity,int dim,int hidden,BnGPUFusedEXL3Workspace*out) {
    if(!out||groups<1||groups>64||(capacity!=64&&capacity!=128&&capacity!=256)||
       dim<128||hidden<128||dim>65536||hidden>65536||dim%128||hidden%128||
       dim>INT32_MAX/hidden||(size_t)groups*capacity>SIZE_MAX/4/(size_t)(dim+hidden))return -1;
    BnGPUFusedEXL3Workspace w={(size_t)groups*capacity*dim,(size_t)groups*capacity*hidden,(size_t)groups*capacity*(dim+hidden)*4};*out=w;return 0;
}
typedef struct {int smem_bytes,threads,blocks_per_sm,multiprocessors,max_groups;} BnGPUFusedEXL3LaunchInfo;
typedef struct {
    int format,tokens,topk,experts,dim,hidden,capacity,groups,group_width;
    int tile_m,tile_n,resident_count,slot_rows;
    uint32_t codebook_marker;
    int expert_ids[BN_GPU_FUSED_EXL3_MAX_EXPERTS];
    const void *weight[3][BN_GPU_FUSED_EXL3_MAX_EXPERTS];
    float act_limit;
    BnGPUResidentView x,counts,order,weights,slot_base;
    BnGPUResidentView state_g,state_u,intermediate_g,intermediate_u,slots;
} BnGPUFusedEXL3ResidentPlan;

/* Serialized cohort submission. Every plan retains its original tile and
 * full-tier group_width. The backend may schedule fewer independent expert
 * groups to fit cooperative residency, but never changes split-K arithmetic.
 * Views/locks remain request-owned; weights stay leased through completion of
 * ALL consumers. Owner must occur exactly once among 2..16 distinct requests.
 * All plans are validated before queueing. No forward allocation or host
 * activation bridge. Setup reserves fixed readiness/completion events.
 * Compatible plans share kernels; singleton buckets are counted explicitly.
 * Consumer/plan arrays are borrowed only until submission returns. */
#define BN_GPU_FUSED_EXL3_MAX_CONSUMERS 16
typedef struct {
    void *request;
    const BnGPUFusedEXL3ResidentPlan *plan;
} BnGPUFusedEXL3Consumer;
typedef struct {
    int n_consumers;
    const BnGPUFusedEXL3Consumer *consumers;
} BnGPUFusedEXL3GroupPlan;
typedef struct {
    uint64_t kernel_launches, joint_kernel_launches, consumers;
} BnGPUFusedEXL3GroupStats;

/* Small-batch routed EXL3 decode, distinct from fused prefill arithmetic.
 * Gate/up raw GEMV is F16; nonlinear/clipping and down output transforms are
 * F32. Down slots are reduced with the source 16-warp weighted order.
 * All selected nonzero-weight experts must be borrowed together. Bindings
 * may contain extra experts; unlisted/out-of-range IDs are masked. This is
 * not an accumulation API for residency windows. Weights outlive completion.
 * All views exact, disjoint, request-owned; no allocation, host copy or fence.
 * Counters must be zeroed ONCE before first use; source kernels reset them
 * for reuse. Do not zero while queued. Tables/indices64 are private metadata
 * bytes (int64 device arrays), overwritten by GPU binding each execution.
 * Scratch has slots_max=8*topk even when fewer rows execute; shrinking it
 * changes split-K policy. Shared dense experts are executed separately. */
#define BN_GPU_DECODE_EXL3_MAX_EXPERTS 48
typedef struct {
    size_t had_count,intermediate_count,down_count,counter_count,table_bytes,indices64_bytes,total_bytes;
} BnGPUDecodeEXL3Workspace;
static inline int bn_gpu_decode_exl3_workspace(int tokens,int topk,int experts,int dim,int hidden,BnGPUDecodeEXL3Workspace*out) {
    if(!out||tokens<1||tokens>8||topk<1||topk>32||experts<topk||experts>512||dim<128||hidden<128||dim>32768||hidden>65536||dim%128||hidden%128||dim>INT32_MAX/hidden)return -1;
    size_t s=(size_t)8*topk;
    BnGPUDecodeEXL3Workspace w={s*dim,s*hidden,s*dim,s*(hidden/128)+(size_t)8*(dim/128)+2+(s+1)+s,(size_t)9*experts*8,(size_t)tokens*topk*8,0};
    w.total_bytes=w.had_count*4+w.intermediate_count*6+w.down_count*4+w.counter_count*4+w.table_bytes+w.indices64_bytes;
    *out=w;return 0;
}
typedef struct {
    int format,tokens,topk,experts,dim,hidden,resident_count;
    uint32_t codebook_marker;
    int expert_ids[BN_GPU_DECODE_EXL3_MAX_EXPERTS];
    const void *weight[3][BN_GPU_DECODE_EXL3_MAX_EXPERTS];
    float act_limit;
    BnGPUResidentView x,indices,weights,tables,indices64;
    BnGPUResidentView had_gate,had_up,gate,up,act,down,counters,out;
} BnGPUDecodeEXL3ResidentPlan;

/* Native EXL3 assignment preparation. Input IDs I32[A], route weights F16[A],
 * A=tokens*topk. All metadata arrays I32; sorted_weights F16. counts/starts
 * have E+1 entries: invalid IDs map to sentinel E, which the host planner
 * rejects before weight admission. Small (A<=32) unstable sort follows the
 * source GPU network; larger arrays preserve original assignment order on
 * expert ties. Read counts once to plan source fused/batch tiers. Reserve
 * sorting workspace explicitly before execution; no per-call allocation or
 * host copies/fences. Views must be exact, disjoint and request-owned. */
typedef struct {
    int tokens,topk,experts;
    BnGPUResidentView indices,weights,keys,values,sorted_keys,order,inverse;
    BnGPUResidentView sorted_weights,counts,starts;
} BnGPUNativeMoEIndexPlan;
/* Group input gathering: padding is +0 half, equivalent to source sentinel
 * row. order holds sorted original assignment IDs, not token IDs. Metadata
 * comes from validated host counts/starts, passed by value as launch args. */
typedef struct {
    int tokens,topk,dim,batch,rows;
    int starts[16],counts[16];
    BnGPUResidentView x,order,out;
} BnGPUNativeMoEGroupPlan;
/* Fixed top-k-order slot accumulation, matching the original gather.
 * kind0 skips,kind1 already weighted,kind2 multiplies sorted F16 weights.
 * Existing F32 out is incremented; a token with no slots remains untouched.
 * Metadata must come from the matching index/tier plan. Bad device indices
 * are bounded defensively; structural errors reject before any launch. */
typedef struct {
    int tokens,topk,dim,experts,slot_rows;
    BnGPUResidentView indices,inverse,starts,slot_base,slot_kind,weights;
    BnGPUResidentView slots,out;
} BnGPUNativeMoEReducePlan;
static inline int bn_gpu_native_moe_shape(int tokens,int topk,int experts) {
    return tokens<1||tokens>16384||topk<1||topk>32||experts<topk||experts>65536?-1:0;
}

#define BN_GPU_FOLDED_EXL3_MAX_BATCH 16
/* Folded original-basis grouped projection, distinct from rotated DQ above.
 * Existing immutable MUL1 weight handles are borrowed, not decoded caches.
 * One batched folded reconstruction and one batched GEMM; no token loops.
 * Caller retains every handle/cache lease until all queued consumers finish.
 * Only weights[0..batch) are used. Exact request F16 x[B,M,K],decoded[B,K,N],
 * and F16/F32 out[B,M,N], all disjoint. One decoded slab may be reused across
 * sequential projections. B<=16,B*M<=16384,K/N128-divisible and <=65536.
 * No allocation, copy or synchronization during execution. */
typedef struct {
    int format,k,n,n_tokens,batch;
    uint32_t codebook_marker;
    const void *weights[BN_GPU_FOLDED_EXL3_MAX_BATCH];
    BnGPUResidentView x,decoded,out;
} BnGPUFoldedEXL3ResidentPlan;
typedef struct {
    size_t x_count,decoded_count,out_count;
} BnGPUFoldedEXL3Workspace;
static inline int bn_gpu_folded_exl3_workspace(int batch,int rows,int k,int n,BnGPUFoldedEXL3Workspace*out) {
    if(!out||batch<1||batch>BN_GPU_FOLDED_EXL3_MAX_BATCH||rows<1||rows>16384/batch||
       k<128||n<128||k>65536||n>65536||k%128||n%128||k>INT32_MAX/n||
       (size_t)batch>SIZE_MAX/(size_t)k/(size_t)n/2||
       (size_t)batch*rows>SIZE_MAX/(size_t)k/2||
       (size_t)batch*rows>SIZE_MAX/(size_t)n/4)return -1;
    BnGPUFoldedEXL3Workspace w={(size_t)batch*rows*k,(size_t)batch*k*n,(size_t)batch*rows*n};*out=w;return 0;
}
typedef struct {int fp16_partial32,tile_n,tuned;} BnGPUFoldedGemmInfo;
/* Expert group: up then gate, half SiLU/clipping, then down. Gate/up F16,
 * down F32; all three projections validate before any work. Gate/up share
 * x. Down.x is the activation, allowed to alias up.out exactly (in-place),
 * otherwise disjoint from persistent tensors. Decoded scratch may overlap
 * across projections, but never overlap any persistent input/output.
 * Outputs remain unweighted; route-slot accumulation is a separate stage. */
typedef struct {
    BnGPUFoldedEXL3ResidentPlan gate,up,down;
    BnGPUTrellisFFNActivationProfile activation_profile;
    float act_limit;
} BnGPUFoldedEXL3FFNResidentPlan;


/* Writable only between successful acquire and submit/cancel. A submitted
 * lease expires immediately; reacquire before modifying its pinned bytes.
 * Generation checks reject stale leases after slot reuse. Request owns memory.
 */
typedef struct {
    void *request, *data;
    size_t capacity;
    uint64_t generation;
    uint32_t slot;
} BnGPUHostUploadLease;

enum { BN_GPU_UPLOAD_BUSY = 1 };

/* Independent reference output kernels have two FMA contractions. Select a
 * numerical contract explicitly; never infer it from tensor values or length.
 * Both first round history_dot * exp(decay). LOCAL_FMA rounds its scaled
 * history before fusing local_dot * scale; HISTORY_FMA rounds scaled local
 * before fusing history * scale. Zero preserves existing callers. */
typedef enum {
    BN_GPU_GDN_OUTPUT_LOCAL_FMA = 0,
    BN_GPU_GDN_OUTPUT_HISTORY_FMA = 1
} BnGPUGDNOutputProfile;

typedef struct {
    int n_tokens, hk, dk, hv, dv;
    float q_scale;
    BnGPUGDNOutputProfile output_profile;
    /* Prepared planar Q/K [tokens,hk,dk], V/out [tokens,hv,dv] are BF16;
     * log_decay/beta [tokens,hv] and states [hv,dv,dk] are F32.
     * Each view describes exactly its logical tensor. Q/K/V starts must be
     * aligned to 16 bytes. Outputs must be disjoint from every input/each other.
     */
    BnGPUResidentView q, k, v, log_decay, beta;
    BnGPUResidentView initial_state, next_state, out;
} BnGPUGDNPrefillResidentPlan;

typedef enum {
    BN_GPU_GDN_DECAY_MULTIPLIER = 0,
    BN_GPU_GDN_DECAY_LOG = 1
} BnGPUGDNDecayDomain;

typedef enum {
    BN_GPU_GDN_PREFILL_CP = 0,
    /* Sequential 64-token blocks; decay participates in the rounded inverse.
     * This is a different numerical contract from context parallel prefill. */
    BN_GPU_GDN_PREFILL_NONCP = 1
} BnGPUGDNPrefillRecurrence;

/* Explicit SM120 context-parallel GDN numerical contract. Single sequence,
 * HK16/HV48, D128, BF16 Q/K/V/output, F32 beta/state. The alpha view holds
 * decay multipliers, or log-decay when decay_domain is LOG. LOG explicitly
 * selects CUDA expf staging, never the fast exponential.
 * chunk_tokens is a multiple of 64 in [64,4096]; n_tokens in [1,8192].
 * next_state/out must be disjoint from every input and each other. */
typedef struct {
    int n_tokens, chunk_tokens, hk, dk, hv, dv;
    float q_scale;
    BnGPUGDNDecayDomain decay_domain;
    BnGPUResidentView q, k, v, alpha, beta;
    BnGPUResidentView initial_state, next_state, out;
    /* Zero preserves CP arithmetic. NONCP reuses the reserved CP workspace
     * and chunk admission bounds, but executes one sequential state sweep. */
    BnGPUGDNPrefillRecurrence recurrence;
} BnGPUGDNCPPrefillResidentPlan;

/* Packed-prefill sequence boundary: activations belong to the coordinator;
 * only initial/next state belong to request. Distinct state owners, common
 * device/backend, and disjoint writes are required. */
typedef struct {
    void *request;
    BnGPUGDNCPPrefillResidentPlan plan;
} BnGPUGDNCPPrefillSequence;
typedef struct {
    int n_sequences;
    const BnGPUGDNCPPrefillSequence *sequences;
} BnGPUGDNCPPrefillGroupPlan;


typedef struct {
    int n_tokens, hk, dk, hv, dv;
    float norm_eps, softplus_threshold;
    void *a_log, *dt_bias; /* Borrowed immutable backend weight handles. */
    /* Packed BF16 X=[Q,K,V], BA=[beta,alpha]; planar BF16 q/k/v outputs
     * and F32 log_decay/beta. Exact tensor extents, all outputs disjoint from
     * inputs/each other. Preparation applies no query scale or recurrence. */
    BnGPUResidentView x, ba, q, k, v, log_decay, beta;
} BnGPUGDNPrepareResidentPlan;

/* Distinct independently qualified F32 recurrent normalization contracts.
 * PACKED preserves sqrt/divide and its local reduction. RSQRT uses a
 * 128-thread reduction and reciprocal square root, as in fused sigmoid update.
 * No state or normalized Q/K is rounded to BF16 by either decode profile. */
typedef enum {
    BN_GPU_GDN_DECODE_PACKED = 0,
    BN_GPU_GDN_DECODE_RSQRT = 1,
} BnGPUGDNDecodeProfile;
typedef struct {
    int hk, dk, hv, dv;
    float q_scale;
    void *a_log, *dt_bias;
    /* One raw packed BF16 QKV/BA token; F32 [hv,dv,dk] states.
     * Decode keeps normalized Q/K in F32, unlike BF16 prefill preparation.
     * Supports dk=dv=128, hv divisible by hk, fixed epsilon1e-6/threshold20.
     * Output and next state must be disjoint from all inputs/each other.
     * Previous state is never written; commit after successful completion. */
    BnGPUResidentView x, ba, initial_state, next_state, out;
    /* 0 or 1 preserves single-sequence decode. Otherwise [2,65535]
     * independent one-token sequences, packed sequence-major in every view.
     * Each sequence has its own F32 state; weights and geometry are shared.
     * This is not a multi-token recurrence. All views belong to the supplied
     * backend request (a batch workspace may own the packed state). */
    int n_sequences;
    BnGPUGDNDecodeProfile profile; /* zero preserves existing callers */
} BnGPUGDNDecodeResidentPlan;

typedef enum {
    BN_GPU_KDA_A_LOG = 0,
    BN_GPU_KDA_A_NEGATIVE_EXP = 1,
} BnGPUKDAParameterStorage;

typedef struct {
    int n_tokens, num_heads, head_dim; /* Equal K/V heads, head_dim=128. */
    float q_scale, lower_bound; /* Finite positive scale, negative gate floor. */
    BnGPUKDAParameterStorage parameter_storage;
    /* Immutable F32[heads] parameter and F32[heads,128] bias. Caller supplies
     * finite parameters/bias; NEGATIVE_EXP values must be nonpositive. Preserve
     * stored -exp(A_log) directly, without a log/exp round trip. */
    void *decay_parameter, *gate_bias;
    /* One request's contiguous chunk. BF16 x=[tokens,Q|K|V], gate=[tokens,H,K],
     * beta_raw=[tokens,H]; F32 states=[H,V,K]; BF16 out=[tokens,H,V].
     * Q/K normalize in F32 with sum-epsilon1e-6, never round before recurrence.
     * Per-key bounded decay, sigmoid beta, contracted local dot reductions,
     * F32 recurrent state, final BF16 core output before output norm/gating.
     * Outputs disjoint from every input/each other. Initial state immutable;
     * caller commits next_state after success. No allocation/copy/fence. */
    BnGPUResidentView x, gate, beta_raw, initial_state, next_state, out;
} BnGPUKDARecurrentResidentPlan;

/* One token per distinct, live participant request. F32 initial/next states
 * must belong to that participant. BF16 x/ba/out may belong to that participant
 * or to the explicit coordinator (packed projection rows). Every write is
 * disjoint from all other rows' reads/writes; shared read-only inputs are legal.
 * Unrelated request owners are never accepted. */
typedef struct {
    void *request;
    BnGPUResidentView x, ba, initial_state, next_state, out;
} BnGPUGDNDecodeSequence;
typedef struct {
    int hk, dk, hv, dv, n_sequences;
    float q_scale;
    void *a_log, *dt_bias;
    const BnGPUGDNDecodeSequence *sequences;
    BnGPUGDNDecodeProfile profile; /* uniform across the submitted group */
} BnGPUGDNDecodeGroupPlan;
enum { BN_GPU_GROUP_BUSY = 1 };
/* Public resident group operations, independent of backend shader IDs. */
typedef enum {
    BN_GPU_RESIDENT_GROUP_GDN,
    BN_GPU_RESIDENT_GROUP_CONV,
    BN_GPU_RESIDENT_GROUP_QK_PREP,
    BN_GPU_RESIDENT_GROUP_ATTENTION
} BnGPUResidentGroupKind;



typedef struct {
    int n_tokens, dim, kernel;
    /* Immutable [dim,kernel], kernel2/3/4, no bias. BF16 for the BF16
     * operation; F32 for causal_conv_f32_weight_bf16_resident. */
    void *weight;
    /* BF16 token-major input/output; history [kernel-1,dim], oldest first.
     * Outputs disjoint from every input/each other. Previous history remains
     * unchanged; commit next history only after successful completion. */
    BnGPUResidentView x, initial_history, next_history, out;
} BnGPUCausalConvResidentPlan;

/* One token per distinct participant. Histories remain participant-owned;
 * activation views may also belong to the explicit coordinator. */
typedef struct {
    void *request;
    BnGPUResidentView x, initial_history, next_history, out;
} BnGPUCausalConvSequence;
typedef struct {
    int n_sequences, dim, kernel;
    void *weight;
    const BnGPUCausalConvSequence *sequences;
} BnGPUCausalConvGroupPlan;
/* Explicit prefill boundaries, one positive token count per sequence.
 * Reuses the convolution group's bounded workspace and BUSY contract.
 * Histories remain participant-owned; activations may be coordinator-owned.
 * Legacy decode groups always retain one-token semantics. */
typedef struct {
    BnGPUCausalConvGroupPlan group;
    const int *token_counts;
} BnGPUCausalConvPrefillGroupPlan;



typedef struct {
    int n_tokens, dim, kernel, dilation; /* kernel2/3/4, dilation1..8. */
    void *weight; /* Immutable BF16[dim,kernel]. */
    /* Chronological BF16 history [(kernel-1)*dilation,dim]. x/residual/
     * outer/out are [tokens,dim]. Conv rounds before SiLU; residual sum
     * rounds before the outer addition. Outputs disjoint from all inputs
     * and each other. Commit next_history only after successful completion.
     * One request's contiguous chunk; prefill and decode share arithmetic. */
    BnGPUResidentView x, residual, outer, initial_history, next_history, out;
} BnGPUPLEConvResidentPlan;

typedef struct {
    int n_tokens, dim, streams; /* Qualified widths2560/4096, streams1/4. */
    float eps;
    void *norm_key, *norm_query, *norm_conv; /* BF16[streams,dim], 1+gamma. */
    /* BF16 key/hidden [tokens,streams,dim], value [tokens,dim].
     * Caller-owned normalized_key/query are scratch [tokens,streams,dim].
     * gated/normed have the same wide shape. All four writable views must
     * be disjoint from every other view. Inputs and weights remain unchanged.
     * Eager BF16 materializations and contiguous F32 reductions are retained.
     * Validation rejection queues nothing; no allocation/copy/internal fence. */
    BnGPUResidentView key, value, hidden, normalized_key, normalized_query, gated, normed;
} BnGPUPLEGateResidentPlan;

typedef enum {
    BN_GPU_GATED_NORM_SILU = 0,
    BN_GPU_GATED_NORM_SIGMOID = 1,
} BnGPUGatedNormActivation;

typedef enum {
    BN_GPU_GATED_NORM_FLA = 0,
    /* Fixed 16 lanes x 8 columns, FP32 libdevice exp and direct SiLU
     * division. Caller selects the compiled arithmetic contract explicitly;
     * no inference from actual batch size or compiler/autotune defaults.
     * Initial contract is SiLU, 128 columns, BF16 gamma/input/output. */
    BN_GPU_GATED_NORM_COMPILED_R2 = 1,
    /* GLM KDA output: fixed two rows/warp, eight columns/lane, F32 gamma,
     * approximate sqrt followed by division, sigmoid gate, BF16 output.
     * Selected explicitly by the F32-weight entry point; no batch heuristic. */
    BN_GPU_GATED_NORM_GLM_R2 = 2,
} BnGPUGatedNormProfile;

typedef struct {
    int rows, dim; /* Initial qualified profile:128 columns per row. */
    float eps;
    BnGPUGatedNormActivation activation;
    BnGPUGatedNormProfile profile; /* Zero preserves the FLA contract. */
    /* Shared immutable gamma[dim], BF16 or F32 according to the entry point.
     * No bias/weight offset. GLM F32 operation requires shape [1,128]. */
    void *weight;
    /* out = rmsnorm(x) * activation(z), all intermediates F32, final BF16.
     * Exact BF16 extents; output disjoint from both read-only inputs. */
    BnGPUResidentView x, z, out;
} BnGPUGatedRMSNormResidentPlan;

typedef enum {
    BN_GPU_RMSNORM_REDUCTION_DEFAULT = 0,
    /* Four contiguous F32 accumulators and a rounded mean multiplication.
     * Widths are multiples of four, at least 2048, and <= 130560.
     * The logical four-lane/two-axis reduction is retained across batches.
     * Wider rows require a multi-CTA reference reduction and are rejected. */
    BN_GPU_RMSNORM_REDUCTION_CONTIGUOUS_F32 = 1,
    /* Fixed 5120-column tiled reduction, shared gamma, offset=1, eps=1e-6.
     * Fused mean scaling/epsilon. Distinct persistent two-input and looped
     * three-input square pairing. Reject unsupported shapes; never fallback. */
    BN_GPU_RMSNORM_REDUCTION_TILED_5120_FMA = 2,
    /* Same fixed shape/gamma/epsilon contract; two-input residual RMS uses
     * 1024-column tiles and four F32 accumulators: square tile 1, FMA tile 0,
     * then FMA tiles 2, 3, 4. Plain
     * and three-input RMS retain TILED_5120_FMA arithmetic. No auto-selection
     * from current row count, and no fallback to another reduction. */
    BN_GPU_RMSNORM_REDUCTION_TILED_5120_LOOPED_ADD_FMA = 3,
    /* Fixed 5120/offset1/eps1e-6/shared-gamma contracts for all input counts.
     * Persistent: logical8192 columns, 512 threads; three-input squares are
     * rounded before addition, one/two-input preserve fused square pairs.
     * Looped: logical1024 columns, 256 threads; tile1 square then tile0 FMA,
     * followed by tile2/3/4 FMAs. Both fuse mean scaling with epsilon.
     * Explicit plan choices; no row-count selection or fallback. */
    BN_GPU_RMSNORM_REDUCTION_TILED_5120_PERSISTENT = 4,
    BN_GPU_RMSNORM_REDUCTION_TILED_5120_LOOPED = 5,
    /* Persistent 8192-column layout with fused square pairs for every input
     * count, including three-input RMS. Same strict fixed-shape contract. */
    BN_GPU_RMSNORM_REDUCTION_TILED_5120_PERSISTENT_FUSED_SQUARES = 6,
    /* Eight contiguous BF16 features per thread, increasing-distance warp
     * reduction and sequential warp aggregation with RN mean division. Width512/1536, eps1e-5,
     * shared gamma, offset0. Explicit MLA projection reference layout. */
    BN_GPU_RMSNORM_REDUCTION_VECTOR8_F32 = 7,
    /* Plain RMS only: original4096-column vector8/F32 reduction, eps1e-5,
     * offset0/shared gamma. Default reference launch uses512 threads for
     * fewer than256 rows,256 otherwise; preserve that arithmetic boundary. */
    BN_GPU_RMSNORM_REDUCTION_VECTOR8_4096 = 8,
} BnGPURMSNormReduction;

typedef struct {
    int n_tokens, dim, weight_offset;
    BnGPURMSNormReduction reduction;
    float eps;
    void *weight; /* Immutable BF16 gamma; offset is applied after conversion. */
    /* Fused residual sum in F32, RMSNorm, then separate BF16 outputs.
     * Exact extents, outputs disjoint from every other view. No host bridge. */
    BnGPUResidentView x, residual, out, residual_out;
    /* Zero/one shares gamma across rows. Otherwise gamma is [groups,dim],
     * row i selects i % groups, and n_tokens must be divisible by groups.
     * Grouped weights require CONTIGUOUS_F32 reduction. */
    int weight_groups;
} BnGPUAddRMSNormResidentPlan;

typedef struct {
    int n_tokens, dim, weight_offset;
    BnGPURMSNormReduction reduction;
    float eps;
    void *weight;
    /* ((F32(x) + F32(residual)) + F32(addend)), without an intermediate
     * BF16 store. Normalize this F32 sum and publish separate BF16 norm/sum
     * outputs. Exact BF16 extents; outputs disjoint from every other view.
     * Supports CONTIGUOUS_F32 reduction and the same grouped-gamma contract. */
    BnGPUResidentView x, residual, addend, out, residual_out;
    int weight_groups;
} BnGPUAdd3RMSNormResidentPlan;

typedef struct {
    int n_tokens, dim, weight_offset;
    BnGPURMSNormReduction reduction;
    float eps;
    void *weight; /* Immutable BF16 gamma, offset applied after conversion. */
    BnGPUResidentView x, out; /* Exact BF16 extents, disjoint. */
    /* Same grouped-gamma contract as BnGPUAddRMSNormResidentPlan. */
    int weight_groups;
} BnGPURMSNormResidentPlan;

/* Explicit vector4 FP16/F32 RMSNorm profile. Sequential four-component FMA
 * accumulation, XOR warp/block reduction, FTZ arithmetic and approximate
 * divide/rsqrt. FP16 inputs clamp to [-65504,65504]. Weighted output computes
 * (x * (gamma + constant_bias)) * (inverse_rms * constant_scale).
 * Immutable raw gamma is F16/BF16 [weight_groups,dim], or NULL unweighted
 * with weight_groups=1. Request views are exact [n_tokens,dim] F16/F32.
 * Partial overlap rejects; exact same-type input/output alias is allowed.
 * dim is a positive multiple of4. No residual or span-head mode. */
typedef struct {
    int n_tokens,dim,weight_groups;
    float eps,constant_bias,constant_scale;
    void *weight;
    BnGPUResidentView x,out;
} BnGPUVector4RMSNormResidentPlan;

/* Source F32 inverse-frequency table and exact sign-negated table. Uses
 * libdevice powf and source-ordered rounded F32 operations; no fused ramp
 * epilogue. YaRN scalar reciprocal is resolved in double BEFORE F32 cast.
 * Plain: factor1/start0/scale1. Dim even2..2048,1<base<=1e9,
 * 1<=factor<=65536,0<=ramp_start<=dim-1,0<ramp_scale<=1e9.
 * Two disjoint exact F32[dim/2] request-owned outputs. No allocation/copy/fence. */
typedef struct {
    int rotary_dim,yarn;
    float base,factor,ramp_start,ramp_scale;
    BnGPUResidentView out,negative;
} BnGPUF32RopeFrequencyResidentPlan;
static inline int bn_gpu_rope_frequency_shape(int dim,int yarn,float base,float factor,float start,float scale) {
    return dim<2||dim>2048||dim%2||(yarn!=0&&yarn!=1)||!(base>1.f&&base<=1e9f)||
        !(factor>=1.f&&factor<=65536.f)||!(start>=0.f&&start<=dim-1)||!(scale>0.f&&scale<=1e9f)||
        (!yarn&&(factor!=1.f||start!=0.f||scale!=1.f))?-1:0;
}

/* Source GPT-J rotation in-place on a strided F16 [tokens,heads,stride]
 * allocation. The operated head is [slice_offset,slice_offset+dim); other
 * channels remain untouched. Rotate rotary_dim channels at rotate_offset
 * within that head. Optional immutable F16/BF16 raw norm[1,dim] uses the
 * source pair reduction and rounding (not the vector4 RMSNorm profile).
 * positions is empty for pos0+row, or I32[tokens] absolute nonnegative
 * positions <=INT32_MAX. Metadata/frequencies remain immutable while queued.
 * inv_freq F32[rotary_dim/2] is explicit: no implicit theta/YaRN inference.
 * Exact request-owned views, no overlap with metadata. No alloc/copy/fence. */
typedef struct {
    int n_tokens,heads,dim,head_stride,slice_offset,rotary_dim,rotate_offset,pos0;
    float eps;
    void *norm;
    BnGPUResidentView x,inv_freq,positions;
} BnGPUGPTJF16ResidentPlan;
/* Low-rank F16 QKV projection profile with no Q-head normalization:
 * q_a -> vector4 RMSNorm -> q_b, x -> kv; then separate GPT-J Q slice
 * rotation and fused KV pair-RMSNorm/rotation. Five live activations are
 * disjoint; source projections/norm/rotations validate before any work.
 * q_rope uses a trailing rotary-only slice, kv_rope a full normalized head.
 * Both use identical explicit positions/frequencies. Reserve through the
 * existing linear_f16_reserve; no extra hidden workspace or KV append. */
typedef struct {
    BnGPUF16LinearResidentPlan q_a,q_b,kv;
    BnGPUVector4RMSNormResidentPlan q_norm;
    BnGPUGPTJF16ResidentPlan q_rope,kv_rope;
} BnGPUF16LatentQKVResidentPlan;



/* Unrotated compressed latents -> F16 key projection -> vector4 RMSNorm,
 * then GPT-J rotation of keys and latents at (entry0+row)*rate. Explicit
 * request-owned I32 positions are OUTPUT scratch generated on the device.
 * Both RoPE plans use this same view and frequency table; one head, trailing
 * rotary-only slice, no fused norm, pos0=0. Three F16 live arrays (latents,
 * projected keys, normalized/rotated keys) and metadata are all disjoint.
 * Norm has one immutable F16/BF16 group, bias0/scale1. Whole plan validates
 * before any writes. entry0>=0, rate>=1, all resulting positions<=INT32_MAX.
 * Reserve via linear_f16_reserve. No allocation/copy/fence or pool publication.
 * Device execution failure invalidates uncommitted outputs/latent state. */
typedef struct {
    int entry0,rate;
    BnGPUF16LinearResidentPlan project;
    BnGPUVector4RMSNormResidentPlan norm;
    BnGPUGPTJF16ResidentPlan keys_rope,latents_rope;
} BnGPUF16LatentKeysResidentPlan;

/* Publish rotated F16 latent/index rows into split paged pools. Pool layout
 * is [pages,entries_per_page,channels]; block_table is I32[pages]. Caller
 * supplies immutable valid physical page IDs in [0,pages), injective across
 * the logical pages written together. All six views are exact/request-owned
 * and disjoint. Input latents [rows,dim], keys [rows,index_dim], split pools
 * [pages*entries_per_page,dim-rotary_dim / rotary_dim / index_dim]. Only
 * entries [entry0,entry0+rows) change; all other entries remain untouched.
 * This operation copies bits; no quantization, allocation, transfer or fence.
 * Caller commits visibility only after completion. Invalid metadata contents
 * violate the precondition; view validation does not read back device IDs. */
typedef struct {
    int n_rows,entry0,entries_per_page,pages,dim,rotary_dim,index_dim;
    BnGPUResidentView latents,keys,block_table,pool_c,pool_r,pool_idx;
} BnGPUPagedLatentPublishResidentPlan;

/* Source sliding-window ring geometry: window plus two token pages of slack.
 * Update AFTER all attention consumers of the pre-update ring on this stream.
 * Source append/256-page shift/large-chunk rebase preserve unused/stale rows.
 * Scratch snapshots old rows before an overlapping move; all three exact F16
 * views are disjoint. No allocation, host transfer, synchronization or cursor
 * commit. Window begin/cursor ownership and rollback validity belong to caller. */
typedef enum { BN_GPU_SWA_APPEND=0,BN_GPU_SWA_SHIFT,BN_GPU_SWA_REBASE } BnGPUSWARingMode;
typedef struct {
    BnGPUSWARingMode mode;
    int next_beg,ring_src,old_rows,chunk_src,chunk_rows,chunk_dst;
} BnGPUSWARingUpdateInfo;
static inline int bn_gpu_swa_ring_rows(int window) {
    return window<1||window>65535?0:((window+512+255)/256)*256;
}
static inline int bn_gpu_swa_ring_update_info(int rows,int pos0,int beg,int window,
    int ring_rows,BnGPUSWARingUpdateInfo*out) {
    if(!out||rows<1||rows>65535||pos0<0||beg<0||beg%256||pos0<beg||
       pos0>INT32_MAX-rows||ring_rows!=bn_gpu_swa_ring_rows(window)||ring_rows<1||pos0-beg>ring_rows)return -1;
    int offset=pos0-beg,end=pos0+rows;
    BnGPUSWARingUpdateInfo v={BN_GPU_SWA_APPEND,beg,0,0,0,rows,offset};
    if(offset+rows>ring_rows) {
        if(rows<256) {
            int shift=((offset+rows-ring_rows+255)/256)*256;
            v.mode=BN_GPU_SWA_SHIFT;v.next_beg=beg+shift;v.ring_src=shift;v.old_rows=ring_rows-shift;v.chunk_dst=offset-shift;
        } else {
            int start=end-(window-1);if(start<0)start=0;
            v.mode=BN_GPU_SWA_REBASE;v.next_beg=start/256*256;
            int keep=end-v.next_beg;v.chunk_rows=keep<rows?keep:rows;
            v.old_rows=keep-v.chunk_rows;v.ring_src=pos0-v.old_rows-beg;
            v.chunk_src=rows-v.chunk_rows;v.chunk_dst=v.old_rows;
        }
    }
    if(v.next_beg<beg||v.ring_src<0||v.old_rows<0||v.ring_src>ring_rows-v.old_rows||
       v.chunk_dst<0||v.chunk_rows<1||v.chunk_dst>ring_rows-v.chunk_rows)return -1;
    *out=v;return 0;
}
typedef struct {
    int n_tokens,pos0,window_beg,window,ring_rows,dim;
    BnGPUResidentView kv,ring,scratch;
} BnGPUSWARingUpdateResidentPlan;

/* F16 latent K=V attention, H64/D512=(448 latent|64 RoPE), group-major
 * output [8,rows,4096]. Window rows are read from the PRE-update ring or the
 * current chunk, newest first, then logical pool entries in selection order.
 * Dense pools use floor((pos0+row+1)/rate); gathered indices are I32[rows,512],
 * -1 padded, otherwise valid/causal logical IDs supplied by the caller.
 * Table is immutable I32[pages], IDs in [0,pages); pools are F16[pages*epp,448/64].
 * pages=epp=entries=0 uses canonical empty pools/table. Window0 uses empty KV/ring
 * and ring_rows0; ring_rows0 is also allowed when win_floor==pos0 (no history).
 * Optional immutable raw F32[1,64] sink handle; NULL omits sinks. Optional F32[32]
 * inv_freq view is the NEGATIVE output de-rotation table. FP32 de-rotation
 * precedes final F16 store. Query and KV inputs are already rotated, finite F16.
 * Source auto split8/16 for rows<=8; one-shot otherwise. All activation views
 * exact, request-owned and disjoint. Empty unused views must be canonical.
 * No allocation, transfer, ring update, cursor commit or completion fence. */
typedef struct {int splits,head_tile;size_t ml_count,acc_count;} BnGPUF16LatentAttentionWorkspace;
static inline int bn_gpu_latent_attention_workspace(int rows,int entries,int rate,
    int dense,int window,int pos0,BnGPUF16LatentAttentionWorkspace*out) {
    if(!out||rows<1||rows>16384||entries<0||entries>1048576||rate<1||rate>256||
       (dense!=0&&dense!=1)||window<0||window>128||pos0<0||pos0>INT32_MAX-rows||
       (size_t)rows>SIZE_MAX/64/512/2)return -1;
    int splits=1;
    if(rows<=8){int pool=dense?(pos0+rows)/rate:512;if(pool>entries&&dense)pool=entries;
        splits=window+pool>256?16:8;}
    BnGPUF16LatentAttentionWorkspace w={splits,splits>1?16:32,0,0};
    if(splits>1){w.ml_count=(size_t)rows*64*splits*2;w.acc_count=(size_t)rows*64*splits*512;}
    *out=w;return 0;
}
typedef struct {
    int n_tokens,pool_entries,rate,dense_pool,window,pos0,ring_beg,win_floor,ring_rows,pages,entries_per_page;
    void *sinks;
    BnGPUResidentView queries,ring,kv,pool_c,pool_r,block_table,indices,inv_freq,out,scratch_ml,scratch_acc;
} BnGPUF16LatentAttentionResidentPlan;

/* F16 lightning-indexer query projection: normalized low-rank q -> queries,
 * x -> RAW signed head weights, then trailing-slice GPT-J rotation of queries.
 * No query head norm and no head-weight scaling here. Reserve existing F16
 * linear workspace. Four activation views plus frequency metadata disjoint;
 * sequential positions use query_rope.pos0, positions empty. Preflight all
 * stages before writing. No allocation, transfer or fence. */
typedef struct {
    BnGPUF16LinearResidentPlan query,weights;
    BnGPUGPTJF16ResidentPlan query_rope;
} BnGPUF16IndexQueryResidentPlan;
/* Source F16 signed index scores, H32/D128, scale1/64. R<=4 uses balanced
 * head reduction; R>4 ordered head FMAs. Tensor-core dot, ReLU per head, then
 * final F16 store; future key>=min((pos0+row+1)/rate,n_keys) becomes -inf.
 * scores F16[rows,stride], only columns<n_keys written. stride multiple128
 * and >=n_keys rounded up128. q F16[rows,32,128], w F16[rows,32].
 * Contiguous: pages=epp=0, keys F16[n_keys,128], block_table canonical empty.
 * Paged: pages>0,epp128/256, keys F16[pages*epp,128], I32 table[pages],
 * n_keys<=pages*epp; caller guarantees valid immutable physical page IDs.
 * Inputs finite and all views exact/disjoint/request-owned. No selection,
 * candidate masking, allocation, transfer, scratch or synchronization. */
typedef struct {
    int n_tokens,n_keys,pos0,rate,score_stride,pages,entries_per_page;
    BnGPUResidentView queries,weights,keys,block_table,scores;
} BnGPUF16IndexScoreResidentPlan;
typedef struct {
    BnGPUF16IndexQueryResidentPlan query;
    BnGPUF16IndexScoreResidentPlan score;
} BnGPUF16IndexQueryScoreResidentPlan;

/* Source-compatible FP16 index selection. Ordered FP16 bit keys greater
 * than key(-inf) participate (positive NaN payloads participate as in source).
 * Emit scores strictly above the cutoff in ascending index order, then enough
 * cutoff ties in ascending index order; remaining output slots become -1.
 * k=0 selects nothing; k may exceed the number of participating entries.
 * Workspace is request-owned, explicitly planned before execution; potential
 * split shapes require it even when alignment selects the scalar fallback.
 * No allocation, host transfer or completion fence inside the operation. */
typedef struct {
    int split_groups;
    size_t indices_count,scores_count,counts_count;
} BnGPUF16IndexTopKWorkspace;
static inline int bn_gpu_index_topk_workspace(int rows,int keys,int stride,int k,int pad,
                                             BnGPUF16IndexTopKWorkspace *out) {
    BnGPUF16IndexTopKWorkspace v={0};
    if(!out||rows<1||rows>65535||keys<0||keys>1048576||stride<1||stride<keys||
       k<0||pad<1||pad>1048576||k>pad||(k&&keys==0)||
       (size_t)rows>SIZE_MAX/(size_t)stride/2||(size_t)rows>SIZE_MAX/(size_t)pad/4)return -1;
    if(k&&rows<=16&&keys>=32768&&pad<=4096) {
        int groups=keys/(8*k);if(groups<2)groups=2;if(groups>32)groups=32;
        v.split_groups=groups;v.counts_count=(size_t)rows*groups;
        v.indices_count=v.counts_count*pad;v.scores_count=v.indices_count;
    }
    *out=v;return 0;
}
typedef struct {
    int n_tokens,n_keys,score_stride,k,k_pad;
    BnGPUResidentView scores,indices,scratch_indices,scratch_scores,scratch_counts;
} BnGPUF16IndexTopKResidentPlan;

/* Candidate-source block mask for the long-context indexer: block size8,
 * retain up to2048 blocks, always pin max((pos0+row+1)/rate-1,0)/8.
 * There are keys/8+1 blocks, including an empty tail when keys%8==0.
 * Input scores already contain causal -inf; finite values and infinities are
 * permitted, NaNs are outside this operation's input contract. Equal maxima
 * retain the later FP16 payload, preserving signed zeros. Output drop is I8
 * [rows,blocks], 1=dropped, 0=retained; unreachable -inf blocks stay dropped.
 * block_scores and block_indices are explicit scratch; block score padding
 * remains untouched. Selection scratch comes from the workspace planner.
 * All views exact, disjoint and request-owned. No allocation/copy/fence. */
typedef struct {
    int blocks;
    size_t block_scores_count,block_indices_count,drop_count;
    BnGPUF16IndexTopKWorkspace select;
} BnGPUIndexCandidateWorkspace;
static inline int bn_gpu_index_candidate_workspace(int rows,int keys,int stride,int block_stride,
                                                   int pos0,int rate,BnGPUIndexCandidateWorkspace*out) {
    BnGPUIndexCandidateWorkspace v={0};
    if(!out||rows<1||rows>65535||keys<=16384||keys>1048576||stride<keys||
       pos0<0||pos0>INT32_MAX-rows||rate<1||rate>256||
       (size_t)rows>SIZE_MAX/(size_t)stride/2)return -1;
    v.blocks=keys/8+1;
    int last=(pos0+rows)/rate-1;if(last<0)last=0;
    if(last/8>=v.blocks||bn_gpu_index_topk_workspace(rows,v.blocks,block_stride,2048,2048,&v.select))return -1;
    v.block_scores_count=(size_t)rows*block_stride;
    v.block_indices_count=(size_t)rows*2048;v.drop_count=(size_t)rows*v.blocks;
    *out=v;return 0;
}
typedef struct {
    int n_tokens,n_keys,score_stride,block_stride,pos0,rate;
    BnGPUResidentView scores,block_scores,block_indices,drop,scratch_indices,scratch_scores,scratch_counts;
} BnGPUIndexCandidateSourceResidentPlan;

/* Source score slabs are 256 rows with the final <=16-row tail absorbed. */
static inline int bn_gpu_index_slab_rows(int remaining) {
    return remaining<1||remaining>65535?0:remaining>272?256:remaining;
}
/* One candidate-use slab. drop is canonical I8[rows,keys/8+1], with <=2048
 * retained blocks per row. Union entries are emitted in ascending logical ID
 * order and gathered from contiguous/paged F16 keys. Capacity is bounded by
 * min(keys,rows*16384). Private GPU counts drive scoring/selection; inactive
 * tails remain untouched. Input queries already carry actual-position RoPE.
 * Scores use disabled causality, followed by drop and actual-position masks.
 * Output is I32[rows,512] logical entry IDs with -1 padding. All workspace is
 * exact, disjoint, request-owned and preallocated; no allocation/copy/fence.
 * Caller partitions a full chunk with bn_gpu_index_slab_rows(). */
typedef struct {
    int blocks,tiles,capacity;
    size_t gathered_count,scores_count,output_count,drop_count;
    BnGPUF16IndexTopKWorkspace select;
} BnGPUIndexCandidateUseWorkspace;
static inline int bn_gpu_index_candidate_use_workspace(int rows,int keys,int stride,int pos0,int rate,
                                                       BnGPUIndexCandidateUseWorkspace*out) {
    BnGPUIndexCandidateUseWorkspace v={0};
    if(!out||rows<1||rows>272||keys<=16384||keys>1048576||pos0<0||pos0>INT32_MAX-rows||rate<1||rate>256)return -1;
    v.blocks=keys/8+1;v.tiles=(v.blocks+255)/256;v.capacity=rows*16384;if(v.capacity>keys)v.capacity=keys;
    if(stride<((v.capacity+127)/128)*128||stride%128||
       bn_gpu_index_topk_workspace(rows,v.capacity,stride,512,512,&v.select))return -1;
    v.gathered_count=(size_t)v.capacity*128;v.scores_count=(size_t)rows*stride;
    v.output_count=(size_t)rows*512;v.drop_count=(size_t)rows*v.blocks;
    if(v.gathered_count>SIZE_MAX/2)return -1;
    *out=v;return 0;
}
typedef struct {
    int n_tokens,n_keys,score_stride,pos0,rate,pages,entries_per_page;
    BnGPUResidentView queries,weights,keys,block_table,drop;
    BnGPUResidentView entry_ids,entry_count,gathered_keys,scores,indices;
    BnGPUResidentView block_counts,block_offsets,tile_offsets;
    BnGPUResidentView scratch_indices,scratch_scores,scratch_counts;
} BnGPUIndexCandidateUseResidentPlan;

/* Stateful non-overlapping source compression. Input projections are F16
 * [tokens,dim]; persistent projected-row rings are F16[ring_rows,dim].
 * norm is explicit F16[dim] (source setup casts BF16 weights once), ape
 * F32[rate,dim], inv_freq F32[rotary_dim/2]. Pooling/norm/rotation stay F32
 * until the output half conversion. Do not substitute an F16 intermediate.
 * Caller initializes rings and provides the retained pre-pos0 history on
 * append/rewind. No position commit is implicit. Emitted rows are
 * floor((pos0+tokens)/rate)-floor(pos0/rate). Relative staging writes from
 * row0; otherwise writes at absolute compressed positions. Optional split
 * destinations partition channels. Relative staging requires split_dim=dim.
 * All views exact/disjoint; kv==gate is allowed only as an exact readonly
 * alias. Outputs beyond this call's emitted range remain unchanged. Device
 * execution failure invalidates uncommitted state; restore/reset before reuse.
 * No allocation, host copies or synchronization. No overlap/snapshots/pages. */
typedef struct {
    int n_tokens,pos0,rate,dim,rotary_dim,ring_rows,dest_rows,split_dim,relative;
    float eps;
    BnGPUResidentView kv,gate,ring_kv,ring_gate,ape,norm,inv_freq,dest_a,dest_b;
} BnGPUCompressF16ResidentPlan;
/* Cached F16 compressor projection + stateful compression. Rate1 is gateless:
 * gate plan is all-zero, compress.gate==compress.kv. Other rates use separate
 * same-input projections. Source BC rows<=32 use the existing16MiB handle;
 * longer/gateless rows use the32MiB torch-matmul handle. Reserve through
 * linear_f16_reserve. The whole plan validates before projecting or mutating
 * rings. Weights immutable/backend-owned, mutable state request-owned. */
typedef struct {
    BnGPUF16LinearResidentPlan kv,gate;
    BnGPUCompressF16ResidentPlan compress;
} BnGPUF16CompressorResidentPlan;

/* HC4 carry-pre profile: F32 streams [R,4,D], immutable raw F32 fn [24,4D],
 * base [1,24], scale [1,3]. R<=32 rounds fn to FP16 on read (F32 traffic).
 * Each site publishes its own pre weights; collapse consumes pre_in from the
 * previous site, or stream0 when pre_in is the all-zero empty view. Reset this
 * carry at the first site of each forward, not between sites of that forward.
 * Exact F32 request extents: partials queried below, post/pre_out [R,4],
 * comb [R,4,4], collapsed [R,D]. All supplied views pairwise disjoint.
 * R in [1,65535], D in [128,65536], D%4==0. CUDA SM80+ only.
 * Fixed reduction trees, FTZ/fast math; no allocation, copy or fence. */
static inline size_t bn_gpu_hc4_partials_count(int rows,int dim) {
    if(rows<1||rows>65535||dim<128||dim>65536||dim%4)return 0;
    int n=512/rows;if(n<1)n=1;if(n>128)n=128;
    int c=((4*dim/n+255)/256)*256;
    return (size_t)rows*((4*dim+c-1)/c)*25;
}
typedef struct {
    int n_tokens,dim,sinkhorn_iters; /* iterations in [1,256] */
    float rms_eps,hc_eps; /* finite, positive */
    void *fn,*base,*scale;
    BnGPUResidentView streams,partials,post,comb,collapsed,pre_in,pre_out;
} BnGPUHC4CarryResidentPlan;

/* In-place F32 streams [R,4,D] <- post*y + comb^T*streams. y [R,D] F16/F32;
 * post [R,4], comb [R,4,4] F32. Same geometry and arithmetic as carry above.
 * All views pairwise disjoint. Loads every input stream before storing a column.
 * This is a residual apply, not HyperHead's separate final carry collapse. */
typedef struct {
    int n_tokens,dim;
    BnGPUResidentView streams,y,post,comb;
} BnGPUHC4ApplyResidentPlan;

/* Ordinary text: gather immutable raw BF16 [vocab,dim], convert exactly to
 * F32 and broadcast into four streams. No scaling, normalization, indexed
 * embeddings or draft overrides. Exact I32 indices[R], F32 out[R,4,D],
 * disjoint; same geometry as HC4. Call embedding_bf16_reserve first (shares
 * its four-byte request error flag). Bad IDs never read outside the table;
 * the next explicit fence fails and poisons the request. Output is valid
 * only after a successful fence. No allocation/copy/fence in execution. */
typedef struct {
    int n_tokens,dim,vocab_size;
    void *weight;
    BnGPUResidentView indices,out;
} BnGPUBF16EmbeddingHC4ResidentPlan;

/* Final carried-pre HC4 collapse: FP32 batched matrix multiplication with
 * TF32 disabled. F32 streams[R,4,D],pre[R,4],out[R,D], exact extents and
 * pairwise disjoint, same geometry as HC4. Separate from the HCmix FMA loop:
 * vendor reduction choices can differ. Reserve the shared request FP32
 * cuBLAS workspace first; no allocation/copy/fence during execution. */
typedef struct {
    int n_tokens,dim;
    BnGPUResidentView streams,pre,out;
} BnGPUHC4HeadResidentPlan;

/* Deterministic sqrt(softplus) no-group routing. Immutable derived layout:
 * F16 gate[E,D], int8 hi/lo[E,D], F32 row scales, mean-centered F16 bias,
 * F32 mean and fixed validation status. Nonfinite gate/bias/mean or centered
 * bias that overflows F16 rejects before publication. Setup takes packed F16[K,N]
 * gate and raw F32[1,E] bias handles, never mutates either. E1..512,
 * D128..32768 in multiples128. Bytes include alignment/control padding. */
static inline size_t bn_gpu_router_weight_bytes(int experts,int dim) {
    if(experts<1||experts>512||dim<128||dim>32768||dim%128)return 0;
    size_t end=((size_t)experts*dim*4+(size_t)experts*6+3)&~(size_t)3;
    return (end+8+255)&~(size_t)255;
}
typedef struct {
    int experts,dim;
    void *gate,*bias;
    size_t device_budget_bytes;
} BnGPURouterWeightPlan;
typedef struct {
    int chunks,slices;
    size_t a_count,scale_count,part_count;
} BnGPURouterWorkspace;
/* Exact request-owned extents; single-row FMA path needs no quant workspace.
 * Multi-row path uses deterministic int8 tensor-core GEMM, never token loops
 * or vendor GEMM fallback. Slice count depends only on shape. */
static inline int bn_gpu_router_workspace(int rows,int dim,int experts,BnGPURouterWorkspace *out) {
    if(!out||!bn_gpu_router_weight_bytes(experts,dim)||rows<1||rows>65535||rows>INT32_MAX/dim)return -1;
    int kc=dim/128,tiles=((experts+63)/64)*((rows+127)/128),s=1;
    while(tiles*s<256&&s<8&&s*2<=kc)s*=2;
    int ks=((kc+s-1)/s)*128,se=(dim+ks-1)/ks;
    BnGPURouterWorkspace w={kc,se,rows==1?0:(size_t)rows*dim,
        rows==1?0:(size_t)rows*kc,rows==1||se==1?0:(size_t)se*rows*experts};
    *out=w;return 0;
}
typedef struct {
    int n_tokens,dim,experts,k; /* K1..32, K<=E */
    float scaling_factor; /* finite, positive */
    void *weight; /* dedicated immutable router handle; buffer_destroy owns it */
    BnGPUResidentView x,logits,selected,weights; /* F16[R,D], F16[R,E], I32[R,K], F16[R,K] */
    BnGPUResidentView ahi,alo,ascale,partials; /* I8[a_count] twice, F32[scale_count/part_count] */
    /* All present views pairwise disjoint; absent workspace is all-zero view.
     * Bias changes selection only, unbiased affinities normalize over selected
     * experts with epsilon1e-20, then scale and round F16. Original warp tie
     * order retained (not a global expert-index sort). No allocation/copy/fence. */
} BnGPURouterResidentPlan;




typedef struct {
    int rows, cols, divisor; /* divisor is 1/2/4/8/16. */
    /* BF16 division, then SiLU in F32 and a separate BF16 result.
     * Exact BF16 extents; output disjoint from input. */
    BnGPUResidentView x, out;
} BnGPUScaledSiluResidentPlan;

/* Qualified single-group F32 sigmoid+bias top-eight profile for E=288.
 * Bias affects selection; weights retain the original block/warp arithmetic
 * transition at 1024/1025 tokens, including large-batch biased-score recovery.
 * Stable lower-ID ties, nonfinite candidates excluded; all-invalid rows yield
 * the original uniform 1/8 fallback. Weights sum to route_scale for normal
 * inputs. Immutable F32 bias[1,E], exact disjoint request-owned views. */
typedef struct {
    int n_tokens,n_experts,k;
    float route_scale;
    void *bias;
    BnGPUResidentView logits,weights,indices;
} BnGPUBiasedSigmoidF32ResidentPlan;

typedef struct {
    int n_tokens, n_experts, k, renormalize;
    /* Initial CUDA profile: BF16 logits [tokens,512], k=10. F32 weights
     * and I32 IDs/source rows [tokens,k]. Rank probabilities with lower-ID
     * ties; nonfinite probabilities become zero. Source rows are rank*tokens
     * + token. All views are exact, disjoint, and owned by one request.
     * Preserves small/large batch reduction profiles; no allocation or fence. */
    BnGPUResidentView logits, weights, indices, source_rows;
} BnGPUMoETopKBF16ResidentPlan;

typedef struct {
    int n_tokens, n_experts, k, dim;
    /* E=1..512, k=1..min(E,32). Group slot=token*k+rank by expert.
     * I32 offsets[E+1], cursors[E], slots/inverse[tokens*k] and status[1];
     * BF16 x[tokens,dim], gathered[tokens*k,dim]. slots maps grouped row to
     * original slot, inverse maps original slot to grouped row. Row order
     * within an expert is unspecified. Every call resets counts and status.
     * All views exact, disjoint and request-owned. No allocation/copy/fence.
     * Invalid device IDs set status bit1 and leave slots/inverse/gathered
     * untouched; offsets describe only valid IDs. Inspect status before use
     * outside guarded resident operations. No duplicate-ID restriction. */
    BnGPUResidentView indices, x, offsets, cursors, slots, inverse, gathered, status;
} BnGPUMoEGroupBF16ResidentPlan;

typedef struct {
    int n_tokens, k, dim;
    /* k=1..32. grouped is BF16[tokens*k,dim], inverse I32[tokens*k], weights
     * F32[tokens,k]. weighted receives each product rounded to BF16 in the
     * ORIGINAL slot order, then out[tokens,dim] receives an ordered F32 sum
     * rounded to BF16. All views exact, disjoint and request-owned.
     * status is I32[1] from grouping; nonzero skips work. Out-of-range inverse
     * indices set bit2 and suppress both weighted and final output writes.
     * Does not reset status. No allocation, transfer or completion fence. */
    BnGPUResidentView grouped, inverse, weights, weighted, out, status;
} BnGPUMoEReduceBF16ResidentPlan;

typedef struct {
    int n_tokens, dim;
    /* BF16 sigmoid[tokens] = sigmoid(logits[tokens]); shared[tokens,dim] =
     * BF16(sigmoid * down), out = BF16(routed + shared). Each boundary rounds
     * independently, including subnormal sigmoid outputs. NaN payload is
     * unspecified. All views exact, disjoint, aligned and request-owned.
     * No allocation, transfer, norm/residual or completion fence. */
    BnGPUResidentView routed, down, logits, sigmoid, shared, out;
} BnGPUMoESharedBF16ResidentPlan;

typedef struct {
    int n_tokens, total_tokens, rows, cols, type;
    /* Borrowed raw single-expert projection. Weight lifetime extends through
     * stream completion. BF16 x/out have exact [n_tokens,cols/rows] extents.
     * Select vector arithmetic through 64 TOTAL tokens and matrix arithmetic
     * above 64, even for a small gathered expert batch. total_tokens is the
     * current MoE input batch/chunk, not KV position or whole prompt length.
     * No norm/residual,
     * allocation, transfer or fence. Reserve A8 scratch before submission. */
    void *weight;
    BnGPUResidentView x, out;
} BnGPUMoEProjectBF16ResidentPlan;

typedef struct {
    int rows, cols, type;
    size_t row_stride_bytes;
    /* Selected raw rows staged from a host table. Initial profile IQ4_NL:
     * packed is an exact I32 byte container with each row padded to16 bytes;
     * stride must equal round_up(serialized_row_bytes,16). out is exact
     * BF16[rows,cols], disjoint. No weight handles, allocation or readback.
     * F32 dequantization then one BF16 RNE, independent of A8 arithmetic. */
    BnGPUResidentView packed, out;
    /* Separate-scale row profile selected by quant registry: stride=cols/2,
     * packed exact I32[rows*cols/8], scales F16[rows*cols/32], out BF16.
     * All request-owned, disjoint. Source scale values must be finite and
     * nonnegative (validated before upload); this callback validates view
     * structure, not device tensor values. F32 product -> BF16 RNE retains
     * signed zero. Leave scales zeroed for the legacy embedded-scale profile. */
    BnGPUResidentView scales;
} BnGPUQuantRowsBF16ResidentPlan;

typedef struct { size_t packed_i32,scales_f16,out_bf16; } BnGPUScaledRowsWorkspace;
static inline int bn_gpu_scaled_rows_shape(int rows,int cols,BnGPUScaledRowsWorkspace *out) {
    if(!out||rows<1||rows>1048576||cols<32||cols>65536||cols%32||
        rows>INT32_MAX/cols||(size_t)rows>SIZE_MAX/(size_t)cols/2)return -1;
    size_t n=(size_t)rows*cols;
    BnGPUScaledRowsWorkspace w={n/8,n/32,n};*out=w;return 0;
}

/* Four-stream manifold HC, distinct from the sigmoid/mean HC operations.
 * SM120 profile: split-K TF32 tensor cores with RN input-weight conversion,
 * F32 coefficients/state and BF16 residuals. dim is a multiple of1024.
 * All views are exact, one request, pairwise disjoint. Weights are immutable
 * F32 fn[24,4*dim], scale[3], base[24], optional BF16 norm[dim]. Parameters
 * must be finite; epsilons positive, post multiplier nonnegative.
 * No allocation/copy/fence. Outputs are uncommitted after device failure. */
typedef struct {
    int n_tokens, dim, streams, n_splits, iterations;
    float rms_eps, pre_eps, sinkhorn_eps, post_mult, norm_eps;
    void *fn, *scale, *base, *norm;
    BnGPUResidentView residual;
    BnGPUResidentView dot, sqrsum; /* F32[split,tokens,24], [split,tokens]. */
    BnGPUResidentView pre, post, comb; /* F32[tokens,4/4/16]. */
    /* BF16[tokens,dim]. collapsed preserves the stream-sum rounding boundary.
     * With norm, variance uses its UNROUNDED F32 stream sum, whereas the norm
     * numerator uses rounded BF16. Without norm, out duplicates collapsed. */
    BnGPUResidentView collapsed, out;
} BnGPUMHCPreResidentPlan;
/* Default deferred post -> next pre profile on SM120, D4096/H4. For T<=32
 * project/SQ use UNROUNDED F32 mixed residuals; next.residual stores their
 * BF16 rounding for collapse and future residual use. For T>32, materialize
 * BF16 post first and use the normal TF32 pre profile. Explicit splits must
 * match the fused setup estimator for the small-batch regimes. All active
 * input/output views are one owner; every output is disjoint from all inputs
 * and other outputs. Caller preallocates all storage. No copies/alloc/fences. */
typedef struct {
    BnGPUMHCPreResidentPlan next;
    BnGPUResidentView previous_residual, previous_branch, previous_post, previous_comb;
} BnGPUMHCFusedResidentPlan;

typedef struct {
    int n_tokens, dim, streams;
    /* BF16 residual/out[tokens,4,dim], branch[tokens,dim]; coefficients F32.
     * Comb[source,destination]. Keep the fixed contracted accumulation order.
     * Outputs disjoint from every input; no request state is committed here. */
    BnGPUResidentView residual, branch, post, comb, out;
} BnGPUMHCPostResidentPlan;

/* Pure four-stream BF16 mean, D4096. F32 accumulation in stream order,
 * then a single mean scaling and BF16 rounding; no gate or norm. */
typedef struct {
    int n_tokens,dim,streams;
    BnGPUResidentView x,out;
} BnGPUStreamMeanResidentPlan;

typedef struct {
    int n_tokens, dim, streams; /* streams is 2/4/8/16. */
    /* [token,stream,dim] read-only normalized values and projected logits.
     * Sigmoid and product each round to BF16 before the F32 stream mean.
     * out is [token,dim], disjoint from both inputs. */
    BnGPUResidentView normalized, gate, out;
} BnGPUHyperMixResidentPlan;

typedef struct {
    int n_tokens, dim, streams; /* streams is 2/4/8/16. */
    /* residual/out [token,stream,dim], block_out [token,dim], projected
     * inject [token,stream]. Division, sigmoid, scale, product and residual
     * addition retain individual BF16 boundaries. Output is disjoint from
     * every read-only input; commit only after successful completion. */
    BnGPUResidentView residual, block_out, inject, out;
} BnGPUHyperCombineResidentPlan;

typedef struct {
    int n_tokens, dim, vocab_size;
    void *weight; /* Immutable raw quantized [vocab,dim], registry profile. */
    BnGPUResidentView indices; /* Exact I32[tokens]. */
    BnGPUResidentView out;     /* Exact BF16[tokens,dim]. No family scaling. */
    /* No full-vocabulary BF16 cache. Invalid device IDs never read weights
     * out of bounds; the next explicit fence reports failure and poisons
     * the request. Outputs are valid only after a successful fence. */
} BnGPUEmbeddingResidentPlan;

typedef enum {
    BN_GPU_ATTENTION_GATE_BF16 = 0,
    BN_GPU_ATTENTION_GATE_COMPILED_F32 = 1
} BnGPUAttentionGateProfile;

typedef struct {
    int n_tokens, n_heads, n_kv_heads, head_size;
    int position, cache_capacity, gated;
    float scale;
    BnGPUAttentionGateProfile gate_profile;
    /* Post-norm/post-RoPE BF16 Q/K and BF16 V, token-major. Append K/V at
     * position, then causal attention over [0,position+n_tokens).
     * Cache is request-owned BF16[capacity,kv_heads,head_size]. The caller
     * commits position only after a successful explicit request fence.
     * Initial qualified kernel profile: head_size=256, full causal attention. */
    BnGPUResidentView q, k, v, key_cache, value_cache, attention_out;
    /* Optional raw BF16 gate. Default rounds sigmoid to BF16 before multiply.
     * COMPILED_F32 retains F32 sigmoid (ex2.approx/div.full) through multiply;
     * it requires gated=1 and a backend capability bit for that contract.
     * gated=0 requires an empty gate view and out==attention_out. Otherwise
     * outputs and all input/cache views must be disjoint, exact extents. */
    BnGPUResidentView gate, out;
} BnGPUAttentionResidentPlan;

typedef enum {
    BN_GPU_ATTENTION_SPLIT_SINGLETON = 0,
    BN_GPU_ATTENTION_SPLIT_BATCH = 1
} BnGPUAttentionSplitPolicy;

/* One decode token per distinct live request. Cache and attention scratch stay
 * participant-owned; activation views may also belong to the coordinator.
 * Initial group profile: head256, at most64 query heads per KV head.
 * SINGLETON permits different head geometry, cache lengths and gate profiles
 * by sequence. See BATCH restrictions below. */
typedef struct {
    void *request;
    BnGPUAttentionResidentPlan plan;
} BnGPUAttentionDecodeSequence;
typedef struct {
    int n_sequences;
    const BnGPUAttentionDecodeSequence *sequences;
    /* Default preserves each singleton's partition/reduction choice. BATCH
     * uses cohort size and maximum KV length, matching batched FA2 selection.
     * BATCH requires equal head geometry and enough reserved split scratch
     * on every participant; cache lengths/capacities may still differ.
     * Invalid or insufficiently reserved cohorts queue no device work. */
    BnGPUAttentionSplitPolicy split_policy;
} BnGPUAttentionDecodeGroupPlan;
/* One or more causal tokens per distinct participant, including resumed
 * prefill chunks. Uses existing attention group workspace/BUSY semantics.
 * KV and FA2 scratch remain participant-owned; activation slices may belong
 * to the coordinator. Validate all sequences before queueing any append. */
typedef BnGPUAttentionDecodeSequence BnGPUAttentionPrefillSequence;
typedef struct {
    int n_sequences;
    const BnGPUAttentionPrefillSequence *sequences;
    /* SINGLETON preserves independent per-row FA2 mode selection. BATCH uses
     * the maximum query length of the entire cohort: if any query is longer
     * than one token, every row follows the causal, non-split prefill path.
     * An all-one-token cohort uses batch-wide decode partition selection.
     * Equal head geometry and reserved split scratch are required in BATCH. */
    BnGPUAttentionSplitPolicy split_policy;
} BnGPUAttentionPrefillGroupPlan;



/* Model-independent native BF16 cache preparation. F32 pow/reciprocal,
 * rounded F32 angles and precise sin/cos, then BF16 RN. Initial profile:
 * rotary_dim64, positions1..262144, packed rows [cos32|sin32]. */
typedef struct {
    int positions, rotary_dim;
    float theta;
} BnGPURotaryCachePlan;

typedef struct {
    int n_tokens, n_heads, n_kv_heads, head_size, rotary_dim, position;
    float eps;
    int weight_offset;
    /* Immutable BF16[head_size] norm weights and BF16[positions,rotary_dim]
     * cache, packed [cos|sin]. Caller supplies architecture-adapted weights.
     * Sequential text positions start at position. Initial profile: head256.
     * q_gate packs per-head [Q|gate]; k_in and v_in are separate projections.
     * This preserves mixed quant formats without a packing copy. All outputs
     * are separate token-major BF16 views; normalization rounds before RoPE. */
    void *q_weight, *k_weight, *cos_sin;
    BnGPUResidentView q_gate, k_in, v_in, q, k, v, gate;
    /* Optional input row strides in BF16 elements; zero selects contiguous
     * rows. Nonzero strides must cover the logical row and be multiples of 8.
     * Each input view spans (n_tokens-1)*stride+row_width elements, excluding
     * trailing padding after the final row. Read-only inputs may overlap,
     * allowing slices of a merged projection without packing copies.
     * Outputs remain contiguous and disjoint from every other view. */
    size_t q_gate_row_stride, k_in_row_stride, v_in_row_stride;
} BnGPUQKPrepResidentPlan;

/* Distinct request owners; each plan retains its own token count, position,
 * geometry and immutable normalization/rotary weights. Activations may belong
 * to that request or the explicit coordinator. No KV mutation here. */
typedef struct {
    void *request;
    BnGPUQKPrepResidentPlan plan;
} BnGPUQKPrepSequence;
typedef struct {
    int n_sequences;
    const BnGPUQKPrepSequence *sequences;
} BnGPUQKPrepGroupPlan;


/* Compiled GLM indexer LayerNorm, D128: BF16 input/output with immutable
 * full F32 gamma/bias[1,128]. Welford/FMA order preserves the original
 * single-row and batched reduction layouts, selected by n_tokens==1.
 * All finite neural inputs, exact disjoint request-owned extents. No
 * allocations, copies, fences or state commits on submission. */
typedef struct {
    int n_tokens;
    float eps;
    void *weight, *bias;
    BnGPUResidentView x, out;
} BnGPULayerNorm128F32BF16ResidentPlan;

/* GLM indexer finite BF16 rows: F32 Hadamard-128, BF16 materialization,
 * E4M3FN RNE saturating quantization with power-of-two F32 scales. out holds
 * raw FP8 bytes in I8 storage (not signed-integer arithmetic). Exact extents,
 * all disjoint, request-owned. No allocation, transfer, fence or commit. */
typedef struct {
    int rows;
    BnGPUResidentView x, out, scales; /* BF16[rows,128], I8[rows,128], F32[rows]. */
} BnGPUFWHT128FP8ResidentPlan;

/* Sequential text, four-token GLM gated key pooling, no speculative rows.
 * Source K and gate score are finite BF16[tokens,128]; APE is immutable
 * F32[4,128]. Tail is BF16[2,4,128]: normalized K then gate score, indexed by
 * logical position modulo4. Initial tail is read-only; next tail preserves
 * unwritten slots. position is caller-owned and no position commits occur.
 * Complete pools produce ((position+tokens)/4-position/4) FP8 rows with F32
 * power-of-two scales, in chronological order. Zero completed pools requires
 * empty output/scales views. The pooled result rounds to BF16, then follows
 * the same Hadamard/FP8 contract as queries. All views disjoint and exact.
 * Reserve outputs before submission; no allocation, transfer or fence here. */
typedef struct {
    int n_tokens, position;
    void *ape;
    BnGPUResidentView key, gate, initial_tail, next_tail, out, scales;
} BnGPUKPoolFP8ResidentPlan;

/* GLM query-scale/head-weight product, with two ordered F32 multiplies:
 * out = (weights * query_scales) * factor. Exact F32[rows] views, all disjoint.
 * factor is finite positive, supplied from anatomy head counts/dimensions. */
typedef struct {
    int rows;
    float factor;
    BnGPUResidentView weights, query_scales, out;
} BnGPUIndexerWeightScaleResidentPlan;

/* Sequential-text QSA pre-indexer: four Q heads, one K head, head128,
 * rotary64, compression ratio4, BF16 side cache, no speculative rows/MRoPE.
 * Gamma is BF16 Gemma weight (adds one in F32). Pooling rounds to BF16 before
 * K normalization; rotary products/FMA round in BF16. Positions are logical
 * sequential text positions. All mutable views must be disjoint.
 * Initial history is a raw-K ring [4,128], indexed by position modulo4.
 * Next history preserves unwritten slots and never mutates initial history.
 * Compressed output appends only newly completed groups, in order, count
 * ((position+n_tokens)/4-position/4)*128. A zero count requires an empty view.
 * No allocations, host transfers, fences or position commits on submission. */
typedef struct {
    int n_tokens, position;
    float eps;
    void *q_weight, *k_weight, *cos_sin; /* BF16[1,128], [1,128], [positions,64]. */
    BnGPUResidentView projected; /* BF16[tokens,640], merged Q512|rawK128. */
    BnGPUResidentView initial_history, next_history, q, compressed;
} BnGPUQSAIndexerResidentPlan;

/* Sequential text QSA score + deterministic packed selection. Four BF16
 * query heads/one compressed key head, D128, ratio4, token budget2048.
 * Scores are sum_h ReLU(dot(K,Q_h)) in F32, no scaling. Future columns are
 * -infinity. cache_groups is the allocated compressed-cache row capacity;
 * it must cover floor((position+n_tokens)/4), and lies in [1,65536]. */
typedef struct {
    int n_tokens, position, cache_groups;
    BnGPUResidentView q, compressed, scores; /* BF16[tokens,4,128], BF16[groups,128], F32[tokens,groups]. */
} BnGPUQSAScoreResidentPlan;
/* Valid visible scores are finite/nonnegative, as produced by the score op.
 * Selects up to512 groups by score; equal scores prefer lower group index.
 * Emits selected groups in chronological order, four raw token indices each,
 * then the current partial group; unused entries=-1. Last column is COUNT,
 * never a token index. This deterministic order/tie rule is explicit: pinned
 * vLLM's atomic top-k output order and tied membership can vary across runs.
 * Both operators allocate/transfer/fence/commit nothing; all views disjoint. */
typedef struct {
    int n_tokens, position, cache_groups;
    BnGPUResidentView scores, out; /* F32[tokens,groups], I32[tokens,2052]. */
} BnGPUQSASelectResidentPlan;

/* GLM FP8 indexer side cache, SM120 F32-scale profile: physical pages
 * contain 64*128 FP8 bytes followed by 64 raw F32 scales (8448 bytes/page).
 * Raw bytes use I8 storage; cache/value/output byte bases are four-byte aligned.
 * Append accepts n_rows>=0, exact disjoint views,
 * and I32 physical slots. Negative/out-of-range slots are padding: no write.
 * Valid slots must be distinct within a submission (scheduler invariant).
 * Physical block/slot zero is valid. No allocations/transfers/fences/commits. */
typedef struct {
    int n_rows,num_cache_blocks;
    BnGPUResidentView values,scales,slots,cache;
} BnGPUIndexerFP8AppendResidentPlan;
/* Gather logical rows [0,n_rows) using I32 block_table[logical_blocks].
 * Exact disjoint request-owned views; table covers ceil(n_rows/64).
 * Invalid physical pages zero their output rows and atomically set status=1.
 * Status is monotonic: caller initializes it to zero before the operation
 * chain and checks it before accepting outputs/state. Cache is read-only.
 * n_rows=0 is a no-op with empty output/scales. F32 scale bits are copied
 * without arithmetic; this does not qualify MX packed-scale cache mode. */
typedef struct {
    int n_rows,num_cache_blocks,logical_blocks;
    BnGPUResidentView cache,block_table,out,scales,status;
} BnGPUIndexerFP8GatherResidentPlan;
/* GLM SM120 F32-scale FP8 scoring for one contiguous request chunk.
 * n_keys must equal floor((position+n_tokens)/4). For row i, only keys below
 * floor((position+i+1)/4) are visible; future columns are written -INFINITY.
 * q=I8[tokens,32,128], keys=I8[n_keys,128], scales=F32[n_keys],
 * weights=F32[tokens,32] (signed, with query scales already folded),
 * scores=F32[tokens,n_keys]. F32 sum_h ReLU(dot(K,Q_h))*weight_h*key_scale
 * preserves the original SM120 MMA/head reduction order. Exact disjoint
 * request-owned views, byte inputs four-byte aligned. n_keys=0 accepts empty
 * keys/scales/scores. No allocation, transfer, fence or position commit.
 * This contiguous gathered-history prefill primitive is not paged decode. */
typedef struct {
    int n_tokens,position,n_keys;
    BnGPUResidentView q,keys,scales,weights,scores;
} BnGPUIndexerFP8ScoreResidentPlan;
/* Same signed SM120 scoring/causality as the contiguous scorer, reading the
 * planar64-row FP8/F32 cache directly through block_table. Single sequential
 * request/chunk; no gathered history or per-query request-index indirection.
 * cache I8[blocks*8448], table I32[logical_blocks] covering ceil(n_keys/64),
 * q I8[tokens,32,128], weights F32[tokens,32], scores F32[tokens,n_keys],
 * status I32[1]. n_keys=floor((position+tokens)/4). Exact disjoint views;
 * byte bases four-byte aligned. A visible invalid physical page writes -inf
 * for that page and sets sticky status=1; future invalid pages are masked.
 * Caller initializes/checks status after fencing before publishing state.
 * Empty scores when no keys. No allocation, gather, transfer, fence or commit. */
typedef struct {
    int n_tokens,position,n_keys,num_cache_blocks,logical_blocks;
    BnGPUResidentView q,weights,cache,block_table,scores,status;
} BnGPUIndexerFP8PagedScoreResidentPlan;

/* GLM signed-score top512 pools and four-token expansion, no count column.
 * n_keys=floor((position+n_tokens)/4). Exact disjoint request views:
 * scores F32[tokens,n_keys], pools I32[tokens,512], out I32[tokens,2051],
 * status I32[1]. Visible nonfinite scores set sticky status=1 and leave that
 * row's pools/out all -1; future columns are ignored. Caller initializes and
 * checks status after its fence before publishing output. Deterministic native
 * order is chronological; equal cutoff scores prefer the lower pool index
 * (signed zeros tie). Original atomic top-k order/tied membership may vary;
 * this explicit policy still needs downstream MLA/generation qualification.
 * Normal mode expands pools into columns0..2047, then up to3 incomplete-tail
 * tokens at2048..2050. Padding=-1. short_prefill=1 requires end<=2048 and
 * empty scores; it emits contiguous causal tokens as the original short-path
 * bypass does. Pools remain diagnostic chronological complete-pool indices.
 * No allocation, transfer, fence or position commit; SM120 profile only. */
typedef struct {
    int n_tokens,position,n_keys,short_prefill;
    BnGPUResidentView scores,pools,out,status;
} BnGPUIndexerSelectResidentPlan;



/* NoPE MLA fp8_ds_mla cache: 64 rows/page, 656 bytes/row: 512 E4M3,
 * four F32 power-of-two scales for contiguous128 groups, then128 zero bytes.
 * Scale=max(group_absmax/448,1e-4), rounded up to a power of two. No Hadamard.
 * kv BF16[tokens,512], slots I32[tokens], cache I8[blocks*64*656]. Exact
 * disjoint request-owned views, BF16/cache bases16B aligned. Negative and
 * out-of-range slots are padding/no-write; valid slots unique per submission
 * (scheduler invariant). n_tokens=0 accepts empty input/slots. No allocation,
 * transfer, fence or position commit. SM120 NoPE profile, not MX scale format. */
typedef struct {
    int n_tokens,num_cache_blocks;
    BnGPUResidentView kv,slots,cache;
} BnGPUMlaFP8CacheAppendResidentPlan;

/* SM120 GLM NoPE sparse MLA, H64/latent512, scale1/sqrt256.
 * query/out BF16[tokens,64,512], cache I8[blocks,64,656]. Selection
 * I32[tokens,2051] contains logical token indices (-1 padding). block_table
 * I32[logical_blocks] maps logical64-token pages to physical cache pages.
 * physical I32[tokens,2176] is writable scratch; the final125 columns=-1.
 * Invalid non-padding, future, or unmapped indices become -1 and OR1 into
 * status I32[1]. Status is sticky, caller initializes/checks before commit.
 * lse F32[tokens,64]. For tokens<=64, reserve partial BF16[tokens,64,34,512]
 * and partial_lse F32[tokens,64,34]. For larger batches both views are empty:
 * batched prefill writes directly. Exact disjoint views; query/cache/physical/
 * out/lse/partial buffers16B aligned. position+tokens fits logical capacity.
 * All index conversion and attention run on the request stream, without
 * allocation, transfer, synchronization, fallback, or position commit.
 * Execution status/caller cancellation requires discarding partial results. */
typedef struct {
    int n_tokens,position,num_cache_blocks,logical_blocks;
    BnGPUResidentView query,cache,selection,block_table,physical,out,lse;
    BnGPUResidentView partial,partial_lse,status;
} BnGPUMlaFP8AttentionResidentPlan;


/* Scatter token-major BF16 K/V rows into request-owned paged caches.
 * row_width is the flattened KV-head width, a multiple of 8 in [8,65536].
 * page_size lies in [1,65536]; total physical slots must fit positive I32.
 * Negative/out-of-range slots are padding and leave the cache untouched.
 * Valid slots MUST be distinct within one submission (scheduler invariant).
 * Slots may be reused in later, stream-ordered submissions. Physical slot0
 * is valid. Views have exact extents; BF16 bases are 16-byte aligned and I32
 * metadata naturally aligned. Readonly K/V may alias; writable caches must
 * be disjoint from each other and every input. Submission performs no
 * allocation, host transfer, synchronization or request-position commit. */
typedef struct {
    int n_tokens, num_cache_blocks, page_size, row_width;
    BnGPUResidentView k, v, slots; /* BF16[nt,width], I32[nt]. */
    BnGPUResidentView k_cache, v_cache; /* BF16[blocks,page_size,width]. */
} BnGPUPagedKVAppendResidentPlan;

/* Paged BF16 GQA: 24 Q heads, 2 KV heads, D256, pages of 16 tokens.
 * Packed selections have 2052 columns, including the trailing count. Invalid
 * GPU request/page/index entries are masked as padding. Input extents are exact;
 * all views belong to the same resident request. BF16/F32 views require 16-byte
 * alignment; I32 metadata requires natural alignment. Readonly inputs may alias;
 * out/workspace must be disjoint from every other view.
 * Token counts/capacities lie in [1,131070]. Failed size queries leave their
 * output unchanged. Query workspace_size(max_tokens) before allocating F32 scratch:
 * it bounds all token counts <= max_tokens and both prefill configurations.
 * The workspace view itself is the bound; submission never grows it. It may
 * be larger than the current operation needs. Single-split calls may supply
 * either that reusable workspace or a zero-initialized empty view.
 * Prepare the kernels once, with serialized backend access and outside capture.
 * Preparation configures device functions, not tensor storage; queries work before
 * preparation. Submission rejects an unprepared backend.
 * Submission allocates/transfers/fences/commits nothing. Cache data is readonly;
 * request state/position commits remain the caller's responsibility. */
typedef struct {
    int n_tokens, use_prefill_config, num_cache_blocks, num_requests, table_width;
    BnGPUResidentView q, k_cache, v_cache; /* BF16[nt,24,256], [blocks,16,2,256]. */
    BnGPUResidentView indices, block_table, token_to_req; /* I32[nt,2052], [requests,width], [nt]. */
    BnGPUResidentView gate, out; /* BF16[nt,24,256]. */
    BnGPUResidentView workspace; /* F32, optional only for single-split. */
} BnGPUQSAAttentionResidentPlan;

typedef struct {
    int n_tokens, outer, inner, width;
    /* BF16 [tokens,outer,inner,width] -> [tokens,inner,outer,width]. */
    BnGPUResidentView x, out;
} BnGPUTransposeAxesResidentPlan;

typedef struct {
    int rows, cols, left_cols;
    /* Split each F32/BF16 row into contiguous left/right matrices, preserving
     * every bit. Exact extents, one request/type, all outputs disjoint. */
    BnGPUResidentView x, left, right;
} BnGPUSplitColumnsResidentPlan;

/* Bit-preserving F32/BF16 row selection, including repeated indices.
 * x[source_rows,row_width] -> out[n_rows,row_width], one resident owner/type,
 * disjoint exact extents. rows is a host array of n_rows indices, borrowed
 * only until the callback returns. Every index is validated before queueing. */
typedef struct {
    int source_rows, row_width, n_rows;
    const int *rows;
    BnGPUResidentView x, out;
} BnGPUGatherRowsResidentPlan;


typedef struct {
    int rows, cols, left_cols;
    /* Join contiguous F32/BF16 left/right rows into x, preserving every bit.
     * Exact extents and one request/type. x must not alias either input. */
    BnGPUResidentView left, right, x;
} BnGPUJoinColumnsResidentPlan;

typedef struct {
    int n_tokens, rows, cols;
    BnGPUBF16LinearWeight weight;
    /* BF16[token,cols] -> BF16[token,rows], registry-selected arithmetic.
     * Exact extents and disjoint input/output. Decoded weights are immutable
     * backend-owned BF16 matrices; raw weights remain required for A8.
     * The qualified CUDA decoded-GEMM profile requires 16-byte input/output
     * starts; unaligned views are rejected before queueing work. */
    BnGPUResidentView x, out;
} BnGPUBF16LinearResidentPlan;

typedef struct {
    size_t count;
    /* BF16 addition with exact extents and one owner. Output is disjoint
     * from both read-only inputs; inputs may alias each other. */
    BnGPUResidentView x,y,out;
} BnGPUAddBF16ResidentPlan;

/* One rounded F32 addition, including signed-zero and subnormal behavior.
 * Exact extents and one request owner. Read-only inputs may overlap; output
 * must be disjoint from both. No casts, workspace, allocation or host fence. */
typedef struct {
    size_t count;
    BnGPUResidentView x, y, out;
} BnGPUAddF32ResidentPlan;

/* Explicit raw F32 weight projection with BF16 activations and F32 output.
 * Caller-owned cast scratch preserves exact BF16->F32 conversion. Reserve
 * the existing resident BLAS handle/workspace before use (BF16 reserve with
 * max_elements=0 suffices). Pedantic F32 compute; no TF32/weight rounding.
 * Exact disjoint views, same request; queues cast plus GEMM without allocation,
 * transfers or fence. Weight is immutable F32[rows,cols] backend storage. */
typedef struct {
    int n_tokens,rows,cols;
    void *weight;
    BnGPUResidentView x,cast,out;
} BnGPUBF16F32LinearResidentPlan;

/* Compatibility entry point restricted to A8 MMVQ/MMQ; rejects decoded GEMM. */
typedef BnGPUBF16LinearResidentPlan BnGPUA8LinearResidentPlan;
/* Per-head BF16 linear projection: weight[heads,rows,cols], x[tokens,heads,cols]
 * -> out[tokens,heads,rows]. F32 accumulation, BF16 output; no bias. Immutable
 * decoded BF16 weight buffer has flattened rows=heads*rows. Exact disjoint
 * request views, 16-byte aligned. Caller reserves BF16 linear handle/workspace
 * before submission. No allocation, copy, fence or position commit. */
typedef struct {
    int n_tokens,heads,rows,cols;
    void *weight;
    BnGPUResidentView x,out;
} BnGPUHeadLinearBF16ResidentPlan;


/* Original clamped SwiGLU default alpha=1,beta=0 contract. Gate has only an
 * upper clamp; up has both clamps. Clamp bounds/intermediates stay F32;
 * SiLU rounds to BF16 before multiplying by (clamped_up + positive zero),
 * then the product rounds to BF16. Exact disjoint BF16 views, one request.
 * No allocation/copy/fence. limit is finite and nonnegative. */
typedef struct {
    int n_tokens, hidden_dim;
    float limit;
    BnGPUResidentView gate_up, out; /* [tokens,2*hidden], [tokens,hidden]. */
} BnGPUClampedSwiGLUResidentPlan;

typedef enum {
    BN_GPU_BF16_FFN_SILU_SEPARATE = 0, /* Torch BF16 SiLU, then BF16 multiply. */
    BN_GPU_BF16_FFN_SILU_FUSED = 1,    /* F32 SiLU/product with RN division, BF16 result. */
    /* Compiled F32 SiLU/product: libdevice exp, PTX div.full.f32, BF16 result.
     * Explicit arithmetic; does not change SEPARATE/FUSED or select by shape. */
    BN_GPU_BF16_FFN_SILU_COMPILED = 2
} BnGPUBF16FFNActivationProfile;

typedef enum {
    BN_GPU_BF16_FFN_SEPARATE = 0,
    BN_GPU_BF16_FFN_STACKED_DECODED = 1,
} BnGPUBF16FFNLayout;

typedef struct {
    int n_tokens, dim, hidden_dim;
    BnGPUBF16FFNWeights weights;
    /* Preserve registered linear profiles. The default activation rounds SiLU
     * and product separately; FUSED keeps their intermediate arithmetic F32.
     * All views are exact BF16 extents, pairwise disjoint, 16-byte aligned.
     * gate/up are caller-reserved request scratch, overwritten by execution.
     * Reserve linear_bf16 workspace for nt*max(dim,hidden_dim) beforehand. */
    BnGPUResidentView x, gate, up, out;
    BnGPUBF16FFNActivationProfile activation_profile;
    /* Optional canonical merged gate/up GEMM. Requires COMPILED activation
     * and decoded gate/up views of gate_up. gate_up_out is [nt,2*hidden],
     * shares gate's exact starting address, and is disjoint from x/up/out.
     * up becomes the product scratch in this layout; scratch contents are
     * otherwise unspecified. SEPARATE requires these optional fields empty. */
    BnGPUBF16FFNLayout layout;
    void *gate_up;
    BnGPUResidentView gate_up_out;
} BnGPUFFNResidentPlan;

typedef struct {
    int total_tokens;
    /* Raw expert weights only, using gathered MoE precision profiles. Gate is
     * overwritten with BF16 activation; up remains the projected values.
     * Both SiLU and its product round to BF16. No norm/residual or routing
     * weighting. All weights/views/scratch are checked before queuing work. */
    BnGPUFFNResidentPlan ffn;
} BnGPUMoEFFNBF16ResidentPlan;

typedef struct {
    unsigned slots, experts_per_slot; /* 1..4 slots, 1..64 experts each. */
    int dim, hidden_dim; /* dim multiple256, hidden_dim multiple32. */
    /* Fixed raw payload budgets, separately for pinned RAM and GPU memory.
     * Allocator/event/host descriptor overhead is excluded. Reserve sizes for
     * the largest supported pair: Q5_K gate/up and Q8_0 down. */
    size_t host_budget_bytes, device_budget_bytes;
    /* Zero retains 512 experts/top-10 compatibility. Otherwise E=1..512,
     * k=1..min(E,32). */
    int n_experts, k;
} BnGPUMoEBankConfig;

typedef struct {
    void *request, *data;
    uint64_t generation;
    uint32_t slot, capacity;
    int gate_type, down_type;
    /* Expert i starts at data+i*expert_bytes: gate, up, then down. Only
     * entries actually submitted need initialization. Bytes expire on submit
     * or cancel; never write until successfully reacquiring this slot. */
    size_t gate_bytes, down_bytes, expert_bytes;
} BnGPUMoEBankLease;

typedef struct {
    int expert, first_row, n_rows;
} BnGPUMoEBankItem;

typedef struct {
    BnGPUMoEBankLease lease;
    int total_tokens, n_experts;
    /* Host descriptors from successful GPU grouping metadata readback.
     * Experts/ranges must be unique/disjoint within a bank. Caller checks
     * grouping status and schedules every active expert exactly once.
     * Borrowed only during submission; no host activation bridge. */
    const BnGPUMoEBankItem *experts;
    /* Full BF16 [total_tokens*k,dim/hidden_dim] views of grouped input,
     * gate/up scratch and output. Exact, disjoint, same request. */
    BnGPUResidentView x, gate, up, out;
    /* Zero selects per-expert BF16 input quantization. Otherwise use this
     * request's most recently prepared original-input generation. x remains
     * required for view/range validation but is not read for gate/up compute. */
    uint64_t prepared_input;
    /* One gate/up, activation, down-quantization and down launch per bank.
     * Requires prepared_input and explicit grouped workspace setup. */
    int grouped;
    /* Per-layer activation contract. Zero retains ordinary BF16 SiLU;
     * clamped=1 uses asymmetric F32 clamps with the separate BF16 SiLU
     * boundary. Limit must be finite and nonnegative. Pool storage is shared
     * across layers; activation settings are validated before submission. */
    int clamped;
    float clamp_limit;
} BnGPUMoEBankResidentPlan;

/* One consumer of a shared raw bank. Each consumer keeps its own prepared
 * input generation, grouped views, row geometry and activation contract.
 * experts contains one entry per source expert, in the SAME order for every
 * consumer; n_rows=0 marks an expert unused by this consumer. Every consumer
 * must use at least one expert, and every source expert must have a consumer.
 * workspace_slot borrows existing grouped-bank metadata, without acquiring
 * that consumer's raw host/device bank. Requests must be distinct. */
typedef struct {
    void *request;
    unsigned workspace_slot;
    int total_tokens;
    const BnGPUMoEBankItem *experts;
    BnGPUResidentView x, gate, up, out;
    uint64_t prepared_input;
    int clamped;
    float clamp_limit;
} BnGPUMoEBankConsumerPlan;
typedef struct {
    BnGPUMoEBankLease lease; /* source owner passed separately to submission */
    int n_experts, n_consumers; /* 1..lease.capacity, 1..16 */
    const BnGPUMoEBankConsumerPlan *consumers;
    /* Zero for unbound banks; otherwise the source owner's current cohort
     * cache ticket. A private cache lease cannot be used for joint banks. */
    uint64_t cache_generation;
} BnGPUMoEBankGroupPlan;

typedef struct {
    int n_tokens;
    /* Original BF16[tokens,dim], grouped I32 slots[tokens*k], I32 status[1]
     * from successful grouping. Quantize original rows once and map complete
     * A8 blocks into grouped order. Nonzero status suppresses the map; caller
     * must check grouping status before submitting banks. No allocation/fence.
     * Repreparing is stream-ordered after previously submitted banks and
     * invalidates older host generation tickets for future submissions. */
    BnGPUResidentView x, slots, status;
} BnGPUMoEBankInputPlan;

enum { BN_GPU_MOE_BANK_BUSY = 1 };

typedef struct {
    size_t reserved_host_bytes, reserved_device_bytes;
    size_t upload_bytes, submitted_banks, submitted_experts, busy_acquires;
    size_t reserved_input_bytes, input_preparations;
    size_t reserved_group_host_bytes, reserved_group_device_bytes;
    size_t metadata_upload_bytes, grouped_banks;
    /* Source-owner counters. The upload is charged once; avoided bytes count
     * repeated active experts, excluding zero-row consumer descriptors. */
    size_t shared_uploads, shared_consumers, shared_upload_bytes_avoided;
    /* Actual joint-dispatch counts, relative to four private launches per
     * active consumer. Projection buckets retain private MMVQ/MMQ/MMA math. */
    size_t joint_kernel_launches, joint_projection_launches, joint_launches_avoided;
} BnGPUMoEBankStats;

/* Model-owned raw expert residency. Config arrays are borrowed only during
 * creation. Pools use exact format payload classes, never decoded weights.
 * Caller serializes cache operations; model/cache outlive borrowing requests.
 * CUDA accepts 1..512 experts and bound banks with top-k 1..32. Grouped
 * participants must have the same expert count/top-k. The backend model must
 * separately qualify its cache binding, layer mapping and payload budgets. */
typedef struct {
    int n_layers, n_experts, dim, hidden_dim;
    const int *gate_types, *down_types;
    size_t device_budget_bytes;
} BnGPUExpertCacheConfig;
/* Owned cache payload only; driver/allocator overhead is separate. */
typedef struct {size_t host_bytes,device_bytes;} BnGPUExpertCacheMemory;
typedef struct {
    void *request;
    uint64_t generation, hits[8];
    int layer;
} BnGPUExpertCacheLayer;
/* Whole-layer routing supplied before any bank upload. Distinct bound
 * requests share one model cache; owner must be one of these rows. Offsets
 * are immutable host readbacks of each request's own GPU grouping. */
typedef struct {
    void *request;
    const int *offsets;
    int n_tokens, prefill;
} BnGPUExpertCacheGroupRow;
typedef struct {
    /* Reservation/residency describe the shared model pool. Other counters
     * describe this request's selected experts and actual fill/eviction work.
     * A cohort lease charges unique union selections to its source owner.
     * bypasses counts decode admission pressure, not intentional prefill
     * borrowing. fill_copy_bytes is bank-to-cache D2D payload. */
    size_t reserved_device_bytes, resident_bytes, resident_experts;
    size_t hits, misses, hit_bytes, miss_bytes, evictions, bypasses;
    size_t fill_copy_bytes;
} BnGPUExpertCacheStats;
/* Setup only: source fills one exact [gate|up|down] raw payload synchronously.
 * Its destination is borrowed until return; callbacks must not retain it or
 * reenter cache operations. No expert becomes visible before complete success.
 * Requires an empty full-capacity cache with no bound requests. On source
 * failure the cache stays empty and retryable; CUDA failures may poison it.
 * Caller serializes setup and all source I/O. Output is unchanged on failure. */
typedef int (*BnGPUExpertCacheRead)(void *source, int layer, int expert,
                                   void *destination, size_t bytes);
typedef struct {
    size_t host_staging_bytes, upload_bytes, resident_bytes, resident_experts;
} BnGPUExpertCachePreloadInfo;

typedef struct {
    void *gate_buf;
    void *up_buf;
    void *down_buf;
    int use_gateup_split;
} BnGPUMoEPrefillExpert;

// Geometry of a gathered expert batch. Input rows follow original token order,
// with at most one row per token. The runtime owns grouping; backends use the
// full geometry to preserve routed projection arithmetic.
typedef struct {
    int total_tokens;
    int n_experts;
    int expert_index;
    // Source gate/up share one logical projection (0 = independent, 1 = merged).
    // This does not require the backend buffers to share storage.
    int gate_up_fused;
} BnGPUMoEExpertBatchPlan;

// Complete dense-residual MoE composition after routed experts have been
// reduced on the host. Borrowed buffers remain valid for this call only.
typedef struct {
    float *out;
    const float *act;
    const float *routed;
    void *gate_buf;
    void *up_buf;
    void *down_buf;
    void *input_norm_buf;
    void *dense_post_norm_buf;
    void *routed_post_norm_buf;
    void *output_norm_buf;
    int n_tokens;
    int dim;
    int hidden_dim;
    int gate_type;
    int up_type;
    int down_type;
    int act_type;
    float norm_eps;
    int raw_output;
} BnGPUMoEDenseResidualBatch;

// Compose an ordered routed MoE projection and its dense residual branch
// while the routed output is still device-resident. All pointers are borrowed.
typedef struct {
    BnGPUMoEDenseResidualBatch dense;
    const float *input;
    const int *indices;
    const float *weights;
    const float *output_scales;
    void *routed_gate_buf;
    void *routed_up_buf;
    void *routed_down_buf;
    int moe_hidden_dim;
    int n_experts;
    int k;
    int routed_gate_type;
    int routed_up_type;
    int routed_down_type;
} BnGPUMoERoutedDenseResidualBatch;

typedef enum {
    BN_GPU_ROPE_FACTOR_NONE = 0,
    BN_GPU_ROPE_FACTOR_MULTIPLY,
    BN_GPU_ROPE_FACTOR_DIVIDE,
} BnGPURopeFactorMode;

// Optional construction recipe for a range of the frequency table. Borrowed
// pointers are valid during init_activations only; backends own prepared data.
typedef struct {
    int offset;
    int pair_count;
    int rotary_dims;
    float theta;
    const float *factors;
    BnGPURopeFactorMode factor_mode;
} BnGPURopeFrequencyPlan;

// Backend-neutral description of request-local activation resources. Model and
// runtime planning populate this value; GPU implementations must not inspect
// BnConfig or other model-owned structures.
typedef struct {
    int dim;
    int n_layers;
    int seq_len;
    int kv_dim;
    int n_heads;
    int head_size;
    int vocab_size;
    int per_layer_input_dim;
    int kv_f16;
    int attention_layer_count;
    int ssm_layer_count;
    int uses_hybrid_ssm;
    int uses_hybrid_moe;
    int uses_moe;
    int hyper_connection_count;
    int hyper_connection_rank;
    int xb2_elements;
    int hb_elements;
    int moe_total_experts;
    int moe_active_experts;
    int moe_expert_hidden_dim;
    int ssm_time_step_rank;
    int ssm_state_size;
    int ssm_inner_size;
    int ssm_group_count;
    int ssm_conv_kernel;
    const float *rope_frequencies;
    int rope_frequency_count;
    const BnGPURopeFrequencyPlan *rope_frequency_plans;
    int rope_frequency_plan_count;
    // Preserve standalone RoPE arithmetic after normalization (e.g. IMRoPE).
    // Zero retains the fused normalization/RoPE arithmetic contract.
    int separate_rope_norm;
} BnGPUActivationPlan;

// GPU compute backend vtable. The caller (e.g., Hull) fills this in
// with their GPU API. bitnet.c calls it for matvec dispatch.
// All function pointers may be NULL (graceful fallback to CPU SIMD).
#ifndef BN_GPU_BACKEND_DECLARED
#define BN_GPU_BACKEND_DECLARED
typedef struct BnGPUBackend BnGPUBackend;
#endif

/* Synchronous immutable F16 preparation. Reader supplies native [N,K] rows
 * in ascending 32-row chunks (last may be shorter); reader is not retained.
 * Backend packs [K,N] bits. Budgets cover final payload + one device tile and
 * one host F16 tile, respectively; caller source staging is additional.
 * Failure drains queued work and releases unpublished allocations. */
typedef struct {
    int rows, cols;
    size_t device_budget_bytes, host_budget_bytes;
    void *source;
    int (*read_rows)(void *source, int row, int count,
        uint16_t *out, size_t out_elems);
} BnGPUF16RowsPlan;

/* Immutable row-major scalar data, without conversion or repacking. Exact
 * 32-row reads except final; F16/BF16/F32 only. Device budget is final payload;
 * host budget is one tile. Publication follows completion of all uploads.
 * Reader/context borrowed only during the synchronous constructor. */
typedef struct {
    int type, rows, cols;
    size_t device_budget_bytes, host_budget_bytes;
    void *source;
    int (*read_rows)(void *source,int row,int count,void *out,size_t capacity_bytes);
} BnGPURawRowsPlan;

struct BnGPUBackend {
    /* Optional request resources. Cleanup owns its device/stream context and
     * finishes pending work without dereferencing the original backend.
     * Allocations and transfers are explicit boundaries, never weight uploads.
     * read/write complete before returning; synchronize completes queued ops.
     * Serialized access only. A device execution failure invalidates a request;
     * callers discard uncommitted outputs/state and release the request.
     */
    /* Dedicated request-owned cuBLAS setup, separate from BF16 math policy.
     * Query excludes caller-owned resident views. Reserve is setup-only and
     * idempotent. Execution queues no allocation, host copy, or synchronization.
     * Validation rejection queues nothing; a device error invalidates request.
     */
    /* Synchronous, explicit immutable packed|pre F16|post F16 upload. Exact
     * payload size and codebook marker required; no decoded/auxiliary cache.
     * Destroy with buffer_destroy after all consumers complete. Storage is
     * returned to the device on destruction (not kept in the idle weight pool).
     */
    void *(*trellis_buffer_create)(void *ctx,const void *bundle,size_t bytes,
        int format,int k,int n,uint32_t marker);
    /* Optional synchronous refill of exclusively owned, idle Trellis storage.
     * Exact original context/format/shape/marker/bytes required. Caller must
     * finish ALL consumers first; never refill a shared/published weight.
     * Rejection leaves data unchanged; a transfer failure may poison contents,
     * so discard the handle on failure. No device allocation or resizing.
     * Successful return completes DMA before non-blocking consumers may use it.
     */
    int (*trellis_buffer_write)(void *ctx,void *buffer,const void *bundle,
        size_t bytes,int format,int k,int n,uint32_t marker);
    /* Optional model-owned pinned staging, serialized and separate from
     * request activations. Create returns the owned state and borrowed writable
     * host storage; rejection leaves *host unchanged. Caller admits bytes in
     * BOTH its total host-staging and pinned budgets (overlapping charges).
     * Batch accepts 1..3 disjoint target buffers, sum(bytes)<=staging capacity.
     * Success completes all DMA before publication, using one completion fence.
     * Failure leaves items unchanged, frees newly created handles, and may
     * poison existing buffers: discard those. Host consumption is drained even
     * on failure. Destroy only after batch returns; frees all pinned storage.
     */
    void *(*trellis_upload_create)(void *ctx,size_t bytes,void **host);
    int (*trellis_upload_batch)(void *ctx,void *upload,
        BnGPUTrellisUploadItem *items,int count);
    void (*trellis_upload_destroy)(void *ctx,void *upload);
    /* Dedicated packed layout, not an ordinary buffer_create/matvec handle.
     * Completion precedes publication; destroy after every consumer completes.
     * Storage is freed directly, not retained by the inactive weight pool. */
    void *(*buffer_create_f16_rows)(void *ctx, const BnGPUF16RowsPlan *plan);
    void *(*buffer_create_raw_rows)(void *ctx, const BnGPURawRowsPlan *plan);
    /* No reserve, allocation, transfer or synchronization during execution. */
    int (*ple_f16_from_bf16_resident)(void *ctx,void *request,
        const BnGPUPLEF16FromBF16ResidentPlan *plan);
    int (*rmsnorm_vector4_f16_f32_resident)(void *ctx,void *request,
        const BnGPUVector4RMSNormResidentPlan *plan);
    int (*rope_gptj_f16_resident)(void *ctx,void *request,const BnGPUGPTJF16ResidentPlan *plan);
    int (*rope_frequency_f32_resident)(void *ctx,void *request,const BnGPUF32RopeFrequencyResidentPlan *plan);
    int (*latent_qkv_f16_resident)(void *ctx,void *request,const BnGPUF16LatentQKVResidentPlan *plan);
    int (*index_candidate_use_f16_resident)(void *ctx,void *request,const BnGPUIndexCandidateUseResidentPlan *plan);
    int (*index_candidate_source_f16_resident)(void *ctx,void *request,const BnGPUIndexCandidateSourceResidentPlan *plan);
    int (*index_topk_f16_resident)(void *ctx,void *request,const BnGPUF16IndexTopKResidentPlan *plan);
    int (*latent_attention_f16_resident)(void *ctx,void *request,const BnGPUF16LatentAttentionResidentPlan *plan);
    int (*index_query_f16_resident)(void *ctx,void *request,const BnGPUF16IndexQueryResidentPlan *plan);
    int (*index_score_f16_resident)(void *ctx,void *request,const BnGPUF16IndexScoreResidentPlan *plan);
    int (*index_query_score_f16_resident)(void *ctx,void *request,const BnGPUF16IndexQueryScoreResidentPlan *plan);
    int (*swa_ring_update_f16_resident)(void *ctx,void *request,const BnGPUSWARingUpdateResidentPlan *plan);
    int (*paged_latent_publish_f16_resident)(void *ctx,void *request,const BnGPUPagedLatentPublishResidentPlan *plan);
    int (*latent_keys_f16_resident)(void *ctx,void *request,const BnGPUF16LatentKeysResidentPlan *plan);
    int (*compress_f16_resident)(void *ctx,void *request,const BnGPUCompressF16ResidentPlan *plan);
    int (*compressor_f16_resident)(void *ctx,void *request,const BnGPUF16CompressorResidentPlan *plan);
    /* Setup-only, validated immutable publication; one bounded allocation,
     * no host tensor staging. Dedicated handles reject ordinary matvec/matmul. */
    void *(*buffer_create_router_weight)(void *ctx,const BnGPURouterWeightPlan *plan);
    int (*router_sqrtsp_resident)(void *ctx,void *request,const BnGPURouterResidentPlan *plan);
    int (*embedding_hc4_f32_resident)(void *ctx,void *request,
        const BnGPUBF16EmbeddingHC4ResidentPlan *plan);
    /* Setup-only, idempotent. Shares the existing 16 MiB FP32/EXL3 cuBLAS
     * workspace, not an additional allocation if already reserved. Query
     * excludes activations and CUDA/library/context overhead. */
    size_t (*hc4_head_workspace_size)(void *ctx);
    int (*hc4_head_reserve)(void *ctx,void *request);
    int (*hc4_head_resident)(void *ctx,void *request,
        const BnGPUHC4HeadResidentPlan *plan);
    int (*hc4_carry_resident)(void *ctx,void *request,
        const BnGPUHC4CarryResidentPlan *plan);
    int (*hc4_apply_resident)(void *ctx,void *request,
        const BnGPUHC4ApplyResidentPlan *plan);
    /* Setup-only, idempotent: FP16 has a dedicated 32 MiB cuBLAS workspace;
     * FP32 shares the 16 MiB trellis workspace. Query totals both, excluding
     * resident views and library/context overhead. Execution allocates/copies/synchronizes
     * nothing. Validation rejection queues nothing; device failure poisons
     * the request. Call linear_f16_reserve before execution. */
    size_t (*linear_f16_workspace_size)(void *ctx);
    int (*linear_f16_reserve)(void *ctx, void *request);
    int (*grouped_output_f16_resident)(void *ctx,void *request,const BnGPUF16GroupedOutputResidentPlan *plan);
    int (*linear_f16_resident)(void *ctx, void *request,
        const BnGPUF16LinearResidentPlan *plan);
    int (*ffn_f16_resident)(void *ctx, void *request,
        const BnGPUF16FFNResidentPlan *plan);
    size_t (*trellis_projection_f16_workspace_size)(void *ctx);
    int (*trellis_projection_f16_reserve)(void *ctx, void *request);
    int (*trellis_projection_f16_resident)(void *ctx, void *request,
        const BnGPUTrellisProjectionF16ResidentPlan *plan);
    int (*trellis_ffn_f16_resident)(void *ctx, void *request,
        const BnGPUTrellisFFNResidentPlan *plan);
    /* Setup-only private lock allocation/zeroing and kernel resource query.
     * Workspace size includes locks only. Temporary slabs use the public
     * helper. Reserve is idempotent; unsupported cooperative residency fails. */
    size_t (*trellis_fused_workspace_size)(void *ctx);
    int (*trellis_fused_reserve)(void *ctx,void *request);
    int (*trellis_fused_launch_info)(void *ctx,void *request,int tile_m,int tile_n,BnGPUFusedEXL3LaunchInfo *out);
    int (*trellis_fused_resident)(void *ctx,void *request,const BnGPUFusedEXL3ResidentPlan *plan);
    int (*trellis_fused_group_resident)(void *ctx,void *owner,
        const BnGPUFusedEXL3GroupPlan *plan,BnGPUFusedEXL3GroupStats *stats);
    int (*trellis_decode_resident)(void *ctx,void *request,const BnGPUDecodeEXL3ResidentPlan *plan);
    /* Explicit assignment-sort workspace setup, then asynchronous views-only operations. */
    size_t (*native_moe_index_workspace_size)(void *ctx,int max_assignments);
    int (*native_moe_index_reserve)(void *ctx,void *request,int max_assignments);
    int (*native_moe_index_resident)(void *ctx,void *request,const BnGPUNativeMoEIndexPlan *plan);
    int (*native_moe_group_resident)(void *ctx,void *request,const BnGPUNativeMoEGroupPlan *plan);
    int (*native_moe_reduce_resident)(void *ctx,void *request,const BnGPUNativeMoEReducePlan *plan);
    /* Setup-only source MMA rate probe; no cached global decision. Caller
     * may probe once during backend/model setup and pin the result for each
     * request. Probe uses 1KiB temporary payload plus CUDA events. */
    int (*trellis_folded_probe)(void *ctx,int *enabled,float *rate_ratio);
    /* Reuses existing 16MiB EXL3 cuBLAS workspace. Explicit f16acc_enabled is
     * 0/1; reserve is idempotent only for the same immutable request policy.
     * Source shape/device rules choose FP32 cuBLAS or 32-term FP16 partials.
     * No probing, allocation, transfers or fences during resident execution. */
    size_t (*trellis_folded_workspace_size)(void *ctx);
    int (*trellis_folded_reserve)(void *ctx,void *request,int f16acc_enabled);
    int (*trellis_folded_gemm_info)(void *ctx,void *request,int batch,int rows,
        int k,int n,BnGPUFoldedGemmInfo *out);
    int (*trellis_folded_resident)(void *ctx,void *request,
        const BnGPUFoldedEXL3ResidentPlan *plan);
    int (*trellis_folded_ffn_resident)(void *ctx,void *request,
        const BnGPUFoldedEXL3FFNResidentPlan *plan);

    void *(*resident_request_create)(void *ctx);
    void (*resident_request_destroy)(void *request);
    int (*resident_alloc)(void *request, int type, size_t count,
                          BnGPUResidentView *out);
    int (*resident_write)(void *request, BnGPUResidentView dst, const void *host);
    /* Explicit setup: 1..8 pinned slots, fixed capacity, payload budget excludes
     * allocator/page/event overhead. Matching reserve is idempotent; no resize.
     * Acquire: 0 success, BN_GPU_UPLOAD_BUSY pending DMA, -1 invalid/failure;
     * output unchanged unless successful. No waiting/allocation on acquire.
     * Submit queues H2D on the request stream and records completion without
     * fencing. Lease stays acquired on validation rejection. Cancel abandons
     * an acquired lease. Teardown finishes pending work before releasing pins.
     */
    int (*resident_upload_reserve)(void *request, unsigned slots,
                                   size_t bytes_per_slot, size_t budget_bytes);
    int (*resident_upload_acquire)(void *request, unsigned slot,
                                   BnGPUHostUploadLease *out);
    int (*resident_upload_submit)(void *request, const BnGPUHostUploadLease *lease,
                                  BnGPUResidentView dst);
    int (*resident_upload_cancel)(void *request, const BnGPUHostUploadLease *lease);
    /* Queue a bitwise zero fill of a validated mutable view. It is ordered
     * with subsequent request operations; synchronize/read establishes host
     * completion. Invalid views queue nothing and do not invalidate a request. */
    int (*resident_zero)(void *request, BnGPUResidentView dst);
    int (*resident_read)(void *request, BnGPUResidentView src, void *host);
    int (*resident_synchronize)(void *request);
    /* Reserve workspace outside execution/capture. The prepared GDN operator
     * queues no allocations, host transfers or synchronization. It never writes
     * initial_state; next_state/out become valid only after successful completion.
     * Validation rejection queues nothing. The first CUDA profile supports
     * hk=16,hv=48,dk=dv=128,q_scale=1/sqrt(128), inverse reduction profile4.
     */
    /* Pure host payload query for the supported GDN profile. Returns zero for
     * invalid/unsupported capacities. Excludes resident tensors, GEMM workspace
     * and allocator/context overhead; performs no CUDA calls or allocations. */
    size_t (*gdn_prefill_bf16_workspace_size)(void *ctx, int max_tokens);
    int (*gdn_prefill_bf16_reserve)(void *ctx, void *request, int max_tokens);
    int (*gdn_prefill_bf16_resident)(void *ctx, void *request,
                                   const BnGPUGDNPrefillResidentPlan *plan);
    /* Separate CP workspace. Query is pure host; reserve is setup-only and
     * fixes the minimum chunk size. Larger execution chunks reuse the same
     * bounded storage. Execution queues only kernels and device copies.
     * Validation rejection leaves the request and all tensors untouched. */
    /* Pure host reference chunk selection using cached SM count, 0 if unsupported. */
    int (*gdn_cp_prefill_chunk_tokens)(void *ctx, int n_tokens);
    size_t (*gdn_cp_prefill_bf16_workspace_size)(void *ctx, int max_tokens, int chunk_tokens);
    int (*gdn_cp_prefill_bf16_reserve)(void *ctx, void *request, int max_tokens, int chunk_tokens);
    int (*gdn_cp_prefill_bf16_resident)(void *ctx, void *request,
                                      const BnGPUGDNCPPrefillResidentPlan *plan);
    uint32_t resident_gdn_decode_profiles; /* bit per BnGPUGDNDecodeProfile */
    uint32_t resident_gdn_prefill_recurrences; /* bit per BnGPUGDNPrefillRecurrence */
    /* Setup-only event storage. CP scratch is reserved separately on the
     * coordinator for the largest individual sequence. Execution reuses that
     * bounded scratch, queues sequences in order, and fences state owners via
     * stream events. Validation rejects the entire group before queueing.
     * Execution failure invalidates coordinator and all participants. */
    size_t (*gdn_cp_prefill_group_workspace_size)(void *ctx, int capacity);
    int (*gdn_cp_prefill_group_reserve)(void *ctx, void *request, int capacity, size_t budget);
    int (*gdn_cp_prefill_group_resident)(void *ctx, void *request,
                                      const BnGPUGDNCPPrefillGroupPlan *plan);
    int (*gdn_prepare_bf16_resident)(void *ctx, void *request,
                                   const BnGPUGDNPrepareResidentPlan *plan);
    int (*gdn_decode_bf16_resident)(void *ctx, void *request,
                                  const BnGPUGDNDecodeResidentPlan *plan);
    /* Setup-only default split estimator; returns -1 for unsupported geometry.
     * Execution accepts explicit legal splits so the scheduler can pin math. */
    int (*clamped_swiglu_bf16_resident)(void *ctx, void *request,
                                      const BnGPUClampedSwiGLUResidentPlan *plan);
    int (*mhc_pre_bf16_split_count)(void *ctx, int n_tokens, int dim);
    int (*mhc_pre_bf16_resident)(void *ctx, void *request,
                               const BnGPUMHCPreResidentPlan *plan);
    int (*mhc_fused_bf16_split_count)(void *ctx, int n_tokens, int dim);
    int (*mhc_fused_post_pre_bf16_resident)(void *ctx, void *request,
                                         const BnGPUMHCFusedResidentPlan *plan);
    int (*mhc_post_bf16_resident)(void *ctx, void *request,
                                const BnGPUMHCPostResidentPlan *plan);
    int (*stream_mean_bf16_resident)(void *ctx,void *request,
                                     const BnGPUStreamMeanResidentPlan *plan);
    int (*kda_recurrent_bf16_resident)(void *ctx, void *request,
                                  const BnGPUKDARecurrentResidentPlan *plan);
    /* Setup-only bounded descriptor workspace for 1..256 participants. Query
     * includes host/device descriptors, host event table and container; excludes
     * driver/event/allocator overhead. Matching reserve is idempotent, no resize.
     * Execution copies descriptors only, never activations or recurrent state.
     * Returns 0 queued, GROUP_BUSY if pinned descriptors are still being copied,
     * or -1 rejected/failed. Invalid plans queue nothing. Caller serializes all
     * access and keeps live participant/coordinator handles through submission.
     * Ready/done stream events order producers and subsequent participant work.
     * After success, either participant or coordinator teardown safely finishes
     * its outstanding dependencies. Outputs/next state require successful
     * participant and coordinator fences (including upstream logical errors).
     * Device submission failure fences the coordinator and poisons participants.
     */
    /* Optional bounded staging queue1..64 slots. Reserve on an empty operator
     * workspace instead of its single-slot reserve. The device table remains
     * ordered on the coordinator stream; pinned source slots/events rotate.
     * Query includes explicit host/device payload, excluding CUDA internals.
     * Full queues return GROUP_BUSY without overwriting in-flight sources. */
    size_t (*resident_group_queue_workspace_size)(void *ctx,
        BnGPUResidentGroupKind kind,int capacity,int slots);
    int (*resident_group_queue_reserve)(void *ctx,void *coordinator,
        BnGPUResidentGroupKind kind,int capacity,int slots,size_t budget_bytes);
    size_t (*gdn_decode_bf16_group_workspace_size)(void *ctx, int max_sequences);
    int (*gdn_decode_bf16_group_reserve)(void *ctx, void *coordinator,
        int max_sequences, size_t budget_bytes);
    int (*gdn_decode_bf16_group_resident)(void *ctx, void *coordinator,
        const BnGPUGDNDecodeGroupPlan *plan);

    int (*causal_conv_bf16_resident)(void *ctx, void *request,
                                   const BnGPUCausalConvResidentPlan *plan);
    /* Same BF16 activation/history ownership contract, but weight is immutable
     * F32[dim,kernel]. Accumulate ordered F32 FMAs without rounding products;
     * SiLU uses the F32 sum and rounds only its output to BF16. No allocation,
     * host transfer, internal fence, bias, or Q/K normalization. */
    int (*causal_conv_f32_weight_bf16_resident)(void *ctx, void *request,
                                   const BnGPUCausalConvResidentPlan *plan);
    /* Independent descriptor storage from GDN groups, with the same bounded
     * reserve/BUSY, ownership, fence and teardown contract. One token per row;
     * shared BF16 weights [dim,kernel], kernel2/3/4, history [kernel-1,dim]. */
    size_t (*causal_conv_bf16_group_workspace_size)(void *ctx, int max_sequences);
    int (*causal_conv_bf16_group_reserve)(void *ctx, void *coordinator,
        int max_sequences, size_t budget_bytes);
    int (*causal_conv_bf16_group_resident)(void *ctx, void *coordinator,
        const BnGPUCausalConvGroupPlan *plan);

    int (*causal_conv_bf16_prefill_group_resident)(void *ctx, void *coordinator,
        const BnGPUCausalConvPrefillGroupPlan *plan);

    int (*ple_conv_bf16_resident)(void *ctx, void *request,
                                const BnGPUPLEConvResidentPlan *plan);
    int (*ple_gate_bf16_resident)(void *ctx, void *request,
                                const BnGPUPLEGateResidentPlan *plan);
    uint32_t resident_gated_norm_profiles; /* BF16-weight entry point's profile bits */
    int (*gated_rmsnorm_bf16_resident)(void *ctx, void *request,
                                     const BnGPUGatedRMSNormResidentPlan *plan);
    /* Initial F32-weight contract: GLM_R2, sigmoid, dim128, positive epsilon.
     * BF16 x/z/out, immutable F32 gamma, disjoint output, no copy/alloc/fence. */
    int (*gated_rmsnorm_f32_weight_bf16_resident)(void *ctx, void *request,
                                     const BnGPUGatedRMSNormResidentPlan *plan);
    /* Bit i declares reduction i supported by all three resident norm
     * callbacks (including add3). Zero leaves legacy callback use unchanged;
     * explicit request-profile admission must not infer support from pointers. */
    uint32_t resident_rmsnorm_reductions;
    /* Additional explicitly declared plain-only RMS reduction profiles. */
    uint32_t resident_plain_rmsnorm_reductions;
    uint32_t resident_ffn_activation_profiles; /* bit per BnGPUBF16FFNActivationProfile */
    int (*add_rmsnorm_bf16_resident)(void *ctx, void *request,
                                   const BnGPUAddRMSNormResidentPlan *plan);
    int (*add3_rmsnorm_bf16_resident)(void *ctx, void *request,
                                    const BnGPUAdd3RMSNormResidentPlan *plan);
    int (*rmsnorm_bf16_resident)(void *ctx, void *request,
                               const BnGPURMSNormResidentPlan *plan);
    int (*scaled_silu_bf16_resident)(void *ctx, void *request,
                                   const BnGPUScaledSiluResidentPlan *plan);
    int (*moe_topk_bf16_resident)(void *ctx, void *request,
                                 const BnGPUMoETopKBF16ResidentPlan *plan);
    int (*moe_group_bf16_resident)(void *ctx, void *request,
                                  const BnGPUMoEGroupBF16ResidentPlan *plan);
    int (*moe_reduce_bf16_resident)(void *ctx, void *request,
                                   const BnGPUMoEReduceBF16ResidentPlan *plan);
    int (*moe_shared_bf16_resident)(void *ctx, void *request,
                                    const BnGPUMoESharedBF16ResidentPlan *plan);
    int (*moe_project_bf16_resident)(void *ctx, void *request,
                                    const BnGPUMoEProjectBF16ResidentPlan *plan);
    int (*moe_ffn_bf16_resident)(void *ctx, void *request,
                                const BnGPUMoEFFNBF16ResidentPlan *plan);
    /* Explicit bounded setup. Matching reserve is idempotent, no resize.
     * Acquire is nonblocking: 0 success, BUSY pending COMPUTE, -1 invalid.
     * Initial raw profiles: Q4_K/Q5_K gate/up, Q5_1/Q8_0 down.
     * Submit validates the entire bank before H2D, then queues copy-stream
     * upload, compute-stream wait/FFNs and a completion event. No allocation
     * or host wait during submission. Validation rejection retains lease.
     * Cancel applies only to an acquired lease. Teardown drains both streams.
     * Access serialized by caller, including calls on different slots. */
    int (*moe_bank_reserve)(void *ctx, void *request, const BnGPUMoEBankConfig *config);
    int (*moe_bank_acquire)(void *request, unsigned slot, int gate_type, int down_type,
                            BnGPUMoEBankLease *out);
    int (*moe_bank_cancel)(void *request, const BnGPUMoEBankLease *lease);
    int (*moe_bank_submit)(void *ctx, void *request, const BnGPUMoEBankResidentPlan *plan);
    /* Same geometry/registry formats across requests. Uses existing grouped
     * reserves; no allocation or host activation transfer. One H2D expert
     * upload feeds joint cohort kernels on the owner stream. Private vector/
     * MMA8/24/64 precision buckets are retained; activation/quantization each
     * use one cohort launch. Kernel descriptors are bounded by-value launch
     * arguments, not additional retained buffers. Inputs/outputs are device-
     * event ordered with each consumer stream; no steady-state host fence.
     * Source completion waits for ALL consumers, protecting reuse/teardown.
     * Source owner may be a consumer only at lease.slot. Bound expert caches
     * require a current owner-issued union lease and its cache_generation. BUSY queues nothing and preserves all leases;
     * retry the same cohort. Invalid metadata queues nothing. A device error
     * drains/poisons the owner and every consumer; discard them all. */
    int (*moe_bank_submit_group)(void *ctx, void *owner,
                                const BnGPUMoEBankGroupPlan *plan);
    int (*moe_bank_stats)(void *request, BnGPUMoEBankStats *out);
    /* Explicit original/grouped A8 workspace, separate from bank weight and
     * linear scratch budgets. Fixed capacity; matching setup is idempotent. */
    int (*moe_bank_input_reserve)(void *ctx, void *request, int max_tokens, size_t budget_bytes);
    int (*moe_bank_prepare_input)(void *ctx, void *request, const BnGPUMoEBankInputPlan *plan,
                                  uint64_t *generation);
    /* Separate payload budgets for pinned descriptors and device descriptors
     * plus grouped down A8 scratch. Fixed capacity, matching setup idempotent. */
    int (*moe_bank_grouped_reserve)(void *ctx, void *request, int max_tokens,
                                    size_t host_budget_bytes, size_t device_budget_bytes);
    void *(*expert_cache_create)(void *ctx, const BnGPUExpertCacheConfig *config);
    void (*expert_cache_destroy)(void *cache);
    int (*expert_cache_memory)(void *cache, BnGPUExpertCacheMemory *out);
    int (*expert_cache_preload)(void *cache, size_t host_budget_bytes,
        BnGPUExpertCacheRead read, void *source, BnGPUExpertCachePreloadInfo *out);
    /* Bind before execution, reserving two pointer tables per bank (one host,
     * one device). Begin pre-pins all layer hits, end records consumption on
     * this request's stream. BUSY makes no layer/counter changes. Cancel closes
     * a failed/incomplete layer without making its unfilled slots visible. */
    int (*expert_cache_bind)(void *request, void *cache,
                              size_t host_budget_bytes, size_t device_budget_bytes);
    int (*expert_cache_begin)(void *request, int layer, int prefill,
                               const int *offsets, int n_tokens, BnGPUExpertCacheLayer *out);
    /* Exclusive union layer lease: pins all selected hits once. Only experts
     * used by decode rows admit/promote; prefill-only selections borrow.
     * Stats charge unique union placement work to owner. BUSY/invalid input
     * leaves output and membership unchanged. End/cancel is owner-issued;
     * cancel/teardown of any participant releases the whole cohort lease.
     * Existing bank completion protects pending weights across cancellation.
     * Cache/model outlive all participants; operations are caller-serialized. */
    int (*expert_cache_begin_group)(void *owner, int layer,
        const BnGPUExpertCacheGroupRow *rows, int count, BnGPUExpertCacheLayer *out);
    int (*expert_cache_end)(void *request, const BnGPUExpertCacheLayer *layer);
    int (*expert_cache_cancel)(void *request);
    int (*expert_cache_stats)(void *request, BnGPUExpertCacheStats *out);
    int (*quant_rows_bf16_resident)(void *ctx, void *request,
                                   const BnGPUQuantRowsBF16ResidentPlan *plan);
    int (*hyper_connection_mix_bf16_resident)(void *ctx, void *request,
                                            const BnGPUHyperMixResidentPlan *plan);
    int (*hyper_connection_combine_bf16_resident)(void *ctx, void *request,
                                                const BnGPUHyperCombineResidentPlan *plan);
    /* Read-only explicit device scratch payload, no allocation or device work.
     * Zero means unsupported/invalid. Excludes driver/allocator bookkeeping. */
    size_t (*embedding_bf16_workspace_size)(void *ctx);
    int (*embedding_bf16_reserve)(void *ctx, void *request);
    int (*embedding_bf16_resident)(void *ctx, void *request,
                                 const BnGPUEmbeddingResidentPlan *plan);
    int (*qk_prep_bf16_resident)(void *ctx, void *request,
                               const BnGPUQKPrepResidentPlan *plan);
    /* Capacity 1..256, fixed until request destruction. Query counts explicit
     * host/device payload, excluding CUDA event/allocator overhead. Group calls
     * validate all aliases before submission, then join participant streams.
     * BUSY submits nothing. Success allocates nothing and uses no host fence.
     * Partial submission failure fences the coordinator and poisons participants.
     * Fence coordinator AND participants before committing logical progress. */
    size_t (*qk_prep_group_workspace_size)(void *ctx, int capacity);
    int (*qk_prep_group_reserve)(void *ctx, void *coordinator,
                                int capacity, size_t budget_bytes);
    int (*qk_prep_group_bf16_resident)(void *ctx, void *coordinator,
                                     const BnGPUQKPrepGroupPlan *plan);
    int (*indexer_select_resident)(void *ctx,void *request,
        const BnGPUIndexerSelectResidentPlan *plan);
    int (*indexer_fp8_paged_score_resident)(void *ctx,void *request,
        const BnGPUIndexerFP8PagedScoreResidentPlan *plan);
    int (*indexer_fp8_score_resident)(void *ctx,void *request,
        const BnGPUIndexerFP8ScoreResidentPlan *plan);
    int (*indexer_fp8_append_resident)(void *ctx,void *request,
        const BnGPUIndexerFP8AppendResidentPlan *plan);
    int (*indexer_fp8_gather_resident)(void *ctx,void *request,
        const BnGPUIndexerFP8GatherResidentPlan *plan);
    int (*layernorm128_f32_bf16_resident)(void *ctx,void *request,
        const BnGPULayerNorm128F32BF16ResidentPlan *plan);
    int (*fwht128_fp8_resident)(void *ctx,void *request,
        const BnGPUFWHT128FP8ResidentPlan *plan);
    int (*kpool_fp8_resident)(void *ctx,void *request,
        const BnGPUKPoolFP8ResidentPlan *plan);
    int (*indexer_weight_scale_resident)(void *ctx,void *request,
        const BnGPUIndexerWeightScaleResidentPlan *plan);
    int (*qsa_indexer_bf16_resident)(void *ctx, void *request,
                                   const BnGPUQSAIndexerResidentPlan *plan);
    int (*qsa_score_bf16_resident)(void *ctx, void *request,
                                 const BnGPUQSAScoreResidentPlan *plan);
    int (*qsa_select_resident)(void *ctx, void *request,
                             const BnGPUQSASelectResidentPlan *plan);
    int (*paged_kv_append_bf16_resident)(void *ctx, void *request,
                                       const BnGPUPagedKVAppendResidentPlan *plan);
    int (*qsa_attention_bf16_prepare)(void *ctx);
    int (*qsa_attention_bf16_workspace_size)(void *ctx, int max_tokens, size_t *bytes);
    int (*qsa_attention_bf16_resident)(void *ctx, void *request,
                                     const BnGPUQSAAttentionResidentPlan *plan);
    size_t (*attention_bf16_workspace_size)(void *ctx, int max_tokens,
                                           int max_heads, int max_context);
    int (*attention_bf16_reserve)(void *ctx, void *request, int max_tokens,
                                 int max_heads, int max_context);
    uint32_t resident_attention_gate_profiles; /* bit per BnGPUAttentionGateProfile */
    uint32_t resident_attention_split_policies; /* bit per BnGPUAttentionSplitPolicy */
    int (*attention_bf16_resident)(void *ctx, void *request,
                                 const BnGPUAttentionResidentPlan *plan);
    /* Reserve fixed capacity1..256 on coordinator after reserving singleton
     * attention scratch on every participant. Budget excludes CUDA event and
     * allocator overhead. Execution validates all rows/aliases before submit,
     * joins streams, and returns BUSY without submission while metadata is in
     * use. No allocation/host fence on success; partial failure fences and
     * poisons the group. Fence coordinator AND participants before commit. */
    size_t (*attention_decode_group_workspace_size)(void *ctx, int capacity);
    int (*attention_decode_group_reserve)(void *ctx, void *coordinator,
                                         int capacity, size_t budget_bytes);
    int (*attention_decode_group_bf16_resident)(void *ctx, void *coordinator,
                                              const BnGPUAttentionDecodeGroupPlan *plan);
    int (*attention_prefill_group_bf16_resident)(void *ctx, void *coordinator,
        const BnGPUAttentionPrefillGroupPlan *plan);
    int (*transpose_axes_bf16_resident)(void *ctx, void *request,
                                       const BnGPUTransposeAxesResidentPlan *plan);
    int (*split_columns_resident)(void *ctx, void *request,
                                  const BnGPUSplitColumnsResidentPlan *plan);
    int (*gather_rows_resident)(void *ctx, void *request,
                               const BnGPUGatherRowsResidentPlan *plan);
    int (*join_columns_resident)(void *ctx, void *request,
                                 const BnGPUJoinColumnsResidentPlan *plan);
    int (*linear_a8_bf16_reserve)(void *ctx, void *request, size_t max_input_elements);
    int (*linear_a8_bf16_resident)(void *ctx, void *request,
                                  const BnGPUA8LinearResidentPlan *plan);
    /* Reserve A8 scratch plus a request-owned GEMM handle/workspace outside
     * execution/capture (0 input elements permits GEMM-only requests).
     * Generic linear queues no host transfers, allocations or synchronization. */
    /* Explicit GEMM-only request workspace payload; pure host query, no allocation. */
    size_t (*linear_bf16_workspace_size)(void *ctx);
    size_t (*linear_bf16_input_workspace_size)(void *ctx, size_t max_elements);
    int (*linear_bf16_reserve)(void *ctx, void *request, size_t max_input_elements);
    int (*add_bf16_resident)(void *ctx,void *request,const BnGPUAddBF16ResidentPlan *plan);
    int (*add_f32_resident)(void *ctx,void *request,const BnGPUAddF32ResidentPlan *plan);
    int (*moe_biased_sigmoid_f32_resident)(void *ctx,void *request,
                                        const BnGPUBiasedSigmoidF32ResidentPlan *plan);
    int (*linear_bf16_f32_weight_resident)(void *ctx,void *request,
                                         const BnGPUBF16F32LinearResidentPlan *plan);
    int (*mla_fp8_cache_append_resident)(void *ctx,void *request,
        const BnGPUMlaFP8CacheAppendResidentPlan *plan);
    int (*mla_fp8_attention_resident)(void *ctx,void *request,
        const BnGPUMlaFP8AttentionResidentPlan *plan);
    int (*head_linear_bf16_resident)(void *ctx,void *request,
        const BnGPUHeadLinearBF16ResidentPlan *plan);
    int (*linear_bf16_resident)(void *ctx, void *request,
                              const BnGPUBF16LinearResidentPlan *plan);
    /* Complete validation precedes submission. No allocation, host transfer,
     * synchronization, or mutable backend-context scratch during execution. */
    int (*dense_ffn_bf16_resident)(void *ctx, void *request,
                                 const BnGPUFFNResidentPlan *plan);
    uint32_t resident_ffn_layouts; /* bit per BnGPUBF16FFNLayout; zero is legacy separate */
    // Upload quantized weight data to GPU. Returns opaque buffer handle.
    // type: BN_GGUF_TENSOR_* constant, or -1 for untyped auxiliary bytes.
    // data/size: raw tensor bytes; the consuming operation defines auxiliary layout.
    // Returns NULL on failure.
    void *(*buffer_create)(void *ctx, const void *data, size_t size,
                           int type, int rows, int cols);
    // Upload quantized weight data without optional backend-side auxiliary
    // caches. Optional; callers use this for memory-sensitive resident caches.
    void *(*buffer_create_quant_only)(void *ctx, const void *data, size_t size,
                                      int type, int rows, int cols);
    // Wrap immutable host-backed weight storage without copying. Optional;
    // callers must keep data alive for the lifetime of the returned handle.
    void *(*buffer_create_borrowed)(void *ctx, const void *data, size_t size,
                                    int type, int rows, int cols);
    // Wrap immutable host storage in a backend-native layout that can be
    // consumed by ordinary matvec graph operations. Optional.
    void *(*buffer_create_native_matvec_borrowed)(
        void *ctx, const void *data, size_t size, int type, int rows, int cols);
    int (*native_matvec_borrowed_supported)(
        void *ctx, size_t size, int type, int rows, int cols);
    size_t (*buffer_cache_charge)(
        void *ctx, size_t size, int type, int rows, int cols);
    // Upload K-quant weight data and request an FP32 auxiliary cache when the
    // backend can fit it. Optional; callers use this for paths that are faster
    // with resident dequantized weights.
    void *(*buffer_create_kquant_f32_cache)(void *ctx, const void *data,
                                            size_t size, int type,
                                            int rows, int cols);
    // Upload quantized weight data and force an FP16 auxiliary cache when the
    // backend can allocate it. Optional; callers use this only after their own
    // memory fit check for resident CUDA layouts.
    void *(*buffer_create_f16_cache)(void *ctx, const void *data,
                                     size_t size, int type,
                                     int rows, int cols);
    // Upper bound on additional resident bytes for buffer_create_f16_cache,
    // excluding raw weights but including packed layouts and companion caches.
    // SIZE_MAX means overflow. Optional; the fallback assumes one FP16 copy.
    size_t (*buffer_f16_cache_extra_bytes)(void *ctx, int type,
                                          int rows, int cols);
    void  (*buffer_destroy)(void *ctx, void *buffer);

    // Optional independent dense BF16 layout of a borrowed quantized handle.
    // Rounding stages are declared by the quant registry. The source remains
    // unchanged and may be destroyed after this call returns. The caller owns
    // and budgets the new handle (2 * rows * cols payload bytes).
    void *(*buffer_create_bf16_decoded)(void *ctx, void *source);
    /* Explicit registry-declared identity-projection weight materialization,
     * matching the Q8_1 activation basis used by generic quantized-weight
     * getters. Separate from direct dequantization; returns owned BF16. */
    void *(*buffer_create_bf16_identity_materialized)(void *ctx,void *source);
    /* Preparation only: concatenate two decoded BF16 matrices with matching
     * column counts. Inputs remain owned by the caller. Success returns one
     * owned parent and two borrowed row-view handles; destroy views before
     * parent. The parent owns the sole new device allocation. Failure leaves
     * all three initially-NULL outputs unchanged and frees partial storage. */
    int (*buffer_create_bf16_stacked_pair)(void *ctx, void *first, void *second,
        void **parent, void **first_view, void **second_view);
    /* Returns independently owned immutable storage, ready on return. */
    void *(*buffer_create_rotary_bf16)(void *ctx, const BnGPURotaryCachePlan *plan);

    // Upload quantized weight data with fused bias. Returns opaque buffer handle.
    // The bias data (float[bias_size/4]) is appended to the repacked weight buffer.
    // Returns NULL if not supported for this type, or on failure.
    // Optional (NULL = not supported; caller falls back to separate bias upload).
    void *(*buffer_create_biased)(void *ctx, const void *data, size_t size,
                                   int type, int rows, int cols,
                                   const void *bias, size_t bias_size);

    // Upload two adjacent logical weight tensors as one stacked GPU buffer.
    // Optional (NULL = caller combines data before buffer_create).
    void *(*buffer_create_stacked2)(void *ctx,
                                    const void *data0, size_t size0,
                                    const void *data1, size_t size1,
                                    int type, int rows, int cols);
    void *(*buffer_create_stacked3)(void *ctx,
                                    const void *data0, size_t size0,
                                    const void *data1, size_t size1,
                                    const void *data2, size_t size2,
                                    int type, int rows, int cols);
    void *(*buffer_create_stacked3_biased)(void *ctx,
                                           const void *data0, size_t size0,
                                           const void *data1, size_t size1,
                                           const void *data2, size_t size2,
                                           int type, int rows, int cols,
                                           const void *bias,
                                           size_t bias_size);

    // Quantized matvec: out[rows] = W[rows, cols] @ x[cols]
    // W_buf: opaque handle from buffer_create.
    // x: host float[cols], out: host float[rows] (GPU copies to/from device).
    // Returns 0 on success, nonzero on error (falls back to CPU).
    int (*matvec)(void *ctx, float *out, void *W_buf, const float *x,
                  int rows, int cols, int type);

    // Batch matvec: out[n_tokens * rows] = W @ X[n_tokens * cols]
    // Optional (NULL = repeated single matvec or CPU fallback).
    int (*matmul)(void *ctx, float *out, void *W_buf, const float *X,
                  int rows, int cols, int n_tokens, int type);

    // Native dense BF16 W/X/output with FP32 accumulation. W_buf must already
    // hold BF16 weights; this does not choose quantized activation arithmetic.
    // Supports aliasing when the host allocation has room for the output.
    // Rejected/failed calls leave host output unchanged.
    int (*matmul_bf16)(void *ctx, uint16_t *out, void *W_buf,
                        const uint16_t *X, int rows, int cols, int n_tokens);

    // Explicit BF16 GGUF linear dispatch. The raw handle identifies the format.
    // The caller supplies its corresponding independently budgeted decoded
    // BF16 handle when the backend-quant plan requires one; no lazy admission.
    // A8 execution permits decoded=NULL. Output/alias/failure
    // semantics match matmul_bf16. Unsupported formats reject explicitly.
    int (*matmul_quant_bf16)(void *ctx, uint16_t *out, void *raw,
        void *decoded, const uint16_t *X, int rows, int cols, int n_tokens);

    // Batched matmul: multiple W_i @ X projections sharing the same
    // X[n_tokens, x_cols]. Outputs go to host pointers in each op.
    // Optional (NULL = repeated matmul or CPU fallback).
    int (*matmul_batch)(void *ctx, const BnGPUMatvecOp *ops, int n_ops,
                        const float *X, int n_tokens, int x_cols);

    // Optional FP32 weighted RMS normalization of X[n_tokens, dim].
    // norm_buf holds at least dim FP32 weights. Supports out == X;
    // rejected calls leave host output unchanged for CPU fallback.
    int (*rmsnorm_batch)(void *ctx, float *out, void *norm_buf,
                         const float *X, int n_tokens, int dim, float eps);

    // Optional native BF16 input/output normalization, FP32 accumulation.
    // norm_buf contains dim BF16 parameters. weight_offset selects gamma=w
    // (0) or gamma=1+float(w) (1); parameter rounding precedes the FP32 add.
    // Supports out == X. Rejected calls leave host output unchanged.
    // This primitive alone does not declare a complete BF16 execution mode.
    int (*rmsnorm_bf16_batch)(void *ctx, uint16_t *out, void *norm_buf,
                             const uint16_t *X, int n_tokens, int dim,
                             float eps, int weight_offset);

    // Normalize the F32 sum of BF16 X/residual; separately return its BF16
    // rounded residual. gamma follows the BF16 RMSNorm offset contract.
    // Outputs must be disjoint, but may alias either input. Both host outputs
    // remain unchanged on rejection or failure.
    int (*add_rmsnorm_bf16_batch)(void *ctx, uint16_t *out,
        uint16_t *residual_out, void *norm_buf, const uint16_t *X,
        const uint16_t *residual, int n_tokens, int dim, float eps,
        int weight_offset);

    // Optional grouped RMSNorm for token-major [n_tokens, streams, dim].
    // Each stream has its own contiguous dim-element slice in norm_buf.
    int (*rmsnorm_grouped_batch)(void *ctx, float *out, void *norm_buf,
                                 const float *X, int n_tokens, int streams,
                                 int dim, float eps);

    // Optional elementwise FP32 sigmoid over a host-visible batch.
    // Supports out == X; rejected calls leave host output unchanged.
    int (*sigmoid_batch)(void *ctx, float *out, const float *X, int count);

    // Optional scaled SiLU over a host-visible hyperconnection low-rank batch.
    // Supports in-place operation; rejected calls leave host data unchanged.
    int (*hyper_connection_scaled_silu_batch)(
        void *ctx, float *values, int count, float scale);

    // Optional hyperconnection sigmoid, stream reduction, and mean over
    // host-visible token-major [tokens, streams, dim] inputs.
    int (*hyper_connection_mix_batch)(
        void *ctx, float *out, const float *norm, const float *gates,
        int n_tokens, int dim, int streams);

    // Optional HC pre-mix pipeline. Intermediates stay backend-resident;
    // outputs are [tokens,dim] mixed activations and [tokens,streams] injection.
    // Rejection leaves both host outputs unchanged. Inputs/outputs are disjoint.
    int (*hyper_connection_prefill_batch)(
        void *ctx, float *out, float *inject, const float *residual,
        void *norm, void *down, void *up, void *inject_weight,
        int n_tokens, int dim, int streams, int rank, float eps);

    // Optional hyperconnection scatter over host-visible prompt rows:
    // residual += block_out * (2 * sigmoid(inject / streams)).
    int (*hyper_connection_combine_batch)(
        void *ctx, float *residual, const float *block_out,
        const float *inject, int n_tokens, int dim, int streams);

    // Optional one-token BF16 GDN decode. X is packed logical [Q,K,V],
    // BA=[beta,alpha]. Resident natural A_log is F32 and dt_bias is BF16.
    // Host caller-owned F32 state is [hv,128,128] in [V,K] order.
    // Supports dk=dv=128, hv divisible by hk, Q/K additive epsilon 1e-6,
    // softplus threshold 20 and caller-supplied positive finite query scale.
    // out may alias X; state and BA must be disjoint from all other vectors.
    // Output and state both remain unchanged on failure.
    int (*gdn_decode_bf16)(void *ctx, uint16_t *out, float *state,
        void *a_log, void *dt_bias, const uint16_t *X, const uint16_t *BA,
        int hk, int dk, int hv, int dv, float q_scale);

    // Optional GDN prefill preparation. BF16 packed X/out=[Q,K,V], with
    // dim=2*hk*dk+hv*dv; BA=[beta,alpha] has 2*hv columns per token.
    // Normalizes Q/K with caller-supplied additive epsilon, preserves V bits, returns
    // FP32 log-decay and sigmoid beta [tokens,hv]. Resident A_log is natural
    // log decay (FP32), dt_bias is BF16. The caller supplies positive finite
    // norm_eps and nonnegative finite softplus_threshold. No query scaling or recurrence.
    // Supports dk/dv<=256 with matching padded power-of-two widths, as in
    // the reference operator. out may alias X; BA and FP32 output vectors
    // must be disjoint from all other vectors. All outputs unchanged on failure.
    int (*gdn_prefill_prepare_bf16)(void *ctx, uint16_t *out,
        float *log_decay, float *beta, void *a_log, void *dt_bias,
        const uint16_t *X, const uint16_t *BA, int n_tokens,
        int hk, int dk, int hv, int dv, float norm_eps, float softplus_threshold);

    // Optional BF16 causal convolution + SiLU, without bias. Weights are
    // resident BF16 [dim,kernel]; X/out are [n_tokens,dim]. Mutable caller-owned
    // history is BF16 [kernel-1,dim], oldest first. Supports kernels 2/3/4.
    // Products round to BF16 before F32 accumulation; SiLU rounds on output.
    // out may alias X; history must be disjoint from both. Output and history
    // remain unchanged on failure. No Q/K normalization or F32 fallback.
    int (*causal_conv_silu_bf16_batch)(void *ctx, uint16_t *out,
        uint16_t *history, void *weight, const uint16_t *X,
        int n_tokens, int dim, int kernel);

    // Optional stateful SSM convolution followed by explicit Q/K normalization.
    // qkv and conv_state are host-visible; convolution weights stay resident.
    // qk_norm_mode is a BnGPUSSMQKNormMode.
    int (*ssm_conv_l2norm_batch)(void *ctx, float *qkv, float *conv_state,
        void *conv_weight_buf, int n_tokens, int qkv_dim, int conv_kernel,
        int num_k_heads, int head_k_dim, float norm_eps, int qk_norm_mode);

    // Optional recurrent delta update followed by per-head gated RMSNorm.
    // Inputs are already projected, convolved, and Q/K-normalized. Alpha and
    // beta are raw projection results; the backend applies bias, softplus,
    // decay, and sigmoid. Caller-owned recurrent state is copied back.
    int (*ssm_delta_gate_batch)(void *ctx, float *out, float *state,
        const float *qkv, const float *z, const float *alpha,
        const float *beta, void *norm_buf, void *dt_bias_buf,
        void *a_log_buf, int n_tokens,
        int num_k_heads, int head_k_dim, int num_v_heads, int head_v_dim,
        float q_scale, float norm_eps, int sigmoid_gate);

    // Generic positional-mixing primitives. These operate on host-visible
    // arrays while the backend owns all temporary device storage.
    // Processes one token. batch_tokens preserves its logical batch shape.
    // reduction_threads=0 lets the backend select the batch reduction order;
    // explicit 32/128/512 modes retain their existing arithmetic contract.
    int (*signed_sqrt_gate)(void *ctx, float *gate, float *gated,
                            const float *key, const float *query,
                            const float *value, int dim, int streams,
                            int reduction_threads, int batch_tokens);
    int (*dilated_conv_silu)(void *ctx, float *out, const float *current,
                             const float *history, void *weight_buf,
                             int channels, int kernel, int dilation);

    // Optional out = ((RMSNorm(X) * post_scale) * norm_weight), each
    // multiplication rounded separately. norm_buf contains raw FP32 weights.
    // Supports out == X; rejected calls leave host output unchanged.
    int (*rmsnorm_scaled_batch)(void *ctx, float *out, void *norm_buf,
        const float *X, int n_tokens, int dim, float eps, float post_scale);

    // out = (RMSNorm(X) * norm_weight) + residual. The backend preserves its
    // model-graph contraction. Host output may alias X or residual; rejection
    // leaves it unchanged. No session cache or model state is modified.
    int (*rmsnorm_residual_batch)(void *ctx, float *out, void *norm_buf,
        const float *X, const float *residual, int n_tokens, int dim, float eps);

    // Batched matvec: encode multiple dispatches in one GPU submission.
    // All ops share the same input x[x_cols]. Outputs go to separate host ptrs.
    // Optional (NULL = fall back to individual matvec calls).
    // Returns 0 on success, -1 on error.
    int (*matvec_batch)(void *ctx, const BnGPUMatvecOp *ops, int n_ops,
                        const float *x, int x_cols);

    // Dense FFN fast path: out[dim] = down(activation(gate(x)) * up(x)).
    // All weight buffers are opaque handles from buffer_create. This is
    // optional and intended for backends that can keep FFN intermediates
    // resident across gate/up activation and down projection.
    int (*dense_ffn)(void *ctx, float *out,
                     void *gate_buf, void *up_buf, void *down_buf,
                     const float *x, int dim, int hidden_dim,
                     int gate_type, int up_type, int down_type,
                     int act_type);

    // Batched dense FFN fast path for prompt processing:
    // out[n_tokens, dim] = down(activation(gate(X)) * up(X)).
    // Optional; backends may keep all gate/up/down intermediates resident.
    int (*dense_ffn_batch)(void *ctx, float *out,
                           void *gate_buf, void *up_buf, void *down_buf,
                           const float *X, int n_tokens,
                           int dim, int hidden_dim,
                           int gate_type, int up_type, int down_type,
                           int act_type);

    // Explicit BF16 SwiGLU: down(silu(gate(X)) * up(X)). BF16 matrix outputs
    // and activation output; FP32 internal arithmetic follows the qualified
    // linear/activation contracts. Caller budgets any decoded weight handles.
    // One host input/output boundary; intermediates stay on the device.
    // No normalization or residual addition. Output may alias X; rejected or
    // failed calls leave host output unchanged. Optional; no F32 fallback.
    int (*dense_ffn_bf16)(void *ctx, uint16_t *out,
        const BnGPUBF16FFNWeights *weights, const uint16_t *X,
        int n_tokens, int dim, int hidden_dim);

    // FFN(norm(F32(X)+F32(residual))) and BF16(X+residual). All intermediate
    // vectors remain on device. No final FFN/residual addition is performed.
    // Disjointness, aliasing and transactional failure semantics match
    // add_rmsnorm_bf16_batch; no F32 execution fallback.
    int (*dense_ffn_bf16_add_norm)(void *ctx, uint16_t *out,
        uint16_t *residual_out, const BnGPUBF16FFNWeights *weights,
        void *norm_buf, const uint16_t *X, const uint16_t *residual,
        int n_tokens, int dim, int hidden_dim, float eps, int weight_offset);

    // Compose the routed and dense FFN branches on the device in one call.
    int (*moe_dense_residual_batch)(
        void *ctx, const BnGPUMoEDenseResidualBatch *batch);
    int (*moe_routed_dense_residual_batch)(
        void *ctx, const BnGPUMoERoutedDenseResidualBatch *batch);

    // Batched dense FFN with input RMSNorm fused into the backend. Optional.
    // This is used by prompt processing to avoid a per-layer CPU norm pass
    // before uploading the prompt activation batch to the backend.
    int (*dense_ffn_batch_norm)(void *ctx, float *out,
                                void *gate_buf, void *up_buf,
                                void *down_buf, void *norm_buf,
                                const float *X, int n_tokens,
                                int dim, int hidden_dim,
                                int gate_type, int up_type, int down_type,
                                int act_type, float norm_eps);

    // Same as dense_ffn_batch_norm, returning X + FFN(norm(X)).
    int (*dense_ffn_batch_norm_resid)(void *ctx, float *out,
                                      void *gate_buf, void *up_buf,
                                      void *down_buf, void *norm_buf,
                                      const float *X, int n_tokens,
                                      int dim, int hidden_dim,
                                      int gate_type, int up_type,
                                      int down_type, int act_type,
                                      float norm_eps);

    // Batched routed MoE FFN for prompt processing. X is the expert input
    // token batch [n_tokens, dim]. expert_offsets/counts describe flat
    // token_ids/weights assignments for each expert. Expert buffers must be
    // resident backend handles. Returns weighted sum in out[n_tokens, dim].
    int (*moe_ffn_batch)(void *ctx, float *out,
                         const BnGPUMoEPrefillExpert *experts,
                         int n_experts,
                         const int *expert_offsets,
                         const int *expert_counts,
                         const int *token_ids,
                         const float *weights,
                         const float *X,
                         int n_tokens, int dim, int hidden_dim,
                         int gate_type, int up_type, int down_type,
                         int act_type,
                         void *shared_gate_buf, void *shared_up_buf,
                         void *shared_down_buf, void *shared_gate_weight_buf,
                         int shared_hidden_dim,
                         int shared_gate_type, int shared_up_type,
                         int shared_down_type);

    // Raw gathered expert FFN outputs, before expert scaling and route weights.
    // Optional; rejected calls leave host output unchanged. up_buf may be NULL
    // for stacked gate/up weights, as with dense_ffn_batch.
    int (*moe_expert_ffn_batch)(void *ctx, float *out,
                                void *gate_buf, void *up_buf, void *down_buf,
                                const float *X, int n_tokens, int dim,
                                int hidden_dim, int gate_type, int up_type,
                                int down_type, int act_type,
                                const BnGPUMoEExpertBatchPlan *plan);

    // Reduce raw experts[nt,k,dim] with scales/weights[nt,k] in slot order.
    // Each expert scale multiplication rounds before its route weight.
    // k=2..15 uses FMA after the first slot; k=1/16 uses separate multiply/add.
    // Supports out == experts. Rejection leaves host output unchanged.
    int (*moe_reduce_batch)(void *ctx, float *out, const float *experts,
        const float *weights, const float *scales, int nt, int k, int dim);

    // Batched routed MoE FFN using monolithic all-expert resident handles.
    // indices/weights are [n_tokens, k] route results. Optional output_scales
    // are applied and rounded before route weighting. Returns the weighted sum
    // in out[n_tokens, dim].
    int (*moe_routed_ffn_batch)(void *ctx, float *out,
                                void *gate_all_buf, void *up_all_buf,
                                void *down_all_buf,
                                const int *indices,
                                const float *weights,
                                const float *output_scales,
                                const float *X,
                                int n_tokens, int dim, int hidden_dim,
                                int n_experts, int k,
                                int gate_type, int up_type,
                                int down_type, int act_type);

    // Batched MoE routing for prompt processing. X is [n_tokens, dim].
    // Returns indices/weights as [n_tokens, k] on the host.
    int (*moe_route_batch)(void *ctx, int *indices, float *weights,
                           void *router_buf, const float *X,
                           int n_tokens, int dim, int n_experts, int k,
                           int norm_topk_prob, float expert_weights_scale);

    // Combined batched MoE routing and resident routed FFN for prompt
    // processing. Avoids route readback and re-upload.
    int (*moe_route_routed_ffn_batch)(void *ctx, float *out,
                                      void *router_buf,
                                      void *gate_all_buf,
                                      void *up_all_buf,
                                      void *down_all_buf,
                                      const float *X,
                                      int n_tokens, int dim, int hidden_dim,
                                      int n_experts, int k,
                                      int gate_type, int up_type,
                                      int down_type, int act_type,
                                      int norm_topk_prob,
                                      float expert_weights_scale);

    // Same as moe_route_routed_ffn_batch, with input RMSNorm fused before
    // routing and residual add fused into the returned output:
    // out = X + MoE(norm(X)).
    int (*moe_route_routed_ffn_batch_norm_resid)(
                                      void *ctx, float *out,
                                      void *router_buf,
                                      void *gate_all_buf,
                                      void *up_all_buf,
                                      void *down_all_buf,
                                      void *shared_gate_buf,
                                      void *shared_up_buf,
                                      void *shared_down_buf,
                                      void *shared_gate_weight_buf,
                                      void *norm_buf,
                                      const float *X,
                                      int n_tokens, int dim, int hidden_dim,
                                      int n_experts, int k,
                                      int gate_type, int up_type,
                                      int down_type, int act_type,
                                      int shared_hidden_dim,
                                      int shared_gate_type,
                                      int shared_up_type,
                                      int shared_down_type,
                                      float norm_eps,
                                      int norm_topk_prob,
                                      float expert_weights_scale);

    // All attention-prefill hooks use attention_window == 0 for full causal
    // history; a positive value limits visible keys for each prompt query.
    // Batched causal attention for prompt processing:
    // out[n_tokens, n_heads * head_size] =
    // attention(Q[n_tokens, n_heads * head_size],
    //           K/V[n_tokens, n_kv_heads * head_size]).
    // Q and K must already include bias, norm, and RoPE. This prompt helper
    // handles only the current prompt window, so callers should use it only
    // when pos0 == 0 unless the backend documents broader cache support.
    int (*prefill_attention)(void *ctx, float *out,
                             const float *Q, const float *K, const float *V,
                             int n_tokens, int n_heads, int n_kv_heads,
                             int head_size, int kv_mul, int kv_dim,
                             float attention_scale, int attention_window);

    // Prepare projected Q/K (optional RMSNorm and NeoX RoPE), then causal
    // attention and optional sigmoid query gate. Gated Q has adjacent
    // [query,gate] vectors per head; q_row_stride includes any row padding.
    // Norm buffers contain FP32 vectors (typed F32 or raw auxiliary uploads).
    // No bias or value normalization is performed. Host out may alias Q;
    // K_out may alias K and receives F32 prepared keys before cache rounding.
    // On failure, host outputs remain untouched. No session KV state changes.
    int (*prefill_attention_prepared)(void *ctx, float *out, float *K_out,
        const float *Q, const float *K, const float *V,
        void *q_norm_buf, void *k_norm_buf,
        const BnGPUAttentionPrefillPlan *plan);

    // Prepared attention with unit-weight RMS normalization of projected V.
    // Returns F32 prepared K and normalized V before cache rounding; callers
    // own the session cache. out/K_out/V_out may alias Q/K/V respectively.
    // All three output ranges must be distinct. Failure leaves them untouched.
    int (*prefill_attention_prepared_v)(void *ctx, float *out,
        float *K_out, float *V_out, const float *Q, const float *K, const float *V,
        void *q_norm_buf, void *k_norm_buf,
        const BnGPUAttentionPrefillPlan *plan);

    // Backend preparation boundary for shared batched attention. Produces
    // compact Q plus prepared K and optional unit-normalized V, without
    // computing attention or changing session cache state.
    int (*prefill_qkv_prepared)(void *ctx, float *Q_out, float *K_out,
        float *V_out, const float *Q, const float *K, const float *V,
        void *q_norm_buf, void *k_norm_buf,
        const BnGPUAttentionPrefillPlan *plan);

    // One-token QK scores over the backend-resident prefix. Q/K are raw
    // projected FP32 rows; optional norm buffers request fused RMSNorm,
    // norm-weight multiplication, and RoPE. The backend commits current K/V
    // and returns packed attention probabilities plus the prepared Q/K rows.
    int (*decode_attention_scores_prepared)(void *ctx, float *scores,
        float *Q_out, float *K_out,
        const float *Q, const float *K, const float *V,
        void *q_norm_buf, void *k_norm_buf,
        const BnGPUAttentionPrefillPlan *plan);

    // Read-only preflight of prefix geometry, cache capacity and RoPE resources.
    // Returns 1 when supported, 0 otherwise. Does not require populated KV rows.
    int (*prefill_attention_prefix_supported)(void *ctx,
        const BnGPUAttentionPrefillPlan *plan);

    // Fused prompt attention + output projection. Optional; backends may keep
    // the attention intermediate resident before applying W_wo.
    int (*prefill_attention_wo)(void *ctx, float *out, void *wo_buf,
                                const float *Q, const float *K,
                                const float *V, int n_tokens,
                                int n_heads, int n_kv_heads, int head_size,
                                int kv_mul, int kv_dim, int wo_rows,
                                int wo_cols, int wo_type,
                                float attention_scale, int attention_window);

    // Fused prompt QK/WV matmul + Q/K norm/RoPE + attention + W_O.
    // Optional CUDA-oriented fast path. Writes processed K/V rows back to
    // K_out/V_out so the existing session KV cache remains authoritative.
    int (*prefill_qkv_attention_wo)(void *ctx, float *out,
                                    void *qk_buf, void *wv_buf, void *wo_buf,
                                    void *q_norm_buf, void *k_norm_buf,
                                    const float *X, float *K_out,
                                    float *V_out, int n_tokens, int dim,
                                    int n_heads, int n_kv_heads,
                                    int head_size, int kv_mul, int kv_dim,
                                    int qk_rows, int qk_type,
                                    int wv_rows, int wv_type,
                                    int wo_rows, int wo_cols, int wo_type,
                                    int qk_norm_per_head, float norm_eps,
                                    int pos0, int rope_dims,
                                    float attention_scale, int attention_window);

    // Same as prefill_qkv_attention_wo, with input RMSNorm fused into the
    // backend before QK/WV projection.
    int (*prefill_qkv_attention_wo_norm)(
                                    void *ctx, float *out,
                                    void *qk_buf, void *wv_buf, void *wo_buf,
                                    void *attn_norm_buf,
                                    void *q_norm_buf, void *k_norm_buf,
                                    const float *X, float *K_out,
                                    float *V_out, int n_tokens, int dim,
                                    int n_heads, int n_kv_heads,
                                    int head_size, int kv_mul, int kv_dim,
                                    int qk_rows, int qk_type,
                                    int wv_rows, int wv_type,
                                    int wo_rows, int wo_cols, int wo_type,
                                    int qk_norm_per_head, float norm_eps,
                                    int pos0, int rope_dims,
                                    float attention_scale, int attention_window);

    // Same as prefill_qkv_attention_wo_norm, returning X + Attention(norm(X)).
    int (*prefill_qkv_attention_wo_norm_resid)(
                                    void *ctx, float *out,
                                    void *qk_buf, void *wv_buf, void *wo_buf,
                                    void *attn_norm_buf,
                                    void *q_norm_buf, void *k_norm_buf,
                                    const float *X, float *K_out,
                                    float *V_out, int n_tokens, int dim,
                                    int n_heads, int n_kv_heads,
                                    int head_size, int kv_mul, int kv_dim,
                                    int qk_rows, int qk_type,
                                    int wv_rows, int wv_type,
                                    int wo_rows, int wo_cols, int wo_type,
                                    int qk_norm_per_head, float norm_eps,
                                    int pos0, int rope_dims,
                                    float attention_scale, int attention_window);

    // Full dense transformer layer prefill fast path:
    // X + Attention(norm(X)) + FFN(norm(...)), with K/V rows copied back for
    // the existing CPU-owned session KV cache unless K_out/V_out are NULL. If
    // K_out/V_out are NULL, CUDA may write directly to backend KV activation
    // buffers at kv_cache_off using kv_cache_stride rows. Optional hook.
    // CUDA may accept X == NULL to reuse the previous device-resident output,
    // and out == NULL to leave the new output resident for the next layer.
    int (*prefill_dense_layer)(
                                    void *ctx, float *out,
                                    void *qk_buf, void *wv_buf, void *wo_buf,
                                    void *gate_buf, void *up_buf,
                                    void *down_buf, void *attn_norm_buf,
                                    void *ffn_norm_buf,
                                    void *attn_post_norm_buf,
                                    void *ffn_post_norm_buf,
                                    void *q_norm_buf, void *k_norm_buf,
                                    void *q_bias_buf, void *k_bias_buf,
                                    void *v_bias_buf,
                                    const float *X, float *K_out,
                                    float *V_out, int n_tokens, int dim,
                                    int hidden_dim, int n_heads,
                                    int n_kv_heads, int head_size,
                                    int kv_mul, int kv_dim, int qk_rows,
                                    int qk_type, int wv_rows, int wv_type,
                                    int wo_rows, int wo_cols, int wo_type,
                                    int gate_type, int up_type,
                                    int down_type, int act_type,
                                    int qk_norm_per_head, int normalize_v,
                                    float norm_eps,
                                    int pos0, int rope_dims,
                                    size_t rope_freq_offset,
                                    uint32_t kv_cache_off,
                                    int kv_cache_stride,
                                    float attention_scale,
                                    float layer_output_scale,
                                    int attention_window,
                                    int final_ffn_last_row_only);

    // Full MoE transformer layer prefill fast path:
    // X + Attention(norm(X)) + MoE(norm(...)). CUDA may accept X == NULL to
    // reuse the previous device-resident output, and out == NULL to leave the
    // new output resident for the next layer.
    int (*prefill_moe_layer)(
                                    void *ctx, float *out,
                                    void *qk_buf, void *wv_buf, void *wo_buf,
                                    void *router_buf, void *gate_all_buf,
                                    void *up_all_buf, void *down_all_buf,
                                    void *shared_gate_buf,
                                    void *shared_up_buf,
                                    void *shared_down_buf,
                                    void *shared_gate_weight_buf,
                                    void *attn_norm_buf,
                                    void *ffn_norm_buf,
                                    void *q_norm_buf, void *k_norm_buf,
                                    void *q_bias_buf, void *k_bias_buf,
                                    void *v_bias_buf,
                                    const float *X, float *K_out,
                                    float *V_out, int n_tokens, int dim,
                                    int moe_hidden_dim, int n_experts,
                                    int experts_active, int n_heads,
                                    int n_kv_heads, int head_size,
                                    int kv_mul, int kv_dim, int qk_rows,
                                    int qk_type, int wv_rows, int wv_type,
                                    int wo_rows, int wo_cols, int wo_type,
                                    int gate_type, int up_type,
                                    int down_type, int act_type,
                                    int shared_hidden_dim,
                                    int shared_gate_type,
                                    int shared_up_type,
                                    int shared_down_type,
                                    int qk_norm_per_head, float norm_eps,
                                    int pos0, int rope_dims,
                                    uint32_t kv_cache_off,
                                    int kv_cache_stride,
                                    float attention_scale,
                                    int norm_topk_prob,
                                    float expert_weights_scale, int attention_window);

    // Hybrid/SSM prompt block fast path:
    // out[n_tokens, dim] = X + ssm_out(SSM(norm(X))). Backend owns and updates
    // its resident SSM recurrent state. Optional CUDA-oriented hook.
    // qk_norm_mode selects BnGPUSSMQKNormMode independently of FFN activation.
    int (*prefill_ssm_layer)(
                                    void *ctx, float *out,
                                    void *wqkv_buf, void *wz_buf,
                                    void *alpha_buf, void *beta_buf,
                                    void *qkvz_stacked_buf,
                                    void *ab_stacked_buf,
                                    void *ssm_out_buf, void *attn_norm_buf,
                                    void *conv1d_buf, void *dt_bias_buf,
                                    void *a_log_buf, void *ssm_norm_buf,
                                    void *ffn_gate_buf, void *ffn_up_buf,
                                    void *ffn_down_buf, void *ffn_norm_buf,
                                    const float *X, int n_tokens, int dim,
                                    int qkv_dim, int inner_dim,
                                    int num_k_heads, int head_k_dim,
                                    int num_v_heads, int head_v_dim,
                                    int conv_kernel, int ssm_idx,
                                    int wqkv_type, int wz_type,
                                    int alpha_type, int beta_type,
                                    int out_type, int hidden_dim,
                                    int ffn_gate_type, int ffn_up_type,
                                    int ffn_down_type, int act_type,
                                    int sigmoid_gate, float norm_eps,
                                    int qk_norm_mode,
                                    int *did_ffn);

    // GPU-resident forward pass: execute a backend-private lowered command list
    // as a single submission. All intermediate buffers stay on GPU. Only
    // readback_buf is copied to out_host.
    // Returns 0 on success, -1 on error (caller should fall back to CPU).
    int (*execute)(void *ctx, const void *ops, int n_ops,
                   int readback_buf, float *out_host, int out_len);

    // Initialize GPU-resident activation buffers from a backend-neutral plan.
    // Must be called after weight upload, before execute().
    // Returns 0 on success.
    int (*init_activations)(void *ctx, const BnGPUActivationPlan *plan);

    // Clear request-local mutable activation state while retaining allocated
    // backend buffers and immutable uploaded model data. Optional.
    int (*reset_activations)(void *ctx);

    // Free GPU-resident activation buffers.
    void (*free_activations)(void *ctx);

    // Write host data to a GPU-resident activation buffer.
    // buf_idx: backend activation value slot. offset/size in bytes.
    // Returns 0 on success, -1 on error.  Optional (NULL = not supported).
    int (*write_activation)(void *ctx, int buf_idx, const void *data,
                            size_t size, size_t offset);

    // Read GPU-resident activation buffer to host.
    // buf_idx: backend activation value slot. out: host buffer, size in bytes.
    // Returns 0 on success, -1 on error.  Optional (NULL = not supported).
    int (*read_activation)(void *ctx, int buf_idx, void *out,
                           size_t size, size_t offset);

    // Return argmax over a GPU-resident float buffer, optionally applying the
    // same repeat penalty used by greedy CPU sampling. Optional.
    int (*argmax_activation)(void *ctx, int buf_idx, int n,
                             const int *penalty_tokens, int n_penalty_tokens,
                             float repeat_penalty, int *out_token);

    // Compute W * activation[buf_idx] and return argmax without materializing
    // host logits. Optional; intended for greedy decode fast paths.
    int (*matvec_argmax_activation)(void *ctx, void *W_buf, int type,
                                    int rows, int cols, int buf_idx,
                                    const int *penalty_tokens,
                                    int n_penalty_tokens,
                                    float repeat_penalty, int *out_token);

    // Return free/total device memory in bytes. Optional.
    int (*memory_info)(void *ctx, size_t *free_bytes, size_t *total_bytes);

    // Prepare backend-owned memory and runtime state used by CPU-orchestrated
    // operations. Optional and idempotent.
    int (*prepare_cpu_operations)(void *ctx);

    // Select the backend's prepared native-quant weight layout before model
    // buffers are uploaded. Optional; the setting is backend-instance local.
    void (*configure_prepared_native_quant)(void *ctx, int enabled);

    void *ctx;  // opaque backend context

    // Capability flags (set by backend, checked by transformer)
    uint64_t caps;

    // Maximum expert count supported by the backend's GPU router. 0 means
    // the backend does not impose a shape limit beyond its capability bit.
    int max_moe_route_experts;

    // Concrete backend identity for planning/debugging. Future backends should
    // set this instead of relying on capability inference alone.
    BnGPUBackendKind kind;
    BnBackendRuntimePolicy runtime_policy;

    // Maximum storage-buffer binding size in bytes. 0 = unknown; callers
    // should use a conservative fallback when deciding whether to bind a
    // large weight buffer in a GPU-resident graph.
    size_t max_storage_binding_size;

    // Optional borrowed Q/K projection view for prefill_dense_layer only.
    // Keeps each child's quant format and owns no child buffers. Destroy the
    // returned view with buffer_destroy; keep both children alive while used.
    // NULL means unavailable, disabled, or allocation failure.
    void *(*buffer_create_prefill_qk_pair)(void *ctx, void *q, void *k);
    /* Optional setup-only destructive resize of one complete owned allocation.
     * Requires serialized access and completion of all borrowers. The backend
     * fences its request before releasing old storage. Contents are unspecified;
     * success replaces *allocation and invalidates every prior alias, including
     * slices. Same-size calls are idempotent. Validation leaves everything
     * unchanged; failure after release poisons the request (destroy it).
     * Free-before-allocate avoids retaining both payloads transiently.
     * CUDA bounds a request to 65535 allocations and each allocation to 65535
     * resizes; exhaustion rejects without mutation. */
    int (*resident_resize_discard)(void *request, BnGPUResidentView *allocation,
                                   size_t count);
};

int bn_gpu_backend_capture_runtime_policy(BnGPUBackend *gpu);
int bn_gpu_backend_runtime_policy_init(BnBackendRuntimePolicy *policy);
int bn_gpu_backend_capture_runtime_policy_from(
    BnGPUBackend *gpu, const BnBackendRuntimePolicy *policy);
void bn_gpu_backend_release_runtime_policy(BnGPUBackend *gpu);

static inline int bn_gpu_backend_can_create_buffer(
    const BnGPUBackend *gpu) {
    return gpu && gpu->buffer_create;
}

static inline void *bn_gpu_backend_create_prefill_qk_pair(
    BnGPUBackend *gpu, void *q, void *k) {
    if (!gpu || !gpu->buffer_create_prefill_qk_pair || !q || !k)
        return NULL;
    return gpu->buffer_create_prefill_qk_pair(gpu->ctx, q, k);
}

static inline void bn_gpu_backend_configure_prepared_native_quant(
    BnGPUBackend *gpu, int enabled) {
    if (gpu && gpu->configure_prepared_native_quant)
        gpu->configure_prepared_native_quant(gpu->ctx, enabled);
}

static inline BnGPUBackendKind bn_gpu_backend_kind(
    const BnGPUBackend *gpu) {
    return gpu ? gpu->kind : BN_GPU_BACKEND_UNKNOWN;
}

static inline int bn_gpu_backend_is_cuda(const BnGPUBackend *gpu) {
    return bn_gpu_backend_kind(gpu) == BN_GPU_BACKEND_CUDA;
}

static inline int bn_gpu_backend_is_metal(const BnGPUBackend *gpu) {
    return bn_gpu_backend_kind(gpu) == BN_GPU_BACKEND_METAL;
}

static inline int bn_gpu_backend_is_webgpu(const BnGPUBackend *gpu) {
    return bn_gpu_backend_kind(gpu) == BN_GPU_BACKEND_WEBGPU;
}

static inline int bn_gpu_backend_can_create_f16_rows_buffer(const BnGPUBackend *gpu) {
    return gpu && gpu->buffer_create_f16_rows && gpu->buffer_destroy;
}
static inline void *bn_gpu_backend_create_f16_rows_buffer(const BnGPUBackend *gpu,
    const BnGPUF16RowsPlan *plan) {
    return bn_gpu_backend_can_create_f16_rows_buffer(gpu)
        ? gpu->buffer_create_f16_rows(gpu->ctx, plan) : NULL;
}

static inline int bn_gpu_backend_can_create_router_weight(const BnGPUBackend *gpu) {
    return gpu && gpu->buffer_create_router_weight;
}
static inline void *bn_gpu_backend_create_router_weight(const BnGPUBackend *gpu,const BnGPURouterWeightPlan *plan) {
    return bn_gpu_backend_can_create_router_weight(gpu)?gpu->buffer_create_router_weight(gpu->ctx,plan):NULL;
}
static inline int bn_gpu_backend_can_create_raw_rows_buffer(const BnGPUBackend *gpu) {
    return gpu && gpu->buffer_create_raw_rows && gpu->buffer_destroy;
}
static inline void *bn_gpu_backend_create_raw_rows_buffer(const BnGPUBackend *gpu,
    const BnGPURawRowsPlan *plan) {
    return bn_gpu_backend_can_create_raw_rows_buffer(gpu)
        ? gpu->buffer_create_raw_rows(gpu->ctx,plan) : NULL;
}

static inline int bn_gpu_backend_can_create_quant_only_buffer(
    const BnGPUBackend *gpu) {
    return gpu && gpu->buffer_create_quant_only;
}

static inline int bn_gpu_backend_can_create_borrowed_buffer(
    const BnGPUBackend *gpu) {
    return gpu && gpu->buffer_create_borrowed;
}

static inline int bn_gpu_backend_can_create_native_matvec_borrowed_buffer(
    const BnGPUBackend *gpu) {
    return gpu && gpu->buffer_create_native_matvec_borrowed;
}

static inline int bn_gpu_backend_native_matvec_borrowed_supported(
    const BnGPUBackend *gpu, size_t size, int type, int rows, int cols) {
    return gpu && gpu->buffer_create_native_matvec_borrowed &&
           gpu->native_matvec_borrowed_supported &&
           gpu->native_matvec_borrowed_supported(
               gpu->ctx, size, type, rows, cols);
}

static inline size_t bn_gpu_backend_buffer_cache_charge(
    const BnGPUBackend *gpu, size_t size, int type, int rows, int cols) {
    if (gpu && gpu->buffer_cache_charge)
        return gpu->buffer_cache_charge(gpu->ctx, size, type, rows, cols);
    return size;
}

static inline int bn_gpu_backend_can_create_kquant_f32_cache_buffer(
    const BnGPUBackend *gpu) {
    return gpu && gpu->buffer_create_kquant_f32_cache;
}

static inline int bn_gpu_backend_can_create_f16_cache_buffer(
    const BnGPUBackend *gpu) {
    return gpu && gpu->buffer_create_f16_cache;
}

static inline size_t bn_gpu_backend_f16_cache_extra_bytes(
    const BnGPUBackend *gpu, int type, int rows, int cols) {
    if (gpu && gpu->buffer_f16_cache_extra_bytes)
        return gpu->buffer_f16_cache_extra_bytes(gpu->ctx, type, rows, cols);
    if (rows <= 0 || cols <= 0) return 0;
    if ((size_t)rows > SIZE_MAX / (size_t)cols / sizeof(uint16_t))
        return SIZE_MAX;
    return (size_t)rows * (size_t)cols * sizeof(uint16_t);
}

static inline int bn_gpu_backend_can_create_biased_buffer(
    const BnGPUBackend *gpu) {
    return gpu && gpu->buffer_create_biased;
}

static inline int bn_gpu_backend_can_create_stacked2_buffer(
    const BnGPUBackend *gpu) {
    return gpu && gpu->buffer_create_stacked2;
}

static inline int bn_gpu_backend_can_create_stacked3_buffer(
    const BnGPUBackend *gpu) {
    return gpu && gpu->buffer_create_stacked3;
}

static inline int bn_gpu_backend_can_create_stacked3_biased_buffer(
    const BnGPUBackend *gpu) {
    return gpu && gpu->buffer_create_stacked3_biased;
}

static inline int bn_gpu_backend_can_matvec(
    const BnGPUBackend *gpu) {
    return gpu && gpu->matvec;
}

static inline int bn_gpu_backend_can_matmul(
    const BnGPUBackend *gpu) {
    return gpu && gpu->matmul;
}

static inline int bn_gpu_backend_can_matvec_batch(
    const BnGPUBackend *gpu) {
    return gpu && gpu->matvec_batch;
}

static inline int bn_gpu_backend_can_matmul_batch(
    const BnGPUBackend *gpu) {
    return gpu && gpu->matmul_batch;
}

static inline int bn_gpu_backend_can_rmsnorm_residual_batch(const BnGPUBackend *gpu) {
    return gpu && gpu->rmsnorm_residual_batch;
}

static inline int bn_gpu_backend_can_rmsnorm_scaled_batch(const BnGPUBackend *gpu) {
    return gpu && gpu->rmsnorm_scaled_batch;
}

static inline int bn_gpu_backend_can_rmsnorm_batch(const BnGPUBackend *gpu) {
    return gpu && gpu->rmsnorm_batch;
}

static inline int bn_gpu_backend_can_add_rmsnorm_bf16_batch(const BnGPUBackend *gpu) {
    return gpu && gpu->add_rmsnorm_bf16_batch;
}

static inline int bn_gpu_backend_add_rmsnorm_bf16_batch(const BnGPUBackend *gpu,
    uint16_t *out, uint16_t *residual_out, void *norm, const uint16_t *X,
    const uint16_t *residual, int n_tokens, int dim, float eps, int weight_offset) {
    if (!bn_gpu_backend_can_add_rmsnorm_bf16_batch(gpu)) return -1;
    return gpu->add_rmsnorm_bf16_batch(gpu->ctx, out, residual_out, norm,
        X, residual, n_tokens, dim, eps, weight_offset);
}

static inline int bn_gpu_backend_can_dense_ffn_bf16_add_norm(const BnGPUBackend *gpu) {
    return gpu && gpu->dense_ffn_bf16_add_norm;
}

static inline int bn_gpu_backend_dense_ffn_bf16_add_norm(const BnGPUBackend *gpu,
    uint16_t *out, uint16_t *residual_out, const BnGPUBF16FFNWeights *weights,
    void *norm, const uint16_t *X, const uint16_t *residual, int n_tokens,
    int dim, int hidden_dim, float eps, int weight_offset) {
    if (!bn_gpu_backend_can_dense_ffn_bf16_add_norm(gpu)) return -1;
    return gpu->dense_ffn_bf16_add_norm(gpu->ctx, out, residual_out, weights,
        norm, X, residual, n_tokens, dim, hidden_dim, eps, weight_offset);
}

static inline int bn_gpu_backend_can_rmsnorm_bf16_batch(const BnGPUBackend *gpu) {
    return gpu && gpu->rmsnorm_bf16_batch;
}

static inline int bn_gpu_backend_can_create_rotary_bf16(const BnGPUBackend *gpu) {
    return gpu && gpu->buffer_create_rotary_bf16;
}

static inline void *bn_gpu_backend_create_rotary_bf16(const BnGPUBackend *gpu,
    const BnGPURotaryCachePlan *plan) {
    return bn_gpu_backend_can_create_rotary_bf16(gpu)
        ? gpu->buffer_create_rotary_bf16(gpu->ctx, plan) : NULL;
}

static inline int bn_gpu_backend_can_bf16_identity_materialize(const BnGPUBackend *gpu) {
    return gpu&&gpu->buffer_create_bf16_identity_materialized;
}
static inline void *bn_gpu_backend_buffer_create_bf16_identity_materialized(const BnGPUBackend *gpu,void *source) {
    return bn_gpu_backend_can_bf16_identity_materialize(gpu)?gpu->buffer_create_bf16_identity_materialized(gpu->ctx,source):NULL;
}

static inline int bn_gpu_backend_can_bf16_decode(const BnGPUBackend *gpu) {
    return gpu && gpu->buffer_create_bf16_decoded;
}

static inline void *bn_gpu_backend_buffer_create_bf16_decoded(
    const BnGPUBackend *gpu, void *source) {
    if (!bn_gpu_backend_can_bf16_decode(gpu)) return NULL;
    return gpu->buffer_create_bf16_decoded(gpu->ctx, source);
}

static inline int bn_gpu_backend_can_stack_bf16_pair(const BnGPUBackend *gpu) {
    return gpu && gpu->buffer_create_bf16_stacked_pair;
}
static inline int bn_gpu_backend_buffer_create_bf16_stacked_pair(
    const BnGPUBackend *gpu,void *first,void *second,
    void **parent,void **first_view,void **second_view) {
    if(!bn_gpu_backend_can_stack_bf16_pair(gpu))return -1;
    return gpu->buffer_create_bf16_stacked_pair(gpu->ctx,first,second,parent,first_view,second_view);
}

static inline int bn_gpu_backend_can_matmul_bf16(const BnGPUBackend *gpu) {
    return gpu && gpu->matmul_bf16;
}

static inline int bn_gpu_backend_can_matmul_quant_bf16(const BnGPUBackend *gpu) {
    return gpu && gpu->matmul_quant_bf16;
}

static inline int bn_gpu_backend_matmul_quant_bf16(const BnGPUBackend *gpu,
    uint16_t *out, void *raw, void *decoded, const uint16_t *X,
    int rows, int cols, int n_tokens) {
    if (!bn_gpu_backend_can_matmul_quant_bf16(gpu)) return -1;
    return gpu->matmul_quant_bf16(gpu->ctx, out, raw, decoded, X,
        rows, cols, n_tokens);
}

static inline int bn_gpu_backend_matmul_bf16(const BnGPUBackend *gpu,
    uint16_t *out, void *weight, const uint16_t *X,
    int rows, int cols, int n_tokens) {
    if (!bn_gpu_backend_can_matmul_bf16(gpu)) return -1;
    return gpu->matmul_bf16(gpu->ctx, out, weight, X, rows, cols, n_tokens);
}

static inline int bn_gpu_backend_moe_dense_residual_batch(
    const BnGPUBackend *gpu, const BnGPUMoEDenseResidualBatch *batch) {
    if (!gpu || !gpu->moe_dense_residual_batch) return -1;
    return gpu->moe_dense_residual_batch(gpu->ctx, batch);
}

static inline int bn_gpu_backend_moe_routed_dense_residual_batch(
    const BnGPUBackend *gpu,
    const BnGPUMoERoutedDenseResidualBatch *batch) {
    if (!gpu || !gpu->moe_routed_dense_residual_batch) return -1;
    return gpu->moe_routed_dense_residual_batch(gpu->ctx, batch);
}

static inline int bn_gpu_backend_can_rmsnorm_grouped_batch(
    const BnGPUBackend *gpu) {
    return gpu && gpu->rmsnorm_grouped_batch;
}

static inline int bn_gpu_backend_can_sigmoid_batch(const BnGPUBackend *gpu) {
    return gpu && gpu->sigmoid_batch;
}

static inline int bn_gpu_backend_can_hyper_connection_scaled_silu_batch(
    const BnGPUBackend *gpu) {
    return gpu && gpu->hyper_connection_scaled_silu_batch;
}

static inline int bn_gpu_backend_can_hyper_connection_mix_batch(
    const BnGPUBackend *gpu) {
    return gpu && gpu->hyper_connection_mix_batch;
}

static inline int bn_gpu_backend_can_hyper_connection_combine_batch(
    const BnGPUBackend *gpu) {
    return gpu && gpu->hyper_connection_combine_batch;
}

static inline int bn_gpu_backend_can_ssm_delta_gate_batch(
    const BnGPUBackend *gpu) {
    return gpu && gpu->ssm_delta_gate_batch;
}

static inline int bn_gpu_backend_can_gdn_decode_bf16(const BnGPUBackend *gpu) {
    return gpu && gpu->gdn_decode_bf16;
}

static inline int bn_gpu_backend_can_gdn_prefill_prepare_bf16(const BnGPUBackend *gpu) {
    return gpu && gpu->gdn_prefill_prepare_bf16;
}

static inline int bn_gpu_backend_can_causal_conv_silu_bf16_batch(
    const BnGPUBackend *gpu) {
    return gpu && gpu->causal_conv_silu_bf16_batch;
}

static inline int bn_gpu_backend_can_ssm_conv_l2norm_batch(
    const BnGPUBackend *gpu) {
    return gpu && gpu->ssm_conv_l2norm_batch;
}

static inline int bn_gpu_backend_can_dense_ffn(
    const BnGPUBackend *gpu) {
    return gpu && gpu->dense_ffn;
}

static inline int bn_gpu_backend_can_moe_reduce_batch(const BnGPUBackend *gpu) {
    return gpu && gpu->moe_reduce_batch;
}

static inline int bn_gpu_backend_can_moe_expert_ffn_batch(const BnGPUBackend *gpu) {
    return gpu && gpu->moe_expert_ffn_batch;
}

static inline int bn_gpu_backend_can_dense_ffn_batch(
    const BnGPUBackend *gpu) {
    return gpu && gpu->dense_ffn_batch;
}

static inline int bn_gpu_backend_can_dense_ffn_bf16(const BnGPUBackend *gpu) {
    return gpu && gpu->dense_ffn_bf16;
}

static inline int bn_gpu_backend_dense_ffn_bf16(const BnGPUBackend *gpu,
    uint16_t *out, const BnGPUBF16FFNWeights *weights, const uint16_t *X,
    int n_tokens, int dim, int hidden_dim) {
    if (!bn_gpu_backend_can_dense_ffn_bf16(gpu)) return -1;
    return gpu->dense_ffn_bf16(gpu->ctx, out, weights, X,
        n_tokens, dim, hidden_dim);
}

static inline int bn_gpu_backend_can_dense_ffn_batch_norm(
    const BnGPUBackend *gpu) {
    return gpu && gpu->dense_ffn_batch_norm;
}

static inline int bn_gpu_backend_can_dense_ffn_batch_norm_resid(
    const BnGPUBackend *gpu) {
    return gpu && gpu->dense_ffn_batch_norm_resid;
}

static inline int bn_gpu_backend_can_moe_ffn_batch(
    const BnGPUBackend *gpu) {
    return gpu && gpu->moe_ffn_batch;
}

static inline int bn_gpu_backend_can_moe_routed_ffn_batch(
    const BnGPUBackend *gpu) {
    return gpu && gpu->moe_routed_ffn_batch;
}

static inline int bn_gpu_backend_can_moe_route_batch(
    const BnGPUBackend *gpu) {
    return gpu && gpu->moe_route_batch;
}

static inline int bn_gpu_backend_can_moe_route_routed_ffn_batch(
    const BnGPUBackend *gpu) {
    return gpu && gpu->moe_route_routed_ffn_batch;
}

static inline int bn_gpu_backend_can_moe_route_routed_ffn_batch_norm_resid(
    const BnGPUBackend *gpu) {
    return gpu && gpu->moe_route_routed_ffn_batch_norm_resid;
}

static inline int bn_gpu_backend_can_prefill_attention(
    const BnGPUBackend *gpu) {
    return gpu && gpu->prefill_attention;
}

static inline int bn_gpu_backend_can_prefill_attention_prepared_v(
    const BnGPUBackend *gpu) {
    return gpu && gpu->prefill_attention_prepared_v;
}

static inline int bn_gpu_backend_can_prefill_attention_prepared(
    const BnGPUBackend *gpu) {
    return gpu && gpu->prefill_attention_prepared;
}

static inline int bn_gpu_backend_can_decode_attention_scores_prepared(
    const BnGPUBackend *gpu) {
    return gpu && gpu->decode_attention_scores_prepared;
}

static inline int bn_gpu_backend_can_check_prefill_prefix(
    const BnGPUBackend *gpu) {
    return gpu && gpu->prefill_attention_prefix_supported;
}

static inline int bn_gpu_backend_prefill_prefix_supported(
    const BnGPUBackend *gpu, const BnGPUAttentionPrefillPlan *plan) {
    return bn_gpu_backend_can_check_prefill_prefix(gpu) && plan &&
           gpu->prefill_attention_prefix_supported(gpu->ctx, plan) == 1;
}

static inline int bn_gpu_backend_can_prefill_attention_wo(
    const BnGPUBackend *gpu) {
    return gpu && gpu->prefill_attention_wo;
}

static inline int bn_gpu_backend_can_prefill_qkv_attention_wo(
    const BnGPUBackend *gpu) {
    return gpu && gpu->prefill_qkv_attention_wo;
}

static inline int bn_gpu_backend_can_prefill_qkv_attention_wo_norm_resid(
    const BnGPUBackend *gpu) {
    return gpu && gpu->prefill_qkv_attention_wo_norm_resid;
}

static inline int bn_gpu_backend_can_prefill_dense_layer(
    const BnGPUBackend *gpu) {
    return gpu && gpu->prefill_dense_layer;
}

static inline int bn_gpu_backend_can_prefill_moe_layer(
    const BnGPUBackend *gpu) {
    return gpu && gpu->prefill_moe_layer;
}

static inline int bn_gpu_backend_can_prefill_ssm_layer(
    const BnGPUBackend *gpu) {
    return gpu && gpu->prefill_ssm_layer;
}

static inline int bn_gpu_backend_can_destroy_buffer(
    const BnGPUBackend *gpu) {
    return gpu && gpu->buffer_destroy;
}

static inline int bn_gpu_backend_can_execute(
    const BnGPUBackend *gpu) {
    return gpu && gpu->execute;
}

static inline int bn_gpu_backend_can_init_activations(
    const BnGPUBackend *gpu) {
    return gpu && gpu->init_activations;
}

static inline int bn_gpu_backend_can_free_activations(
    const BnGPUBackend *gpu) {
    return gpu && gpu->free_activations;
}

static inline int bn_gpu_backend_can_write_activation(
    const BnGPUBackend *gpu) {
    return gpu && gpu->write_activation;
}

static inline int bn_gpu_backend_can_read_activation(
    const BnGPUBackend *gpu) {
    return gpu && gpu->read_activation;
}

static inline int bn_gpu_backend_can_argmax_activation(
    const BnGPUBackend *gpu) {
    return gpu && gpu->argmax_activation;
}

static inline int bn_gpu_backend_can_matvec_argmax_activation(
    const BnGPUBackend *gpu) {
    return gpu && gpu->matvec_argmax_activation;
}

static inline int bn_gpu_backend_can_query_memory(
    const BnGPUBackend *gpu) {
    return gpu && gpu->memory_info;
}

static inline int bn_gpu_backend_prepare_cpu_operations(
    const BnGPUBackend *gpu) {
    return gpu && gpu->prepare_cpu_operations
        ? gpu->prepare_cpu_operations(gpu->ctx) : 0;
}

static inline int bn_gpu_backend_has_cap(const BnGPUBackend *gpu,
                                         uint64_t cap) {
    return gpu && ((gpu->caps & cap) != 0);
}

static inline size_t bn_gpu_backend_max_storage_binding_size(
    const BnGPUBackend *gpu) {
    return gpu ? gpu->max_storage_binding_size : 0;
}

static inline int bn_gpu_backend_moe_route_shape_supported(
    const BnGPUBackend *gpu, int n_experts) {
    return gpu && n_experts > 0 &&
           (gpu->max_moe_route_experts <= 0 ||
            n_experts <= gpu->max_moe_route_experts);
}

static inline void *bn_gpu_backend_create_buffer(BnGPUBackend *gpu,
                                                 const void *data,
                                                 size_t size,
                                                 int type,
                                                 int rows,
                                                 int cols) {
    if (!bn_gpu_backend_can_create_buffer(gpu))
        return NULL;
    return gpu->buffer_create(gpu->ctx, data, size, type, rows, cols);
}

static inline void *bn_gpu_backend_create_quant_only_buffer(BnGPUBackend *gpu,
                                                           const void *data,
                                                           size_t size,
                                                           int type,
                                                           int rows,
                                                           int cols) {
    if (!bn_gpu_backend_can_create_quant_only_buffer(gpu))
        return NULL;
    return gpu->buffer_create_quant_only(gpu->ctx, data, size, type, rows,
                                         cols);
}

static inline void *bn_gpu_backend_create_borrowed_buffer(
    BnGPUBackend *gpu,
    const void *data,
    size_t size,
    int type,
    int rows,
    int cols) {
    if (!bn_gpu_backend_can_create_borrowed_buffer(gpu))
        return NULL;
    return gpu->buffer_create_borrowed(gpu->ctx, data, size, type, rows, cols);
}

static inline void *bn_gpu_backend_create_native_matvec_borrowed_buffer(
    BnGPUBackend *gpu, const void *data, size_t size,
    int type, int rows, int cols) {
    if (!bn_gpu_backend_can_create_native_matvec_borrowed_buffer(gpu))
        return NULL;
    return gpu->buffer_create_native_matvec_borrowed(
        gpu->ctx, data, size, type, rows, cols);
}

static inline void *bn_gpu_backend_create_kquant_f32_cache_buffer(
    BnGPUBackend *gpu,
    const void *data,
    size_t size,
    int type,
    int rows,
    int cols) {
    if (!bn_gpu_backend_can_create_kquant_f32_cache_buffer(gpu))
        return NULL;
    return gpu->buffer_create_kquant_f32_cache(gpu->ctx, data, size, type,
                                               rows, cols);
}

static inline void *bn_gpu_backend_create_f16_cache_buffer(BnGPUBackend *gpu,
                                                          const void *data,
                                                          size_t size,
                                                          int type,
                                                          int rows,
                                                          int cols) {
    if (!bn_gpu_backend_can_create_f16_cache_buffer(gpu))
        return NULL;
    return gpu->buffer_create_f16_cache(gpu->ctx, data, size, type, rows,
                                        cols);
}

static inline void *bn_gpu_backend_create_stacked2_buffer(
    BnGPUBackend *gpu,
    const void *first_data,
    size_t first_size,
    const void *second_data,
    size_t second_size,
    int type,
    int rows,
    int cols) {
    if (!bn_gpu_backend_can_create_stacked2_buffer(gpu))
        return NULL;
    return gpu->buffer_create_stacked2(gpu->ctx, first_data, first_size,
                                       second_data, second_size, type, rows,
                                       cols);
}

static inline void *bn_gpu_backend_create_biased_buffer(BnGPUBackend *gpu,
                                                       const void *data,
                                                       size_t size,
                                                       int type,
                                                       int rows,
                                                       int cols,
                                                       const float *bias,
                                                       size_t bias_size) {
    if (!bn_gpu_backend_can_create_biased_buffer(gpu))
        return NULL;
    return gpu->buffer_create_biased(gpu->ctx, data, size, type, rows, cols,
                                     bias, bias_size);
}

static inline void *bn_gpu_backend_create_stacked3_buffer(
    BnGPUBackend *gpu,
    const void *first_data,
    size_t first_size,
    const void *second_data,
    size_t second_size,
    const void *third_data,
    size_t third_size,
    int type,
    int rows,
    int cols) {
    if (!bn_gpu_backend_can_create_stacked3_buffer(gpu))
        return NULL;
    return gpu->buffer_create_stacked3(gpu->ctx, first_data, first_size,
                                       second_data, second_size, third_data,
                                       third_size, type, rows, cols);
}

static inline void *bn_gpu_backend_create_stacked3_biased_buffer(
    BnGPUBackend *gpu,
    const void *first_data,
    size_t first_size,
    const void *second_data,
    size_t second_size,
    const void *third_data,
    size_t third_size,
    int type,
    int rows,
    int cols,
    const float *bias,
    size_t bias_size) {
    if (!bn_gpu_backend_can_create_stacked3_biased_buffer(gpu))
        return NULL;
    return gpu->buffer_create_stacked3_biased(
        gpu->ctx, first_data, first_size, second_data, second_size,
        third_data, third_size, type, rows, cols, bias, bias_size);
}

static inline int bn_gpu_backend_matvec(const BnGPUBackend *gpu,
                                        float *out,
                                        void *buffer,
                                        const float *x,
                                        int rows,
                                        int cols,
                                        int type) {
    if (!bn_gpu_backend_can_matvec(gpu))
        return -1;
    return gpu->matvec(gpu->ctx, out, buffer, x, rows, cols, type);
}

static inline int bn_gpu_backend_matmul(const BnGPUBackend *gpu,
                                        float *out,
                                        void *buffer,
                                        const float *X,
                                        int rows,
                                        int cols,
                                        int n_tokens,
                                        int type) {
    if (!bn_gpu_backend_can_matmul(gpu))
        return -1;
    return gpu->matmul(gpu->ctx, out, buffer, X, rows, cols, n_tokens,
                       type);
}

static inline int bn_gpu_backend_sigmoid_batch(const BnGPUBackend *gpu,
                                                float *out,
                                                const float *X,
                                                int count) {
    if (!bn_gpu_backend_can_sigmoid_batch(gpu))
        return -1;
    return gpu->sigmoid_batch(gpu->ctx, out, X, count);
}

static inline int bn_gpu_backend_hyper_connection_scaled_silu_batch(
    const BnGPUBackend *gpu, float *values, int count, float scale) {
    if (!bn_gpu_backend_can_hyper_connection_scaled_silu_batch(gpu))
        return -1;
    return gpu->hyper_connection_scaled_silu_batch(
        gpu->ctx, values, count, scale);
}

static inline int bn_gpu_backend_hyper_connection_mix_batch(
    const BnGPUBackend *gpu, float *out, const float *norm,
    const float *gates, int n_tokens, int dim, int streams) {
    if (!bn_gpu_backend_can_hyper_connection_mix_batch(gpu))
        return -1;
    return gpu->hyper_connection_mix_batch(
        gpu->ctx, out, norm, gates, n_tokens, dim, streams);
}

static inline int bn_gpu_backend_hyper_connection_prefill_batch(
    const BnGPUBackend *gpu, float *out, float *inject, const float *residual,
    void *norm, void *down, void *up, void *inject_weight,
    int n_tokens, int dim, int streams, int rank, float eps) {
    if (!gpu || !gpu->hyper_connection_prefill_batch) return -1;
    return gpu->hyper_connection_prefill_batch(gpu->ctx, out, inject, residual,
        norm, down, up, inject_weight, n_tokens, dim, streams, rank, eps);
}

static inline int bn_gpu_backend_hyper_connection_combine_batch(
    const BnGPUBackend *gpu, float *residual, const float *block_out,
    const float *inject, int n_tokens, int dim, int streams) {
    if (!bn_gpu_backend_can_hyper_connection_combine_batch(gpu))
        return -1;
    return gpu->hyper_connection_combine_batch(
        gpu->ctx, residual, block_out, inject, n_tokens, dim, streams);
}

static inline int bn_gpu_backend_ssm_delta_gate_batch(
    const BnGPUBackend *gpu, float *out, float *state, const float *qkv,
    const float *z, const float *alpha, const float *beta, void *norm_buf,
    void *dt_bias_buf, void *a_log_buf, int n_tokens, int num_k_heads,
    int head_k_dim, int num_v_heads,
    int head_v_dim, float q_scale, float norm_eps, int sigmoid_gate) {
    if (!bn_gpu_backend_can_ssm_delta_gate_batch(gpu))
        return -1;
    return gpu->ssm_delta_gate_batch(
        gpu->ctx, out, state, qkv, z, alpha, beta, norm_buf, dt_bias_buf,
        a_log_buf, n_tokens,
        num_k_heads, head_k_dim, num_v_heads, head_v_dim, q_scale,
        norm_eps, sigmoid_gate);
}

static inline int bn_gpu_backend_gdn_decode_bf16(const BnGPUBackend *gpu,
    uint16_t *out, float *state, void *a_log, void *dt_bias,
    const uint16_t *X, const uint16_t *BA, int hk, int dk, int hv, int dv,
    float q_scale) {
    if (!bn_gpu_backend_can_gdn_decode_bf16(gpu)) return -1;
    return gpu->gdn_decode_bf16(gpu->ctx, out, state, a_log, dt_bias,
        X, BA, hk, dk, hv, dv, q_scale);
}

static inline int bn_gpu_backend_gdn_prefill_prepare_bf16(const BnGPUBackend *gpu,
    uint16_t *out, float *log_decay, float *beta, void *a_log, void *dt_bias,
    const uint16_t *X, const uint16_t *BA, int n_tokens,
    int hk, int dk, int hv, int dv, float norm_eps, float softplus_threshold) {
    if (!bn_gpu_backend_can_gdn_prefill_prepare_bf16(gpu)) return -1;
    return gpu->gdn_prefill_prepare_bf16(gpu->ctx, out, log_decay, beta,
        a_log, dt_bias, X, BA, n_tokens, hk, dk, hv, dv, norm_eps, softplus_threshold);
}

static inline int bn_gpu_backend_causal_conv_silu_bf16_batch(
    const BnGPUBackend *gpu, uint16_t *out, uint16_t *history, void *weight,
    const uint16_t *X, int n_tokens, int dim, int kernel) {
    if (!bn_gpu_backend_can_causal_conv_silu_bf16_batch(gpu)) return -1;
    return gpu->causal_conv_silu_bf16_batch(
        gpu->ctx, out, history, weight, X, n_tokens, dim, kernel);
}

static inline int bn_gpu_backend_ssm_conv_l2norm_batch(
    const BnGPUBackend *gpu, float *qkv, float *conv_state,
    void *conv_weight_buf, int n_tokens, int qkv_dim, int conv_kernel,
    int num_k_heads, int head_k_dim, float norm_eps, int qk_norm_mode) {
    if (!bn_gpu_backend_can_ssm_conv_l2norm_batch(gpu))
        return -1;
    return gpu->ssm_conv_l2norm_batch(
        gpu->ctx, qkv, conv_state, conv_weight_buf, n_tokens, qkv_dim,
        conv_kernel, num_k_heads, head_k_dim, norm_eps, qk_norm_mode);
}

static inline int bn_gpu_backend_matvec_batch(
    const BnGPUBackend *gpu,
    const BnGPUMatvecOp *ops,
    int n_ops,
    const float *x,
    int x_cols) {
    if (!bn_gpu_backend_can_matvec_batch(gpu))
        return -1;
    return gpu->matvec_batch(gpu->ctx, ops, n_ops, x, x_cols);
}

static inline int bn_gpu_backend_matmul_batch(
    const BnGPUBackend *gpu,
    const BnGPUMatvecOp *ops,
    int n_ops,
    const float *X,
    int n_tokens,
    int x_cols) {
    if (!bn_gpu_backend_can_matmul_batch(gpu))
        return -1;
    return gpu->matmul_batch(gpu->ctx, ops, n_ops, X, n_tokens, x_cols);
}

static inline int bn_gpu_backend_rmsnorm_residual_batch(const BnGPUBackend *gpu,
    float *out, void *norm_buf, const float *X, const float *residual,
    int n_tokens, int dim, float eps) {
    if (!bn_gpu_backend_can_rmsnorm_residual_batch(gpu)) return -1;
    return gpu->rmsnorm_residual_batch(gpu->ctx, out, norm_buf, X, residual,
                                      n_tokens, dim, eps);
}

static inline int bn_gpu_backend_rmsnorm_scaled_batch(const BnGPUBackend *gpu,
    float *out, void *norm, const float *X, int nt, int dim, float eps,
    float post_scale) {
    if (!bn_gpu_backend_can_rmsnorm_scaled_batch(gpu)) return -1;
    return gpu->rmsnorm_scaled_batch(gpu->ctx, out, norm, X, nt, dim, eps,
                                    post_scale);
}

static inline int bn_gpu_backend_rmsnorm_batch(
    const BnGPUBackend *gpu, float *out, void *norm_buf,
    const float *X, int n_tokens, int dim, float eps) {
    if (!bn_gpu_backend_can_rmsnorm_batch(gpu)) return -1;
    return gpu->rmsnorm_batch(gpu->ctx, out, norm_buf, X, n_tokens, dim, eps);
}

static inline int bn_gpu_backend_rmsnorm_bf16_batch(
    const BnGPUBackend *gpu, uint16_t *out, void *norm_buf,
    const uint16_t *X, int n_tokens, int dim, float eps, int weight_offset) {
    if (!bn_gpu_backend_can_rmsnorm_bf16_batch(gpu)) return -1;
    return gpu->rmsnorm_bf16_batch(gpu->ctx, out, norm_buf, X, n_tokens,
                                  dim, eps, weight_offset);
}

static inline int bn_gpu_backend_rmsnorm_grouped_batch(
    const BnGPUBackend *gpu, float *out, void *norm_buf, const float *X,
    int n_tokens, int streams, int dim, float eps) {
    if (!bn_gpu_backend_can_rmsnorm_grouped_batch(gpu)) return -1;
    return gpu->rmsnorm_grouped_batch(gpu->ctx, out, norm_buf, X, n_tokens,
                                      streams, dim, eps);
}

static inline int bn_gpu_backend_dense_ffn(
    const BnGPUBackend *gpu,
    float *out,
    void *gate_buf,
    void *up_buf,
    void *down_buf,
    const float *x,
    int dim,
    int hidden_dim,
    int gate_type,
    int up_type,
    int down_type,
    int act_type) {
    if (!bn_gpu_backend_can_dense_ffn(gpu))
        return -1;
    return gpu->dense_ffn(gpu->ctx, out, gate_buf, up_buf, down_buf, x,
                          dim, hidden_dim, gate_type, up_type, down_type,
                          act_type);
}

static inline int bn_gpu_backend_moe_reduce_batch(const BnGPUBackend *gpu,
    float *out, const float *experts, const float *weights, const float *scales,
    int nt, int k, int dim) {
    if (!bn_gpu_backend_can_moe_reduce_batch(gpu)) return -1;
    return gpu->moe_reduce_batch(gpu->ctx, out, experts, weights, scales, nt, k, dim);
}

static inline int bn_gpu_backend_moe_expert_ffn_batch(
    const BnGPUBackend *gpu, float *out, void *gate, void *up, void *down,
    const float *X, int nt, int dim, int hidden, int gate_type,
    int up_type, int down_type, int act_type, const BnGPUMoEExpertBatchPlan *plan) {
    if (!bn_gpu_backend_can_moe_expert_ffn_batch(gpu)) return -1;
    return gpu->moe_expert_ffn_batch(gpu->ctx, out, gate, up, down, X, nt,
        dim, hidden, gate_type, up_type, down_type, act_type, plan);
}

static inline int bn_gpu_backend_dense_ffn_batch(
    const BnGPUBackend *gpu,
    float *out,
    void *gate_buf,
    void *up_buf,
    void *down_buf,
    const float *X,
    int n_tokens,
    int dim,
    int hidden_dim,
    int gate_type,
    int up_type,
    int down_type,
    int act_type) {
    if (!bn_gpu_backend_can_dense_ffn_batch(gpu))
        return -1;
    return gpu->dense_ffn_batch(gpu->ctx, out, gate_buf, up_buf, down_buf,
                                X, n_tokens, dim, hidden_dim, gate_type,
                                up_type, down_type, act_type);
}

static inline int bn_gpu_backend_dense_ffn_batch_norm(
    const BnGPUBackend *gpu,
    float *out,
    void *gate_buf,
    void *up_buf,
    void *down_buf,
    void *norm_buf,
    const float *X,
    int n_tokens,
    int dim,
    int hidden_dim,
    int gate_type,
    int up_type,
    int down_type,
    int act_type,
    float norm_eps) {
    if (!bn_gpu_backend_can_dense_ffn_batch_norm(gpu))
        return -1;
    return gpu->dense_ffn_batch_norm(
        gpu->ctx, out, gate_buf, up_buf, down_buf, norm_buf, X, n_tokens,
        dim, hidden_dim, gate_type, up_type, down_type, act_type, norm_eps);
}

static inline int bn_gpu_backend_dense_ffn_batch_norm_resid(
    const BnGPUBackend *gpu,
    float *out,
    void *gate_buf,
    void *up_buf,
    void *down_buf,
    void *norm_buf,
    const float *X,
    int n_tokens,
    int dim,
    int hidden_dim,
    int gate_type,
    int up_type,
    int down_type,
    int act_type,
    float norm_eps) {
    if (!bn_gpu_backend_can_dense_ffn_batch_norm_resid(gpu))
        return -1;
    return gpu->dense_ffn_batch_norm_resid(
        gpu->ctx, out, gate_buf, up_buf, down_buf, norm_buf, X, n_tokens,
        dim, hidden_dim, gate_type, up_type, down_type, act_type, norm_eps);
}

static inline int bn_gpu_backend_moe_ffn_batch(
    const BnGPUBackend *gpu,
    float *out,
    const BnGPUMoEPrefillExpert *experts,
    int n_experts,
    const int *expert_offsets,
    const int *expert_counts,
    const int *token_ids,
    const float *weights,
    const float *X,
    int n_tokens,
    int dim,
    int hidden_dim,
    int gate_type,
    int up_type,
    int down_type,
    int act_type,
    void *shared_gate_buf,
    void *shared_up_buf,
    void *shared_down_buf,
    void *shared_gate_weight_buf,
    int shared_hidden_dim,
    int shared_gate_type,
    int shared_up_type,
    int shared_down_type) {
    if (!bn_gpu_backend_can_moe_ffn_batch(gpu))
        return -1;
    return gpu->moe_ffn_batch(
        gpu->ctx, out, experts, n_experts, expert_offsets, expert_counts,
        token_ids, weights, X, n_tokens, dim, hidden_dim, gate_type, up_type,
        down_type, act_type, shared_gate_buf, shared_up_buf, shared_down_buf,
        shared_gate_weight_buf, shared_hidden_dim, shared_gate_type,
        shared_up_type, shared_down_type);
}

static inline int bn_gpu_backend_moe_routed_ffn_batch(
    const BnGPUBackend *gpu,
    float *out,
    void *gate_all_buf,
    void *up_all_buf,
    void *down_all_buf,
    const int *indices,
    const float *weights,
    const float *output_scales,
    const float *X,
    int n_tokens,
    int dim,
    int hidden_dim,
    int n_experts,
    int k,
    int gate_type,
    int up_type,
    int down_type,
    int act_type) {
    if (!bn_gpu_backend_can_moe_routed_ffn_batch(gpu))
        return -1;
    return gpu->moe_routed_ffn_batch(
        gpu->ctx, out, gate_all_buf, up_all_buf, down_all_buf, indices,
        weights, output_scales, X, n_tokens, dim, hidden_dim, n_experts, k, gate_type,
        up_type, down_type, act_type);
}

static inline int bn_gpu_backend_moe_route_batch(
    const BnGPUBackend *gpu,
    int *indices,
    float *weights,
    void *router_buf,
    const float *X,
    int n_tokens,
    int dim,
    int n_experts,
    int k,
    int norm_topk_prob,
    float expert_weights_scale) {
    if (!bn_gpu_backend_can_moe_route_batch(gpu))
        return -1;
    return gpu->moe_route_batch(gpu->ctx, indices, weights, router_buf, X,
                                n_tokens, dim, n_experts, k,
                                norm_topk_prob, expert_weights_scale);
}

static inline int bn_gpu_backend_moe_route_routed_ffn_batch(
    const BnGPUBackend *gpu,
    float *out,
    void *router_buf,
    void *gate_all_buf,
    void *up_all_buf,
    void *down_all_buf,
    const float *X,
    int n_tokens,
    int dim,
    int hidden_dim,
    int n_experts,
    int k,
    int gate_type,
    int up_type,
    int down_type,
    int act_type,
    int norm_topk_prob,
    float expert_weights_scale) {
    if (!bn_gpu_backend_can_moe_route_routed_ffn_batch(gpu))
        return -1;
    return gpu->moe_route_routed_ffn_batch(
        gpu->ctx, out, router_buf, gate_all_buf, up_all_buf, down_all_buf,
        X, n_tokens, dim, hidden_dim, n_experts, k, gate_type, up_type,
        down_type, act_type, norm_topk_prob, expert_weights_scale);
}

static inline int bn_gpu_backend_moe_route_routed_ffn_batch_norm_resid(
    const BnGPUBackend *gpu,
    float *out,
    void *router_buf,
    void *gate_all_buf,
    void *up_all_buf,
    void *down_all_buf,
    void *shared_gate_buf,
    void *shared_up_buf,
    void *shared_down_buf,
    void *shared_gate_weight_buf,
    void *norm_buf,
    const float *X,
    int n_tokens,
    int dim,
    int hidden_dim,
    int n_experts,
    int k,
    int gate_type,
    int up_type,
    int down_type,
    int act_type,
    int shared_hidden_dim,
    int shared_gate_type,
    int shared_up_type,
    int shared_down_type,
    float norm_eps,
    int norm_topk_prob,
    float expert_weights_scale) {
    if (!bn_gpu_backend_can_moe_route_routed_ffn_batch_norm_resid(gpu))
        return -1;
    return gpu->moe_route_routed_ffn_batch_norm_resid(
        gpu->ctx, out, router_buf, gate_all_buf, up_all_buf, down_all_buf,
        shared_gate_buf, shared_up_buf, shared_down_buf,
        shared_gate_weight_buf, norm_buf, X, n_tokens, dim, hidden_dim,
        n_experts, k, gate_type, up_type, down_type, act_type,
        shared_hidden_dim, shared_gate_type, shared_up_type,
        shared_down_type, norm_eps, norm_topk_prob, expert_weights_scale);
}

static inline int bn_gpu_backend_prefill_attention(
    const BnGPUBackend *gpu,
    float *out,
    const float *Q,
    const float *K,
    const float *V,
    int n_tokens,
    int n_heads,
    int n_kv_heads,
    int head_size,
    int kv_mul,
    int kv_dim,
    float attention_scale, int attention_window) {
    if (!bn_gpu_backend_can_prefill_attention(gpu))
        return -1;
    return gpu->prefill_attention(gpu->ctx, out, Q, K, V, n_tokens,
                                  n_heads, n_kv_heads, head_size, kv_mul,
                                  kv_dim, attention_scale, attention_window);
}

static inline int bn_gpu_backend_prefill_attention_prepared_v(
    const BnGPUBackend *gpu, float *out, float *K_out, float *V_out,
    const float *Q, const float *K, const float *V, void *q_norm_buf,
    void *k_norm_buf, const BnGPUAttentionPrefillPlan *plan) {
    if (!bn_gpu_backend_can_prefill_attention_prepared_v(gpu)) return -1;
    return gpu->prefill_attention_prepared_v(gpu->ctx, out, K_out, V_out,
        Q, K, V, q_norm_buf, k_norm_buf, plan);
}

static inline int bn_gpu_backend_prefill_attention_prepared(
    const BnGPUBackend *gpu, float *out, float *K_out,
    const float *Q, const float *K, const float *V,
    void *q_norm_buf, void *k_norm_buf, const BnGPUAttentionPrefillPlan *plan) {
    if (!bn_gpu_backend_can_prefill_attention_prepared(gpu)) return -1;
    return gpu->prefill_attention_prepared(gpu->ctx, out, K_out, Q, K, V,
        q_norm_buf, k_norm_buf, plan);
}

static inline int bn_gpu_backend_can_prefill_qkv_prepared(
    const BnGPUBackend *gpu) {
    return gpu && gpu->prefill_qkv_prepared;
}

static inline int bn_gpu_backend_prefill_qkv_prepared(
    const BnGPUBackend *gpu, float *Q_out, float *K_out, float *V_out,
    const float *Q, const float *K, const float *V,
    void *q_norm_buf, void *k_norm_buf,
    const BnGPUAttentionPrefillPlan *plan) {
    if (!bn_gpu_backend_can_prefill_qkv_prepared(gpu)) return -1;
    return gpu->prefill_qkv_prepared(gpu->ctx, Q_out, K_out, V_out,
        Q, K, V, q_norm_buf, k_norm_buf, plan);
}

static inline int bn_gpu_backend_decode_attention_scores_prepared(
    const BnGPUBackend *gpu, float *scores, float *Q_out, float *K_out,
    const float *Q, const float *K, const float *V,
    void *q_norm_buf, void *k_norm_buf,
    const BnGPUAttentionPrefillPlan *plan) {
    if (!bn_gpu_backend_can_decode_attention_scores_prepared(gpu)) return -1;
    return gpu->decode_attention_scores_prepared(
        gpu->ctx, scores, Q_out, K_out, Q, K, V,
        q_norm_buf, k_norm_buf, plan);
}

static inline int bn_gpu_backend_prefill_attention_wo(
    const BnGPUBackend *gpu,
    float *out,
    void *wo_buf,
    const float *Q,
    const float *K,
    const float *V,
    int n_tokens,
    int n_heads,
    int n_kv_heads,
    int head_size,
    int kv_mul,
    int kv_dim,
    int wo_rows,
    int wo_cols,
    int wo_type,
    float attention_scale, int attention_window) {
    if (!bn_gpu_backend_can_prefill_attention_wo(gpu))
        return -1;
    return gpu->prefill_attention_wo(gpu->ctx, out, wo_buf, Q, K, V,
                                     n_tokens, n_heads, n_kv_heads,
                                     head_size, kv_mul, kv_dim, wo_rows,
                                     wo_cols, wo_type, attention_scale, attention_window);
}

static inline int bn_gpu_backend_prefill_qkv_attention_wo(
    const BnGPUBackend *gpu,
    float *out,
    void *qk_buf,
    void *wv_buf,
    void *wo_buf,
    void *q_norm_buf,
    void *k_norm_buf,
    const float *X,
    float *K_out,
    float *V_out,
    int n_tokens,
    int dim,
    int n_heads,
    int n_kv_heads,
    int head_size,
    int kv_mul,
    int kv_dim,
    int qk_rows,
    int qk_type,
    int wv_rows,
    int wv_type,
    int wo_rows,
    int wo_cols,
    int wo_type,
    int qk_norm_per_head,
    float norm_eps,
    int pos0,
    int rope_dims,
    float attention_scale, int attention_window) {
    if (!bn_gpu_backend_can_prefill_qkv_attention_wo(gpu))
        return -1;
    return gpu->prefill_qkv_attention_wo(
        gpu->ctx, out, qk_buf, wv_buf, wo_buf, q_norm_buf, k_norm_buf, X,
        K_out, V_out, n_tokens, dim, n_heads, n_kv_heads, head_size, kv_mul,
        kv_dim, qk_rows, qk_type, wv_rows, wv_type, wo_rows, wo_cols,
        wo_type, qk_norm_per_head, norm_eps, pos0, rope_dims,
        attention_scale, attention_window);
}

static inline int bn_gpu_backend_prefill_qkv_attention_wo_norm_resid(
    const BnGPUBackend *gpu,
    float *out,
    void *qk_buf,
    void *wv_buf,
    void *wo_buf,
    void *attn_norm_buf,
    void *q_norm_buf,
    void *k_norm_buf,
    const float *X,
    float *K_out,
    float *V_out,
    int n_tokens,
    int dim,
    int n_heads,
    int n_kv_heads,
    int head_size,
    int kv_mul,
    int kv_dim,
    int qk_rows,
    int qk_type,
    int wv_rows,
    int wv_type,
    int wo_rows,
    int wo_cols,
    int wo_type,
    int qk_norm_per_head,
    float norm_eps,
    int pos0,
    int rope_dims,
    float attention_scale, int attention_window) {
    if (!bn_gpu_backend_can_prefill_qkv_attention_wo_norm_resid(gpu))
        return -1;
    return gpu->prefill_qkv_attention_wo_norm_resid(
        gpu->ctx, out, qk_buf, wv_buf, wo_buf, attn_norm_buf, q_norm_buf,
        k_norm_buf, X, K_out, V_out, n_tokens, dim, n_heads, n_kv_heads,
        head_size, kv_mul, kv_dim, qk_rows, qk_type, wv_rows, wv_type,
        wo_rows, wo_cols, wo_type, qk_norm_per_head, norm_eps, pos0,
        rope_dims, attention_scale, attention_window);
}

static inline int bn_gpu_backend_prefill_dense_layer(
    const BnGPUBackend *gpu,
    float *out,
    void *qk_buf,
    void *wv_buf,
    void *wo_buf,
    void *gate_buf,
    void *up_buf,
    void *down_buf,
    void *attn_norm_buf,
    void *ffn_norm_buf,
    void *attn_post_norm_buf,
    void *ffn_post_norm_buf,
    void *q_norm_buf,
    void *k_norm_buf,
    void *q_bias_buf,
    void *k_bias_buf,
    void *v_bias_buf,
    const float *X,
    float *K_out,
    float *V_out,
    int n_tokens,
    int dim,
    int hidden_dim,
    int n_heads,
    int n_kv_heads,
    int head_size,
    int kv_mul,
    int kv_dim,
    int qk_rows,
    int qk_type,
    int wv_rows,
    int wv_type,
    int wo_rows,
    int wo_cols,
    int wo_type,
    int gate_type,
    int up_type,
    int down_type,
    int act_type,
    int qk_norm_per_head,
    int normalize_v,
    float norm_eps,
    int pos0,
    int rope_dims,
    size_t rope_freq_offset,
    uint32_t kv_cache_off,
    int kv_cache_stride,
    float attention_scale,
    float layer_output_scale, int attention_window,
    int final_ffn_last_row_only) {
    if (!bn_gpu_backend_can_prefill_dense_layer(gpu))
        return -1;
    return gpu->prefill_dense_layer(
        gpu->ctx, out, qk_buf, wv_buf, wo_buf, gate_buf, up_buf, down_buf,
        attn_norm_buf, ffn_norm_buf, attn_post_norm_buf,
        ffn_post_norm_buf, q_norm_buf, k_norm_buf, q_bias_buf,
        k_bias_buf, v_bias_buf, X, K_out, V_out, n_tokens, dim, hidden_dim,
        n_heads, n_kv_heads, head_size, kv_mul, kv_dim, qk_rows, qk_type,
        wv_rows, wv_type, wo_rows, wo_cols, wo_type, gate_type, up_type,
        down_type, act_type, qk_norm_per_head, normalize_v, norm_eps,
        pos0, rope_dims,
        rope_freq_offset, kv_cache_off, kv_cache_stride, attention_scale,
        layer_output_scale, attention_window, final_ffn_last_row_only);
}

static inline int bn_gpu_backend_prefill_moe_layer(
    const BnGPUBackend *gpu,
    float *out,
    void *qk_buf,
    void *wv_buf,
    void *wo_buf,
    void *router_buf,
    void *gate_all_buf,
    void *up_all_buf,
    void *down_all_buf,
    void *shared_gate_buf,
    void *shared_up_buf,
    void *shared_down_buf,
    void *shared_gate_weight_buf,
    void *attn_norm_buf,
    void *ffn_norm_buf,
    void *q_norm_buf,
    void *k_norm_buf,
    void *q_bias_buf,
    void *k_bias_buf,
    void *v_bias_buf,
    const float *X,
    float *K_out,
    float *V_out,
    int n_tokens,
    int dim,
    int moe_hidden_dim,
    int n_experts,
    int experts_active,
    int n_heads,
    int n_kv_heads,
    int head_size,
    int kv_mul,
    int kv_dim,
    int qk_rows,
    int qk_type,
    int wv_rows,
    int wv_type,
    int wo_rows,
    int wo_cols,
    int wo_type,
    int gate_type,
    int up_type,
    int down_type,
    int act_type,
    int shared_hidden_dim,
    int shared_gate_type,
    int shared_up_type,
    int shared_down_type,
    int qk_norm_per_head,
    float norm_eps,
    int pos0,
    int rope_dims,
    uint32_t kv_cache_off,
    int kv_cache_stride,
    float attention_scale,
    int norm_topk_prob,
    float expert_weights_scale, int attention_window) {
    if (!bn_gpu_backend_can_prefill_moe_layer(gpu))
        return -1;
    return gpu->prefill_moe_layer(
        gpu->ctx, out, qk_buf, wv_buf, wo_buf, router_buf, gate_all_buf,
        up_all_buf, down_all_buf, shared_gate_buf, shared_up_buf,
        shared_down_buf, shared_gate_weight_buf, attn_norm_buf, ffn_norm_buf,
        q_norm_buf, k_norm_buf, q_bias_buf, k_bias_buf, v_bias_buf,
        X, K_out, V_out, n_tokens, dim, moe_hidden_dim, n_experts,
        experts_active, n_heads, n_kv_heads, head_size, kv_mul, kv_dim,
        qk_rows, qk_type, wv_rows, wv_type, wo_rows, wo_cols, wo_type,
        gate_type, up_type, down_type, act_type, shared_hidden_dim,
        shared_gate_type, shared_up_type, shared_down_type, qk_norm_per_head,
        norm_eps, pos0, rope_dims, kv_cache_off, kv_cache_stride,
        attention_scale, norm_topk_prob, expert_weights_scale, attention_window);
}

static inline int bn_gpu_backend_prefill_ssm_layer(
    const BnGPUBackend *gpu,
    float *out,
    void *wqkv_buf,
    void *wz_buf,
    void *alpha_buf,
    void *beta_buf,
    void *qkvz_stacked_buf,
    void *ab_stacked_buf,
    void *ssm_out_buf,
    void *attn_norm_buf,
    void *conv1d_buf,
    void *dt_bias_buf,
    void *a_log_buf,
    void *ssm_norm_buf,
    void *ffn_gate_buf,
    void *ffn_up_buf,
    void *ffn_down_buf,
    void *ffn_norm_buf,
    const float *X,
    int n_tokens,
    int dim,
    int qkv_dim,
    int inner_dim,
    int num_k_heads,
    int head_k_dim,
    int num_v_heads,
    int head_v_dim,
    int conv_kernel,
    int ssm_idx,
    int wqkv_type,
    int wz_type,
    int alpha_type,
    int beta_type,
    int out_type,
    int hidden_dim,
    int ffn_gate_type,
    int ffn_up_type,
    int ffn_down_type,
    int act_type,
    int sigmoid_gate,
    float norm_eps,
    int qk_norm_mode,
    int *did_ffn) {
    if (!bn_gpu_backend_can_prefill_ssm_layer(gpu))
        return -1;
    return gpu->prefill_ssm_layer(
        gpu->ctx, out, wqkv_buf, wz_buf, alpha_buf, beta_buf,
        qkvz_stacked_buf, ab_stacked_buf, ssm_out_buf, attn_norm_buf,
        conv1d_buf, dt_bias_buf, a_log_buf, ssm_norm_buf, ffn_gate_buf,
        ffn_up_buf, ffn_down_buf, ffn_norm_buf, X, n_tokens, dim, qkv_dim,
        inner_dim, num_k_heads, head_k_dim, num_v_heads, head_v_dim,
        conv_kernel, ssm_idx, wqkv_type, wz_type, alpha_type, beta_type,
        out_type, hidden_dim, ffn_gate_type, ffn_up_type, ffn_down_type,
        act_type, sigmoid_gate, norm_eps, qk_norm_mode, did_ffn);
}

static inline int bn_gpu_backend_execute(const BnGPUBackend *gpu,
                                         const void *ops,
                                         int n_ops,
                                         int readback_buf,
                                         float *readback,
                                         int readback_count) {
    if (!bn_gpu_backend_can_execute(gpu))
        return -1;
    return gpu->execute(gpu->ctx, ops, n_ops, readback_buf, readback,
                        readback_count);
}

static inline int bn_gpu_backend_init_activations(const BnGPUBackend *gpu,
                                                  const BnGPUActivationPlan *plan) {
    if (!bn_gpu_backend_can_init_activations(gpu))
        return -1;
    return gpu->init_activations(gpu->ctx, plan);
}

static inline int bn_gpu_backend_reset_activations(const BnGPUBackend *gpu) {
    return gpu && gpu->reset_activations
        ? gpu->reset_activations(gpu->ctx) : 0;
}

static inline void bn_gpu_backend_free_activations(const BnGPUBackend *gpu) {
    if (bn_gpu_backend_can_free_activations(gpu))
        gpu->free_activations(gpu->ctx);
}

static inline int bn_gpu_backend_write_activation(const BnGPUBackend *gpu,
                                                  int buf_idx,
                                                  const void *data,
                                                  size_t size,
                                                  size_t offset) {
    if (!bn_gpu_backend_can_write_activation(gpu))
        return -1;
    return gpu->write_activation(gpu->ctx, buf_idx, data, size, offset);
}

static inline int bn_gpu_backend_read_activation(const BnGPUBackend *gpu,
                                                 int buf_idx,
                                                 void *out,
                                                 size_t size,
                                                 size_t offset) {
    if (!bn_gpu_backend_can_read_activation(gpu))
        return -1;
    return gpu->read_activation(gpu->ctx, buf_idx, out, size, offset);
}

static inline int bn_gpu_backend_argmax_activation(
    const BnGPUBackend *gpu,
    int buf_idx,
    int n,
    const int *penalty_tokens,
    int n_penalty_tokens,
    float repeat_penalty,
    int *out_token) {
    if (!bn_gpu_backend_can_argmax_activation(gpu))
        return -1;
    return gpu->argmax_activation(gpu->ctx, buf_idx, n, penalty_tokens,
                                  n_penalty_tokens, repeat_penalty,
                                  out_token);
}

static inline int bn_gpu_backend_matvec_argmax_activation(
    const BnGPUBackend *gpu,
    void *W_buf,
    int type,
    int rows,
    int cols,
    int buf_idx,
    const int *penalty_tokens,
    int n_penalty_tokens,
    float repeat_penalty,
    int *out_token) {
    if (!bn_gpu_backend_can_matvec_argmax_activation(gpu))
        return -1;
    return gpu->matvec_argmax_activation(
        gpu->ctx, W_buf, type, rows, cols, buf_idx, penalty_tokens,
        n_penalty_tokens, repeat_penalty, out_token);
}

static inline int bn_gpu_backend_query_memory(const BnGPUBackend *gpu,
                                              size_t *free_bytes,
                                              size_t *total_bytes) {
    if (!bn_gpu_backend_can_query_memory(gpu))
        return -1;
    return gpu->memory_info(gpu->ctx, free_bytes, total_bytes);
}

static inline void bn_gpu_backend_destroy_buffer(BnGPUBackend *gpu,
                                                 void *buffer) {
    if (bn_gpu_backend_can_destroy_buffer(gpu) && buffer)
        gpu->buffer_destroy(gpu->ctx, buffer);
}

// Backend capability bits
#define BN_GPU_CAP_FLASH_ATTN  (1u << 0)  // fused flash attention shader available
#define BN_GPU_CAP_NATIVE_QUANT_MATVEC_SPLIT (1u << 1) // native-quant split matvec shader available
#define BN_GPU_CAP_DEINTERLEAVED_KQUANT_MATVEC_SPLIT (1u << 2) // deinterleaved K-quant split matvec shader available
#define BN_GPU_CAP_LOWBIT_BLOCK32_MATVEC_SPLIT (1u << 3) // low-bit block32 split matvec shader available
#define BN_GPU_CAP_LOWBIT_BLOCK32_FUSED_GATEUP_SILU (1u << 4) // low-bit block32 fused gate/up SiLU shader available
#define BN_GPU_CAP_ASYMMETRIC_KQUANT_MATVEC_SPLIT (1u << 5) // asymmetric K-quant split matvec shader available
#define BN_GPU_CAP_MIDBIT_BLOCK32_FUSED_GATEUP_SILU (1u << 6) // mid-bit block32 fused gate/up SiLU shader available
#define BN_GPU_CAP_MIDBIT_BLOCK32_MATVEC_SPLIT (1u << 7) // mid-bit block32 split matvec shader available
#define BN_GPU_CAP_NATIVE_QUANT_FUSED_GATEUP_SILU (1u << 8) // native-quant fused gate/up SiLU shader available
#define BN_GPU_CAP_DEINTERLEAVED_KQUANT_FUSED_GATEUP_SILU (1u << 9) // deinterleaved K-quant fused gate/up SiLU shader available
#define BN_GPU_CAP_LAYERWISE_ROPE (1u << 10) // backend can vary RoPE frequency policy per layer
#define BN_GPU_CAP_MOE_ROUTED_FFN (1u << 11) // resident decode route + routed FFN graph ops
#define BN_GPU_CAP_BORROWED_WEIGHT_BUFFERS (1u << 12) // weight handles may borrow immutable host-mapped storage
#define BN_GPU_CAP_MOE_ROUTED_KQUANT_DOWN_CACHE (1u << 13) // routed K-quant down cache strategy available
#define BN_GPU_CAP_MOE_ROUTED_NATIVE_QUANT (1u << 14) // native-quant routed FFN strategy available
#define BN_GPU_CAP_DECODE_GRAPH_CACHE (1u << 15) // lowered decode op streams may be replayed
#define BN_GPU_CAP_LARGE_GRAPH_NATIVE (1u << 16) // large dense/hybrid graph execution available
#define BN_GPU_CAP_SSM_GRAPH (1u << 17) // state-space graph operations available
#define BN_GPU_CAP_PER_LAYER_INPUT_GRAPH (1u << 18) // per-layer input adapter graph composition available
#define BN_GPU_CAP_REFERENCE_ATTENTION (1u << 19) // shared-value attention preserves reference trajectory semantics
#define BN_GPU_CAP_REFERENCE_ATTENTION_FALLBACK (1u << 20) // backend supports reference-attention CPU handoff
#define BN_GPU_CAP_PREPARED_NATIVE_QUANT (1u << 21) // backend supports prepared weights with block-quantized activations
#define BN_GPU_CAP_MOE_EXPERT_GRAPH (1u << 22) // CPU-routed selected-expert graph execution available
#define BN_GPU_CAP_HYBRID_SSM_MOE_GRAPH (1u << 23) // combined SSM + routed-MoE graph is coherent
#define BN_GPU_CAP_MOE_ROUTED_LOWBIT_BLOCK32 (1u << 24) // routed block32 low-bit FFN kernels available
#define BN_GPU_CAP_DENSE_RESIDUAL_LOWBIT_BLOCK32 (1u << 25) // dense residual low-bit FFN graph available
#define BN_GPU_CAP_REFERENCE_RECURRENT (1u << 26) // recurrent graph preserves reference trajectory semantics
#define BN_GPU_CAP_PREPARED_NATIVE_QUANT_ATTENTION (1u << 27) // prepared native-quant attention projections are coherent
#define BN_GPU_CAP_PREPARED_NATIVE_QUANT_PER_LAYER_FFN (1u << 28) // prepared native-quant FFN gate/up is coherent with per-layer input composition
#define BN_GPU_CAP_PREPARED_NATIVE_QUANT_PER_LAYER_FFN_DOWN (1u << 29) // prepared native-quant FFN down is coherent with per-layer input composition
#define BN_GPU_CAP_REFERENCE_ATTENTION_NATIVE_GRAPH (1u << 30) // ordinary attention graph preserves reference trajectory semantics
#define BN_GPU_CAP_REFERENCE_ATTENTION_TOKEN_FALLBACK (1u << 31) // reference attention requires whole-token CPU orchestration
#define BN_GPU_CAP_HYPER_CONNECTION_GRAPH (UINT64_C(1) << 32) // hyper-connection mix/combine graph operations available
#define BN_GPU_CAP_MOE_ROUTED_MIDBIT_BLOCK32_DOWN (UINT64_C(1) << 33) // routed mid-bit block32 down projection available
#define BN_GPU_CAP_MOE_ROUTED_MIXED_QUANT (UINT64_C(1) << 34) // mixed routed gate/up and down quant formats available
#define BN_GPU_CAP_MOE_ROUTED_MIDBIT_KQUANT_GATEUP (UINT64_C(1) << 35) // routed mid-bit K-quant gate/up projections available
#define BN_GPU_CAP_REFERENCE_RECURRENT_PREFILL (UINT64_C(1) << 36) // batched recurrent prefill preserves reference trajectory semantics
#define BN_GPU_CAP_KQUANT_BLOCK32_LOGITS (UINT64_C(1) << 37) // K-quant logits use backend-native block32 activation quantization
#define BN_GPU_CAP_FP32_GELU (UINT64_C(1) << 38) // GELU retains FP32 input/output, including CPU-executed steps in GPU requests
#define BN_GPU_CAP_MOE_COMBINED_PREFILL_DEFAULT (UINT64_C(1) << 39) // try combined prefill by default; backend checks format eligibility
#define BN_GPU_CAP_KQUANT_BLOCK32_RECURRENT (UINT64_C(1) << 40) // recurrent K-quant projections use backend-native block32 activations

/* RMSNorm graph supports a separately rounded scalar before its weight.
 * Scaled-router uploads retain raw weights when this capability is set. */
#define BN_GPU_CAP_RMSNORM_SEPARATE_SCALE (UINT64_C(1) << 41)
/* WEIGHTED_ADD can round an output scale before route weighting and
 * explicitly choose fused or separate accumulation. */
#define BN_GPU_CAP_WEIGHTED_ADD_SEPARATE_SCALE (UINT64_C(1) << 42)
#define BN_GPU_CAP_PREFILL_LOGICAL_PROJECTION_ROWS (UINT64_C(1) << 43)
#define BN_GPU_CAP_MOE_ROUTED_ORDERED_KQUANT (UINT64_C(1) << 45)
#define BN_GPU_CAP_MOE_ROUTED_E8M0 (UINT64_C(1) << 44) // routed E8M0 gate/up with ordered K-quant down
// Prepared prefix attention: FP16 KV, head256/GQA16, 33..2048 current+prior
// keys (at least two current queries). Backend validates cache bounds.
#define BN_GPU_CAP_PREFILL_PREFIX_KV (UINT64_C(1) << 46)
/* RMSNorm graph can reproduce the model policy's ordered FP32-square/double-sum
 * contract rather than using the backend's parallel reduction. */
#define BN_GPU_CAP_REFERENCE_RMSNORM_ORDER (UINT64_C(1) << 47)

static inline int bn_gpu_backend_prefill_uses_logical_rows(const BnGPUBackend *gpu) {
    return gpu && (gpu->caps & BN_GPU_CAP_PREFILL_LOGICAL_PROJECTION_ROWS) != 0;
}

/* Lazily attach one resident request to a BnBackendSession. The backend must
 * outlive operation submission; request cleanup is independently owned. */
struct BnBackendSession;
void *bn_gpu_backend_resident_request(const BnGPUBackend *gpu,
                                      struct BnBackendSession *session);

static inline int bn_gpu_backend_can_resident_request(const BnGPUBackend *gpu) {
    return gpu && gpu->resident_request_create && gpu->resident_request_destroy &&
        gpu->resident_alloc && gpu->resident_write && gpu->resident_read &&
        gpu->resident_synchronize;
}

static inline int bn_gpu_backend_can_gdn_prefill_bf16_resident(const BnGPUBackend *gpu) {
    return bn_gpu_backend_can_resident_request(gpu) &&
        gpu->gdn_prefill_bf16_reserve && gpu->gdn_prefill_bf16_resident;
}

static inline BnGPUResidentView bn_gpu_resident_view_slice(
    BnGPUResidentView view, size_t offset, size_t count) {
    BnGPUResidentView invalid = {0};
    if (!view.request || !count || offset > view.count ||
        count > view.count - offset || offset > SIZE_MAX - view.offset)
        return invalid;
    view.offset += offset;
    view.count = count;
    return view;
}

#ifdef __cplusplus
}
#endif

#endif // BN_GPU_BACKEND_H
