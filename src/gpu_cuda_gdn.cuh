#ifndef BN_GPU_CUDA_GDN_CUH
#define BN_GPU_CUDA_GDN_CUH

/* Prepared BF16 GDN chunk64 profile: HK16, HV48, K/V128.
 * Preserve explicit FMA/reduction order and BF16 rounding boundaries.
 * KKT consumes raw F32 bits in even/odd feature order; inverse profile4 uses
 * its pinned reduction tree. State update starts MMA at zero then applies
 * fmaf(previous, decay, update). Output gates/scores round before value MMA.
 * This private implementation is included only by gpu_cuda.cu.
 */
namespace bn_cuda_gdn {

static __global__ void scan(float *out, const float *in, int nt, int hv) {
    int head = blockIdx.y, lane = threadIdx.x;
    int t = blockIdx.x * 64 + lane;
    float a = t < nt ? in[(size_t)t * hv + head] : 0,
          b = t + 32 < nt ? in[(size_t)(t + 32) * hv + head] : 0;
    for (int delta = 1; delta < 32; delta *= 2) {
        float x = __shfl_up_sync(0xffffffff, a, delta), y = __shfl_up_sync(0xffffffff, b, delta);
        if (lane >= delta) {
            a = __fadd_rn(a, x);
            b = __fadd_rn(b, y);
        }
    }
    b = __fadd_rn(b, __shfl_sync(0xffffffff, a, 31));
    if (t < nt)
        out[(size_t)t * hv + head] = a;
    if (t + 32 < nt)
        out[(size_t)(t + 32) * hv + head] = b;
}

static __device__ float ex(float x) {
    float r, t = __fmul_rn(x, 0x1.715476p+0f);
    asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(t));
    return r;
}

static __device__ float dot16(const float *a, const float *b, int row, int col, float initial = 0) {
    float s = initial;
    for (int k = 0; k < 16; k++)
        s = fmaf(a[row * 16 + k], b[k * 16 + col], s);
    return s;
}

static __global__ void inverse(float *out, const float *A, int nt, int profile) {
    int row = threadIdx.x / 16, col = threadIdx.x % 16, tid = threadIdx.x, chunk = blockIdx.x,
        head = blockIdx.y;
    __shared__ float d[4][256], a[6][256], v[6][256], t0[256], t1[256], t2[256], work[256];
    for (int block = 0; block < 4; block++) {
        int globalrow = chunk * 64 + block * 16 + row;
        d[block][tid] = row > col && globalrow < nt
                            ? -A[((size_t)globalrow * 48 + head) * 64 + block * 16 + col]
                            : -0.f;
    }
    __syncthreads();
    for (int block = 0; block < 4; block++) {
        for (int i = 2; i < 16 && chunk * 64 + block * 16 + i < nt; i++) {
            int gr = chunk * 64 + block * 16 + i;
            float b = -A[((size_t)gr * 48 + head) * 64 + block * 16 + row];
            work[tid] = __fmul_rn(b, d[block][tid]);
            __syncthreads();
            int first = 8 / profile;
            float partner = work[(row ^ first) * 16 + col];
            __syncthreads();
            work[tid] = fmaf(b, d[block][tid], partner);
            __syncthreads();
            for (int offset = first / 2; offset; offset >>= 1) {
                float x = work[tid], y = work[(row ^ offset) * 16 + col];
                __syncthreads();
                work[tid] = __fadd_rn(x, y);
                __syncthreads();
            }
            if (row == i)
                d[block][tid] = __fadd_rn(
                    -A[((size_t)gr * 48 + head) * 64 + block * 16 + col],
                    (profile == 2 ? __fadd_rn(work[7 * 16 + col], work[15 * 16 + col])
                     : profile == 4
                         ? __fadd_rn(__fadd_rn(work[1 * 16 + col], work[9 * 16 + col]),
                                     __fadd_rn(work[5 * 16 + col], work[13 * 16 + col]))
                         : __fadd_rn(
                               __fadd_rn(__fadd_rn(work[0 * 16 + col], work[8 * 16 + col]),
                                         __fadd_rn(work[4 * 16 + col], work[12 * 16 + col])),
                               __fadd_rn(__fadd_rn(work[2 * 16 + col], work[10 * 16 + col]),
                                         __fadd_rn(work[6 * 16 + col], work[14 * 16 + col])))));
            __syncthreads();
        }
        d[block][tid] = __fadd_rn(d[block][tid], row == col ? 1.f : 0.f);
    }
    __syncthreads();
    int br[6] = {1, 2, 2, 3, 3, 3}, bc[6] = {0, 0, 1, 0, 1, 2};
    for (int j = 0; j < 6; j++) {
        int gr = chunk * 64 + br[j] * 16 + row;
        a[j][tid] = gr < nt ? A[((size_t)gr * 48 + head) * 64 + bc[j] * 16 + col] : 0;
    }
    __syncthreads();
    // I21, I32, I43: two IEEE products each.
    t0[tid] = dot16(d[1], a[0], row, col);
    t1[tid] = dot16(d[2], a[2], row, col);
    t2[tid] = dot16(d[3], a[5], row, col);
    __syncthreads();
    v[0][tid] = -dot16(t0, d[0], row, col);
    v[2][tid] = -dot16(t1, d[1], row, col);
    v[5][tid] = -dot16(t2, d[2], row, col);
    __syncthreads();
    // I31 and I42.
    t0[tid] = dot16(a[2], v[0], row, col, dot16(a[1], d[0], row, col));
    t1[tid] = dot16(a[5], v[2], row, col, dot16(a[4], d[1], row, col));
    __syncthreads();
    v[1][tid] = -dot16(d[2], t0, row, col);
    v[4][tid] = -dot16(d[3], t1, row, col);
    __syncthreads();
    // I41.
    t0[tid] = dot16(a[5], v[1], row, col, dot16(a[4], v[0], row, col, dot16(a[3], d[0], row, col)));
    __syncthreads();
    v[3][tid] = -dot16(d[3], t0, row, col);
    __syncthreads();
    for (int rb = 0; rb < 4; rb++) {
        int gr = chunk * 64 + rb * 16 + row;
        if (gr < nt)
            for (int cb = 0; cb < 4; cb++) {
                float x = 0;
                if (rb == cb)
                    x = d[rb][tid];
                for (int j = 0; j < 6; j++)
                    if (br[j] == rb && bc[j] == cb)
                        x = v[j][tid];
                out[((size_t)gr * 48 + head) * 64 + cb * 16 + col] = x;
            }
    }
}

static __device__ unsigned pair(__nv_bfloat16 a, __nv_bfloat16 b) {
    return (unsigned)__bfloat16_as_ushort(a) | ((unsigned)__bfloat16_as_ushort(b) << 16);
}

static __global__ void cast_inverse(__nv_bfloat16 *out, const float *x, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = __float2bfloat16_rn(x[i]);
}

template <bool Tr>
__device__ void sweep_ld4(const __nv_bfloat16 *x, int pitch, int row, int col, int lane,
                          unsigned (&a)[4]) {
    int r = row + (Tr ? (lane % 8 + (lane / 16) * 8) : lane % 16),
        c = col + (Tr ? ((lane / 8) % 2) * 8 : (lane / 16) * 8);
    unsigned addr = (unsigned)__cvta_generic_to_shared(x + r * pitch + c);
    if (Tr)
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];"
                     : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                     : "r"(addr)
                     : "memory");
    else
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
                     : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                     : "r"(addr)
                     : "memory");
}

template <bool Tr>
__device__ void sweep_ld2(const __nv_bfloat16 *x, int pitch, int row, int col, int lane,
                          unsigned (&b)[2]) {
    int r = row + lane % 8 + (Tr ? ((lane / 8) % 2) * 8 : 0),
        c = col + (Tr ? 0 : ((lane / 8) % 2) * 8);
    unsigned addr = (unsigned)__cvta_generic_to_shared(x + r * pitch + c);
    if (Tr)
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];"
                     : "=r"(b[0]), "=r"(b[1])
                     : "r"(addr)
                     : "memory");
    else
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2];"
                     : "=r"(b[0]), "=r"(b[1])
                     : "r"(addr)
                     : "memory");
}

static __global__ void fused_sweep(float *state, __nv_bfloat16 *h, __nv_bfloat16 *vnew,
                                   const __nv_bfloat16 *__restrict__ w,
                                   const __nv_bfloat16 *__restrict__ u,
                                   const __nv_bfloat16 *__restrict__ k, const float *__restrict__ g,
                                   const float *initial, int nt) {
    int tid = threadIdx.x, lane = tid % 32, warp = tid / 32, head = blockIdx.x,
        vbase = blockIdx.y * 32;
    __shared__ __align__(16) __nv_bfloat16 hs[32 * 136], vs[64 * 40], ks[64 * 136], ws[64 * 136];
    __shared__ float gates[64], decay;
    float history[8][4];
#pragma unroll
    for (int kt = 0; kt < 8; kt++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int kr = kt * 16 + lane / 4 + j / 2 * 8, vc = vbase + warp * 8 + 2 * (lane % 4) + j % 2;
            history[kt][j] = initial[((size_t)head * 128 + vc) * 128 + kr];
        }
    }
    for (int chunk = 0; chunk < (nt + 63) / 64; chunk++) {
        for (int i = tid; i < 64 * 16; i += 128) {
            int t = i / 16, c = (i % 16) * 8, rr = chunk * 64 + t;
            uint4 kv = make_uint4(0, 0, 0, 0), wv = kv;
            if (rr < nt) {
                kv = *(const uint4 *)(k + ((size_t)rr * 16 + head / 3) * 128 + c);
                wv = *(const uint4 *)(w + ((size_t)rr * 48 + head) * 128 + c);
            }
            *(uint4 *)(ks + t * 136 + c) = kv;
            *(uint4 *)(ws + t * 136 + c) = wv;
        }

#pragma unroll
        for (int kt = 0; kt < 8; kt++) {
#pragma unroll
            for (int j = 0; j < 4; j++) {
                int kr = kt * 16 + lane / 4 + j / 2 * 8, vc = warp * 8 + 2 * (lane % 4) + j % 2;
                __nv_bfloat16 value = __float2bfloat16_rn(history[kt][j]);
                hs[vc * 136 + kr] = value;
                h[(((size_t)chunk * 48 + head) * 128 + vbase + vc) * 128 + kr] = value;
            }
        }
        int last = min(chunk * 64 + 63, nt - 1);
        float glast = g[last * 48 + head];
        if (tid == 0)
            decay = ex(glast);
        if (tid < 64) {
            int t = chunk * 64 + tid;
            gates[tid] = t < nt ? ex(__fsub_rn(glast, g[t * 48 + head])) : 0;
        }
        __syncthreads();
        float dr[4][4] = {};
        for (int base = 0; base < 128; base += 16) {
            unsigned bb[2];
            sweep_ld2<false>(hs, 136, warp * 8, base, lane, bb);
            unsigned b0 = bb[0], b1 = bb[1];
#pragma unroll
            for (int mt = 0; mt < 4; mt++) {
                unsigned aa[4];
                sweep_ld4<false>(ws, 136, mt * 16, base, lane, aa);
                unsigned a0 = aa[0], a1 = aa[1], a2 = aa[2], a3 = aa[3];
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                             "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                             : "+f"(dr[mt][0]), "+f"(dr[mt][1]), "+f"(dr[mt][2]), "+f"(dr[mt][3])
                             : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
        }
#pragma unroll
        for (int mt = 0; mt < 4; mt++) {
#pragma unroll
            for (int j = 0; j < 4; j++) {
                int rr = chunk * 64 + mt * 16 + lane / 4 + j / 2 * 8,
                    vc = warp * 8 + 2 * (lane % 4) + j % 2;
                float raw = 0;
                if (rr < nt) {
                    size_t off = ((size_t)rr * 48 + head) * 128 + vbase + vc;
                    raw = __fsub_rn(__bfloat162float(u[off]), dr[mt][j]);
                    vnew[off] = __float2bfloat16_rn(raw);
                }
                vs[(rr - chunk * 64) * 40 + vc] =
                    __float2bfloat16_rn(__fmul_rn(raw, gates[rr - chunk * 64]));
            }
        }
        __syncthreads();
        float du[8][4] = {};
        for (int base = 0; base < 64; base += 16) {
            unsigned bb[2];
            sweep_ld2<true>(vs, 40, base, warp * 8, lane, bb);
            unsigned b0 = bb[0], b1 = bb[1];
#pragma unroll
            for (int kt = 0; kt < 8; kt++) {
                unsigned aa[4];
                sweep_ld4<true>(ks, 136, base, kt * 16, lane, aa);
                unsigned a0 = aa[0], a1 = aa[1], a2 = aa[2], a3 = aa[3];
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                             "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                             : "+f"(du[kt][0]), "+f"(du[kt][1]), "+f"(du[kt][2]), "+f"(du[kt][3])
                             : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
        }
#pragma unroll
        for (int kt = 0; kt < 8; kt++) {
#pragma unroll
            for (int j = 0; j < 4; j++)
                history[kt][j] = fmaf(history[kt][j], decay, du[kt][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int kt = 0; kt < 8; kt++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int kr = kt * 16 + lane / 4 + j / 2 * 8, vc = vbase + warp * 8 + 2 * (lane % 4) + j % 2;
            state[((size_t)head * 128 + vc) * 128 + kr] = history[kt][j];
        }
    }
}

static __device__ unsigned output_pair(const __nv_bfloat16 *x, int pitch, int row, int c) {
    return pair(x[row * pitch + c], x[row * pitch + c + 1]);
}

static __device__ unsigned output_value_pair(const __nv_bfloat16 *x, int t, int c) {
    return pair(x[t * 72 + c], x[(t + 1) * 72 + c]);
}

static __global__ void fused_output(__nv_bfloat16 *out, const __nv_bfloat16 *__restrict__ q,
                                    const __nv_bfloat16 *__restrict__ k,
                                    const __nv_bfloat16 *__restrict__ h,
                                    const __nv_bfloat16 *__restrict__ v,
                                    const float *__restrict__ g, int nt,
                                    BnGPUGDNOutputProfile output_profile) {
    int tid = threadIdx.x, lane = tid % 32, warp = tid / 32, chunk = blockIdx.x, head = blockIdx.y,
        vbase = blockIdx.z * 64;
    __shared__ __align__(16) __nv_bfloat16 qs[64 * 136], ks[64 * 136], hs[64 * 136], as[64 * 72];
    __shared__ float gs[64], gx[64];
    for (int i = tid; i < 64 * 16; i += 256) {
        int row = i / 16, c = i % 16 * 8, t = chunk * 64 + row;
        uint4 qv = make_uint4(0, 0, 0, 0), kv = qv;
        if (t < nt) {
            qv = *(const uint4 *)(q + ((size_t)t * 16 + head / 3) * 128 + c);
            kv = *(const uint4 *)(k + ((size_t)t * 16 + head / 3) * 128 + c);
        }
        *(uint4 *)(qs + row * 136 + c) = qv;
        *(uint4 *)(ks + row * 136 + c) = kv;
        *(uint4 *)(hs + row * 136 + c) =
            *(const uint4 *)(h + (((size_t)chunk * 48 + head) * 128 + vbase + row) * 128 + c);
    }
    if (tid < 64) {
        int t = chunk * 64 + tid;
        gs[tid] = t < nt ? g[t * 48 + head] : 0;
        gx[tid] = ex(gs[tid]);
    }
    __syncthreads();
    float dh[4][4] = {}, da[4][4] = {};
    for (int base = 0; base < 128; base += 16) {
        int c = base + 2 * (lane % 4), bc = warp * 8 + lane / 4;
        unsigned bh0 = output_pair(hs, 136, bc, c), bh1 = output_pair(hs, 136, bc, c + 8),
                 bk0 = output_pair(ks, 136, bc, c), bk1 = output_pair(ks, 136, bc, c + 8);
#pragma unroll
        for (int mt = 0; mt < 4; mt++) {
            int row = mt * 16 + lane / 4;
            unsigned a0 = output_pair(qs, 136, row, c), a1 = output_pair(qs, 136, row + 8, c),
                     a2 = output_pair(qs, 136, row, c + 8),
                     a3 = output_pair(qs, 136, row + 8, c + 8);
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                         "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                         : "+f"(dh[mt][0]), "+f"(dh[mt][1]), "+f"(dh[mt][2]), "+f"(dh[mt][3])
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bh0), "r"(bh1));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                         "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                         : "+f"(da[mt][0]), "+f"(da[mt][1]), "+f"(da[mt][2]), "+f"(da[mt][3])
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bk0), "r"(bk1));
        }
    }
#pragma unroll
    for (int mt = 0; mt < 4; mt++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int row = mt * 16 + lane / 4 + j / 2 * 8, col = warp * 8 + 2 * (lane % 4) + j % 2;
            float a = 0;
            if (chunk * 64 + row < nt && chunk * 64 + col < nt && row >= col)
                a = __fmul_rn(da[mt][j], ex(__fsub_rn(gs[row], gs[col])));
            as[row * 72 + col] = __float2bfloat16_rn(a);
        }
    }
    // All Q reads finish before reusing its storage for the value tile.
    __syncthreads();
    for (int i = tid; i < 64 * 8; i += 256) {
        int row = i / 8, c = i % 8 * 8, t = chunk * 64 + row;
        uint4 a = make_uint4(0, 0, 0, 0);
        if (t < nt)
            a = *(const uint4 *)(v + ((size_t)t * 48 + head) * 128 + vbase + c);
        *(uint4 *)(qs + row * 72 + c) = a;
    }
    __syncthreads();
    float dv[4][4] = {};
    for (int base = 0; base < 64; base += 16) {
        int c = base + 2 * (lane % 4), bc = warp * 8 + lane / 4;
        unsigned b0 = output_value_pair(qs, c, bc), b1 = output_value_pair(qs, c + 8, bc);
#pragma unroll
        for (int mt = 0; mt < 4; mt++) {
            int row = mt * 16 + lane / 4;
            unsigned a0 = output_pair(as, 72, row, c), a1 = output_pair(as, 72, row + 8, c),
                     a2 = output_pair(as, 72, row, c + 8), a3 = output_pair(as, 72, row + 8, c + 8);
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                         "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                         : "+f"(dv[mt][0]), "+f"(dv[mt][1]), "+f"(dv[mt][2]), "+f"(dv[mt][3])
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }
    const float scale = 0x1.6a09e6p-4f;
#pragma unroll
    for (int mt = 0; mt < 4; mt++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int row = mt * 16 + lane / 4 + j / 2 * 8, t = chunk * 64 + row,
                col = vbase + warp * 8 + 2 * (lane % 4) + j % 2;
            if (t < nt) {
                /* Original output configurations differ in contraction.
                 * Preserve the caller's explicit rounding contract even at
                 * BF16 ties caused by history/local cancellation. */
                float history = __fmul_rn(dh[mt][j], gx[row]);
                float value;
                if (output_profile == BN_GPU_GDN_OUTPUT_HISTORY_FMA)
                    value = fmaf(history, scale, __fmul_rn(dv[mt][j], scale));
                else
                    value = fmaf(dv[mt][j], scale, __fmul_rn(history, scale));
                out[((size_t)t * 48 + head) * 128 + col] = __float2bfloat16_rn(value);
            }
        }
    }
}

union WYPack {
    uint4 raw;
    unsigned short e[8];
};

static __device__ unsigned wy_pair(const __nv_bfloat16 *x, int row, int col) {
    return pair(x[row * 72 + col], x[row * 72 + col + 1]);
}

static __device__ unsigned wy_trans_pair(const __nv_bfloat16 *x, int row, int col) {
    return pair(x[row * 72 + col], x[(row + 1) * 72 + col]);
}

static __global__ void wy_tiled(__nv_bfloat16 *w, __nv_bfloat16 *u,
                                const __nv_bfloat16 *__restrict__ A,
                                const __nv_bfloat16 *__restrict__ k,
                                const __nv_bfloat16 *__restrict__ v, const float *__restrict__ beta,
                                const float *__restrict__ g, int nt) {
    int tid = threadIdx.x, lane = tid % 32, warp = tid / 32, chunk = blockIdx.x, head = blockIdx.y,
        feature = blockIdx.z * 64;
    __shared__ __align__(16) __nv_bfloat16 as[64 * 72], ws[64 * 72], us[64 * 72];
    __shared__ float bs[64], es[64];
    for (int i = tid; i < 64 * 8; i += 256) {
        int row = i / 8, c = i % 8 * 8, t = chunk * 64 + row;
        uint4 a = make_uint4(0, 0, 0, 0);
        if (t < nt)
            a = *(const uint4 *)(A + ((size_t)t * 48 + head) * 64 + c);
        *(uint4 *)(as + row * 72 + c) = a;
    }
    if (tid < 64) {
        int t = chunk * 64 + tid;
        bs[tid] = t < nt ? beta[t * 48 + head] : 0;
        es[tid] = t < nt ? ex(g[t * 48 + head]) : 0;
    }
    __syncthreads();
    for (int i = tid; i < 64 * 8; i += 256) {
        int row = i / 8, c = i % 8 * 8, t = chunk * 64 + row;
        WYPack ki, vi, wo, uo;
        ki.raw = make_uint4(0, 0, 0, 0);
        vi.raw = ki.raw;
        if (t < nt) {
            ki.raw = *(const uint4 *)(k + ((size_t)t * 16 + head / 3) * 128 + feature + c);
            vi.raw = *(const uint4 *)(v + ((size_t)t * 48 + head) * 128 + feature + c);
        }
#pragma unroll
        for (int j = 0; j < 8; j++) {
            float a = __bfloat162float(__ushort_as_bfloat16(ki.e[j])),
                  b = __bfloat162float(__ushort_as_bfloat16(vi.e[j]));
            a = __fmul_rn(a, bs[row]);
            a = __fmul_rn(a, es[row]);
            b = __fmul_rn(b, bs[row]);
            wo.e[j] = __bfloat16_as_ushort(__float2bfloat16_rn(a));
            uo.e[j] = __bfloat16_as_ushort(__float2bfloat16_rn(b));
        }
        *(uint4 *)(ws + row * 72 + c) = wo.raw;
        *(uint4 *)(us + row * 72 + c) = uo.raw;
    }
    __syncthreads();
    float dw[4][4] = {}, du[4][4] = {};
    for (int base = 0; base < 64; base += 16) {
        int c = base + 2 * (lane % 4), col = warp * 8 + lane / 4;
        unsigned bw0 = wy_trans_pair(ws, c, col), bw1 = wy_trans_pair(ws, c + 8, col),
                 bu0 = wy_trans_pair(us, c, col), bu1 = wy_trans_pair(us, c + 8, col);
#pragma unroll
        for (int mt = 0; mt < 4; mt++) {
            int row = mt * 16 + lane / 4;
            unsigned a0 = wy_pair(as, row, c), a1 = wy_pair(as, row + 8, c),
                     a2 = wy_pair(as, row, c + 8), a3 = wy_pair(as, row + 8, c + 8);
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                         "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                         : "+f"(dw[mt][0]), "+f"(dw[mt][1]), "+f"(dw[mt][2]), "+f"(dw[mt][3])
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bw0), "r"(bw1));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                         "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                         : "+f"(du[mt][0]), "+f"(du[mt][1]), "+f"(du[mt][2]), "+f"(du[mt][3])
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bu0), "r"(bu1));
        }
    }
#pragma unroll
    for (int mt = 0; mt < 4; mt++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int t = chunk * 64 + mt * 16 + lane / 4 + j / 2 * 8,
                col = feature + warp * 8 + 2 * (lane % 4) + j % 2;
            if (t < nt) {
                size_t off = ((size_t)t * 48 + head) * 128 + col;
                w[off] = __float2bfloat16_rn(dw[mt][j]);
                u[off] = __float2bfloat16_rn(du[mt][j]);
            }
        }
    }
}

union KKPack {
    uint4 raw;
    unsigned short e[8];
};

static __global__ void kkt_tiled(float *out, const __nv_bfloat16 *__restrict__ k,
                                 const float *__restrict__ beta, const float *__restrict__ g,
                                 int nt) {
    int tid = threadIdx.x, lane = tid % 32, warp = tid / 32, chunk = blockIdx.x, head = blockIdx.y;
    __shared__ __align__(16) __nv_bfloat16 ks[64 * 136];
    __shared__ __align__(16) float scaled[64 * 132];
    __shared__ float bs[64], gs[64];
    if (tid < 64) {
        int t = chunk * 64 + tid;
        bs[tid] = t < nt ? beta[t * 48 + head] : 0;
        gs[tid] = t < nt ? g[t * 48 + head] : 0;
    }
    __syncthreads();
    for (int i = tid; i < 64 * 16; i += 256) {
        int row = i / 16, c = i % 16 * 8, t = chunk * 64 + row;
        KKPack value;
        value.raw = make_uint4(0, 0, 0, 0);
        if (t < nt)
            value.raw = *(const uint4 *)(k + ((size_t)t * 16 + head / 3) * 128 + c);
        *(uint4 *)(ks + row * 136 + c) = value.raw;
        float x[8];
#pragma unroll
        for (int j = 0; j < 8; j++)
            x[j] = __fmul_rn(__bfloat162float(__ushort_as_bfloat16(value.e[j])), bs[row]);
        *(float4 *)(scaled + row * 132 + c) = make_float4(x[0], x[1], x[2], x[3]);
        *(float4 *)(scaled + row * 132 + c + 4) = make_float4(x[4], x[5], x[6], x[7]);
    }
    __syncthreads();
    float d[4][4] = {};
    for (int base = 0; base < 128; base += 8) {
        int c = (base / 16) * 16 + 2 * (lane % 4) + (base / 8) % 2, col = warp * 8 + lane / 4;
        unsigned b0 = __float_as_uint(__bfloat162float(ks[col * 136 + c])),
                 b1 = __float_as_uint(__bfloat162float(ks[col * 136 + c + 8]));
#pragma unroll
        for (int mt = 0; mt < 4; mt++)
            if ((mt + 1) * 16 > warp * 8) {
                int row = mt * 16 + lane / 4;
                unsigned a0 = __float_as_uint(scaled[row * 132 + c]),
                         a1 = __float_as_uint(scaled[(row + 8) * 132 + c]),
                         a2 = __float_as_uint(scaled[row * 132 + c + 8]),
                         a3 = __float_as_uint(scaled[(row + 8) * 132 + c + 8]);
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                             "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                             : "+f"(d[mt][0]), "+f"(d[mt][1]), "+f"(d[mt][2]), "+f"(d[mt][3])
                             : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
    }
#pragma unroll
    for (int mt = 0; mt < 4; mt++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int row = mt * 16 + lane / 4 + j / 2 * 8, col = warp * 8 + 2 * (lane % 4) + j % 2,
                t = chunk * 64 + row;
            if (t < nt) {
                float a = 0;
                if (chunk * 64 + col < nt && row > col)
                    a = __fmul_rn(d[mt][j], ex(__fsub_rn(gs[row], gs[col])));
                out[((size_t)t * 48 + head) * 64 + col] = a;
            }
        }
    }
}

/* Request-local mutable allocations. The registry never owns model weights. */
struct ResidentBuffer {
    void *data;
    size_t count, width;
    int type;
    uint32_t generation;
};
struct Workspace {
    void *blocks[8];
    int capacity;
};
struct UploadSlot {
    void *data;
    cudaEvent_t copied;
    uint64_t generation;
    int state; /* 0 available, 1 acquired, 2 copying. */
};
template<typename Row> struct SequenceGroupWorkspace {
    Row *host, *device;
    cudaEvent_t *ready;
    cudaEvent_t *copied, done;
    int capacity, slots, cursor, filled;
};
typedef SequenceGroupWorkspace<BnCudaSequenceRow> DecodeGroupWorkspace;
typedef SequenceGroupWorkspace<BnCudaQKRow> QKGroupWorkspace;
typedef SequenceGroupWorkspace<BnCudaFA2DecodeRow> AttentionGroupWorkspace;
template<typename Row>
static void free_decode_group(SequenceGroupWorkspace<Row> *w) {
    if (!w) return;
    for (int i = 0; i < w->capacity; i++)
        if (w->ready && w->ready[i]) cudaEventDestroy(w->ready[i]);
    for(int i=0;i<w->slots;i++)
        if(w->copied&&w->copied[i])cudaEventDestroy(w->copied[i]);
    if (w->done) cudaEventDestroy(w->done);
    if (w->host) cudaFreeHost(w->host);
    cudaFree(w->device); free(w->ready); free(w->copied); free(w);
}
struct CPGroupWorkspace {
    int capacity;
    cudaEvent_t *ready, done;
};
static void free_cp_group(CPGroupWorkspace *w) {
    if (!w) return;
    for (int i=0;i<w->capacity;i++) if (w->ready && w->ready[i]) cudaEventDestroy(w->ready[i]);
    if (w->done) cudaEventDestroy(w->done);
    free(w->ready); free(w);
}
struct ResidentRequest {
    const void *backend_owner;
    int device, failed;
    cudaStream_t stream;
    ResidentBuffer *buffers;
    uint32_t buffer_count;
    size_t buffer_capacity;
    Workspace workspace;
    DecodeGroupWorkspace *decode_group, *conv_group;
    QKGroupWorkspace *qk_group;
    AttentionGroupWorkspace *attention_group;
    CPGroupWorkspace *cp_group;
    void *cp_workspace;
    int cp_capacity, cp_chunk;
    BnCudaBlockQ8_1 *linear_input;
    size_t linear_input_bytes;
    cublasHandle_t linear_cublas, trellis_cublas, native_f16_cublas;
    void *trellis_workspace;
    int folded_ready,folded_f16acc;
    void *fused_locks;
    int fused_smem,fused_blocks[4],fused_joint_blocks[4];
    cudaEvent_t fused_ready,fused_done;
    void *native_moe_index_workspace;
    size_t native_moe_index_bytes;
    int native_moe_max_assignments;
    void *native_f16_workspace;
    void *linear_workspace;
    size_t linear_workspace_bytes;
    int *embedding_error;
    float *attention_lse, *attention_partial_lse, *attention_partial_out;
    int attention_tokens, attention_heads, attention_context, attention_sms;
    UploadSlot *uploads;
    unsigned upload_count;
    size_t upload_capacity;
    void *moe_banks;
    void (*moe_banks_destroy)(void *);
};
static int set_device(ResidentRequest *request) {
    return request && !request->failed && cudaSetDevice(request->device) == cudaSuccess ? 0 : -1;
}
static int synchronize(void *ptr) {
    ResidentRequest *r = (ResidentRequest *)ptr;
    if (set_device(r))
        return -1;
    int embedding_error = 0;
    cudaError_t copy = r->embedding_error
        ? cudaMemcpyAsync(&embedding_error, r->embedding_error, sizeof(embedding_error),
                          cudaMemcpyDeviceToHost, r->stream) : cudaSuccess;
    cudaError_t fence = cudaStreamSynchronize(r->stream);
    if (copy == cudaSuccess && fence == cudaSuccess && !embedding_error)
        return 0;
    if (embedding_error)
        fprintf(stderr, "CUDA resident embedding: token ID out of range\n");
    r->failed = 1;
    return -1;
}
static void free_workspace(Workspace *w) {
    for (unsigned i = 0; i < 8; ++i)
        if (w->blocks[i])
            cudaFree(w->blocks[i]);
    memset(w, 0, sizeof(*w));
}
static void free_uploads(UploadSlot *slots, unsigned count) {
    if (!slots) return;
    for (unsigned i = 0; i < count; i++) {
        if (slots[i].copied) cudaEventDestroy(slots[i].copied);
        if (slots[i].data) cudaFreeHost(slots[i].data);
    }
    free(slots);
}
static void destroy_request(void *ptr) {
    ResidentRequest *r = (ResidentRequest *)ptr;
    if (!r)
        return;
    int previous = -1;
    (void)cudaGetDevice(&previous);
    if (cudaSetDevice(r->device) == cudaSuccess) {
        /* Completion is mandatory even if the caller never waited/committed. */
        (void)cudaStreamSynchronize(r->stream);
        if (r->moe_banks_destroy) r->moe_banks_destroy(r->moe_banks);
        free_uploads(r->uploads, r->upload_count);
        free_workspace(&r->workspace);
        free_decode_group(r->decode_group);
        free_decode_group(r->conv_group);
        free_decode_group(r->qk_group);
        free_decode_group(r->attention_group);
        free_cp_group(r->cp_group);
        cudaFree(r->cp_workspace);
        if (r->linear_cublas)
            cublasDestroy(r->linear_cublas);
        cudaFree(r->linear_workspace);
        if (r->trellis_cublas) cublasDestroy(r->trellis_cublas);
        cudaFree(r->trellis_workspace);
        cudaFree(r->fused_locks);
        if(r->fused_ready)cudaEventDestroy(r->fused_ready);
        if(r->fused_done)cudaEventDestroy(r->fused_done);
        cudaFree(r->native_moe_index_workspace);
        if (r->native_f16_cublas) cublasDestroy(r->native_f16_cublas);
        cudaFree(r->native_f16_workspace);
        cudaFree(r->linear_input);
        cudaFree(r->embedding_error);
        cudaFree(r->attention_lse);
        cudaFree(r->attention_partial_lse);
        cudaFree(r->attention_partial_out);
        for (uint32_t i = 0; i < r->buffer_count; ++i)
            cudaFree(r->buffers[i].data);
        cudaStreamDestroy(r->stream);
    } else {
        fprintf(stderr, "CUDA resident request: device unavailable during cleanup\n");
    }
    free(r->buffers);
    free(r);
    if (previous >= 0)
        (void)cudaSetDevice(previous);
}
static void *create_request(void *ctx_ptr) {
    BnCudaCtx *ctx = (BnCudaCtx *)ctx_ptr;
    if (!ctx || cuda_ctx_set_device(ctx))
        return NULL;
    ResidentRequest *r = (ResidentRequest *)calloc(1, sizeof(*r));
    if (!r)
        return NULL;
    r->backend_owner = ctx_ptr;
    r->device = ctx->device;
    if (cudaStreamCreateWithFlags(&r->stream, cudaStreamNonBlocking) != cudaSuccess) {
        free(r);
        return NULL;
    }
    return r;
}
static int alloc(void *ptr, int type, size_t count, BnGPUResidentView *out) {
    ResidentRequest *r = (ResidentRequest *)ptr;
    size_t width = bn_quant_format_resident_scalar_bytes(type);
    if (!r || !out || !width || !count || count > SIZE_MAX / width ||
        r->buffer_count == UINT16_MAX || set_device(r))
        return -1;
    if (r->buffer_count == r->buffer_capacity) {
        size_t cap = r->buffer_capacity ? r->buffer_capacity * 2 : 16;
        if (cap > UINT16_MAX)
            cap = UINT16_MAX;
        if (cap > SIZE_MAX / sizeof(ResidentBuffer))
            return -1;
        void *next = realloc(r->buffers, cap * sizeof(ResidentBuffer));
        if (!next)
            return -1;
        r->buffers = (ResidentBuffer *)next;
        r->buffer_capacity = cap;
    }
    void *data = NULL;
    if (cudaMalloc(&data, count * width) != cudaSuccess)
        return -1;
    ResidentBuffer *b = &r->buffers[r->buffer_count];
    b->data = data;
    b->count = count;
    b->width = width;
    b->type = type;
    b->generation = 0;
    BnGPUResidentView view = {r, 0, count, r->buffer_count, type};
    ++r->buffer_count;
    *out = view;
    return 0;
}
static void *view_data(ResidentRequest *r, BnGPUResidentView view) {
    uint32_t index = view.slot & UINT16_MAX;
    if (!r || view.request != r || index >= r->buffer_count || !view.count)
        return NULL;
    const ResidentBuffer *b = &r->buffers[index];
    if ((view.slot >> 16) != b->generation || !b->data || view.type != b->type ||
        view.offset > b->count || view.count > b->count - view.offset)
        return NULL;
    return (unsigned char *)b->data + view.offset * b->width;
}
static int resize_discard(void *ptr, BnGPUResidentView *allocation, size_t count) {
    ResidentRequest *r = (ResidentRequest *)ptr;
    if (!allocation || !view_data(r, *allocation) || allocation->offset ||
        !count || set_device(r)) return -1;
    uint32_t index = allocation->slot & UINT16_MAX;
    ResidentBuffer *b = &r->buffers[index];
    if (allocation->count != b->count || count > SIZE_MAX / b->width) return -1;
    if (count == b->count) return 0;
    if (b->generation == UINT16_MAX || synchronize(r)) return -1;
    if (cudaFree(b->data) != cudaSuccess) { r->failed = 1; return -1; }
    b->data = NULL; b->count = 0; ++b->generation;
    if (cudaMalloc(&b->data, count * b->width) != cudaSuccess) {
        r->failed = 1; return -1;
    }
    b->count = count;
    BnGPUResidentView next = {r, 0, count, index | (b->generation << 16), b->type};
    *allocation = next;
    return 0;
}
static int upload_reserve(void *request, unsigned count, size_t capacity, size_t budget) {
    ResidentRequest *r = (ResidentRequest *)request;
    if (!r || !count || count > 8 || !capacity || capacity > SIZE_MAX / count ||
        capacity * count > budget || set_device(r)) return -1;
    if (r->uploads) return count == r->upload_count && capacity == r->upload_capacity ? 0 : -1;
    if (synchronize(r)) return -1;
    UploadSlot *slots = (UploadSlot *)calloc(count, sizeof(*slots));
    if (!slots) return -1;
    cudaError_t error = cudaSuccess;
    for (unsigned i = 0; i < count; i++) {
        error = cudaHostAlloc(&slots[i].data, capacity, cudaHostAllocDefault);
        if (error != cudaSuccess) break;
        error = cudaEventCreateWithFlags(&slots[i].copied, cudaEventDisableTiming);
        if (error != cudaSuccess) break;
    }
    if (error != cudaSuccess) {
        free_uploads(slots, count);
        if (error != cudaErrorMemoryAllocation) r->failed = 1;
        (void)cudaGetLastError();
        return -1;
    }
    r->uploads = slots; r->upload_count = count; r->upload_capacity = capacity;
    return 0;
}

static int upload_acquire(void *request, unsigned index, BnGPUHostUploadLease *out) {
    ResidentRequest *r = (ResidentRequest *)request;
    if (!r || !out || index >= r->upload_count || set_device(r)) return -1;
    UploadSlot *slot = &r->uploads[index];
    if (slot->state == 1 || slot->generation == UINT64_MAX) return -1;
    if (slot->state == 2) {
        cudaError_t error = cudaEventQuery(slot->copied);
        if (error == cudaErrorNotReady) {
            cudaError_t last = cudaGetLastError();
            if (last != cudaSuccess && last != cudaErrorNotReady) { r->failed = 1; return -1; }
            return BN_GPU_UPLOAD_BUSY;
        }
        if (error != cudaSuccess) { r->failed = 1; return -1; }
    }
    slot->state = 1; ++slot->generation;
    BnGPUHostUploadLease lease = {r, slot->data, r->upload_capacity, slot->generation, index};
    *out = lease;
    return 0;
}

static UploadSlot *upload_lease(ResidentRequest *r, const BnGPUHostUploadLease *lease) {
    if (!r || !lease || lease->request != r || lease->slot >= r->upload_count) return NULL;
    UploadSlot *slot = &r->uploads[lease->slot];
    return slot->state == 1 && lease->generation == slot->generation &&
        lease->data == slot->data && lease->capacity == r->upload_capacity ? slot : NULL;
}

static int upload_cancel(void *request, const BnGPUHostUploadLease *lease) {
    ResidentRequest *r = (ResidentRequest *)request;
    UploadSlot *slot = upload_lease(r, lease);
    if (!slot || set_device(r)) return -1;
    slot->state = 0;
    return 0;
}

static int upload_submit(void *request, const BnGPUHostUploadLease *lease, BnGPUResidentView dst) {
    ResidentRequest *r = (ResidentRequest *)request;
    UploadSlot *slot = upload_lease(r, lease);
    void *device = view_data(r, dst);
    if (!slot || !device || dst.count > SIZE_MAX / r->buffers[dst.slot & UINT16_MAX].width ||
        dst.count * r->buffers[dst.slot & UINT16_MAX].width > r->upload_capacity || set_device(r)) return -1;
    if (cudaGetLastError() != cudaSuccess) { r->failed = 1; return -1; }
    slot->state = 2;
    if (cudaMemcpyAsync(device, slot->data, dst.count * r->buffers[dst.slot & UINT16_MAX].width,
            cudaMemcpyHostToDevice, r->stream) != cudaSuccess ||
        cudaEventRecord(slot->copied, r->stream) != cudaSuccess) {
        r->failed = 1;
        return -1;
    }
    return 0;
}

static int transfer(void *ptr, BnGPUResidentView view, void *host, int to_device) {
    ResidentRequest *r = (ResidentRequest *)ptr;
    void *device = view_data(r, view);
    if (!device || !host || set_device(r))
        return -1;
    size_t bytes = view.count * r->buffers[view.slot & UINT16_MAX].width;
    cudaError_t error =
        to_device ? cudaMemcpyAsync(device, host, bytes, cudaMemcpyHostToDevice, r->stream)
                  : cudaMemcpyAsync(host, device, bytes, cudaMemcpyDeviceToHost, r->stream);
    if (error != cudaSuccess) {
        r->failed = 1;
        return -1;
    }
    return synchronize(r);
}
static int write(void *ptr, BnGPUResidentView dst, const void *host) {
    return transfer(ptr, dst, const_cast<void *>(host), 1);
}
static int zero(void *ptr, BnGPUResidentView dst) {
    ResidentRequest *r = (ResidentRequest *)ptr;
    void *device = view_data(r, dst);
    if (!device || set_device(r))
        return -1;
    size_t bytes = dst.count * r->buffers[dst.slot & UINT16_MAX].width;
    if (cudaMemsetAsync(device, 0, bytes, r->stream) != cudaSuccess) {
        r->failed = 1;
        return -1;
    }
    return 0;
}
static int read(void *ptr, BnGPUResidentView src, void *host) {
    return transfer(ptr, src, host, 0);
}
/* One calculation for admission and allocation. Chunk state storage rounds up
 * at 64-token boundaries, including one-token prefill tails. */
static size_t workspace_sizes(int cap, size_t bytes[8]) {
    if (cap <= 0 || cap > INT_MAX / (48 * 128)) return 0;
    size_t ng = (size_t)cap * 48, na = ng * 64, nw = ng * 128;
    size_t nh = (size_t)((cap + 63) / 64) * 48 * 128 * 128;
    const size_t counts[8] = {ng, na, na, na, nw, nw, nh, nw};
    const size_t widths[8] = {4, 4, 4, 2, 2, 2, 2, 2};
    size_t total = 0;
    for (unsigned i = 0; i < 8; ++i) {
        if (counts[i] > SIZE_MAX / widths[i]) return 0;
        bytes[i] = counts[i] * widths[i];
        if (bytes[i] > SIZE_MAX - total) return 0;
        total += bytes[i];
    }
    return total;
}
static size_t prefill_workspace_size(void *ctx_ptr, int cap) {
    const BnCudaCtx *ctx = (const BnCudaCtx *)ctx_ptr;
    size_t bytes[8];
    return ctx && ctx->compute_capability >= 800 ? workspace_sizes(cap, bytes) : 0;
}
static int reserve(void *ctx_ptr, void *ptr, int cap) {
    BnCudaCtx *ctx = (BnCudaCtx *)ctx_ptr;
    ResidentRequest *r = (ResidentRequest *)ptr;
    size_t bytes[8];
    if (!ctx || !r || r->backend_owner != ctx || ctx->compute_capability < 800 || cap <= 0 ||
        !workspace_sizes(cap, bytes) || set_device(r))
        return -1;
    if (cap <= r->workspace.capacity)
        return 0;
    cudaFuncAttributes attributes;
    int shared_limit = 0;
    if (cudaFuncGetAttributes(&attributes, fused_output) != cudaSuccess ||
        cudaDeviceGetAttribute(&shared_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, r->device) !=
            cudaSuccess ||
        attributes.sharedSizeBytes > (size_t)shared_limit)
        return -1;
    if (synchronize(r))
        return -1;
    Workspace next = {};
    for (unsigned i = 0; i < 8; ++i) {
        if (cudaMalloc(&next.blocks[i], bytes[i]) != cudaSuccess) {
            free_workspace(&next);
            return -1;
        }
    }
    next.capacity = cap;
    free_workspace(&r->workspace);
    r->workspace = next;
    return 0;
}
static int overlap(BnGPUResidentView a, BnGPUResidentView b) {
    if (a.request != b.request || a.slot != b.slot)
        return 0;
    return a.offset <= b.offset ? b.offset - a.offset < a.count : a.offset - b.offset < b.count;
}
static int prefill(void *ctx_ptr, void *ptr, const BnGPUGDNPrefillResidentPlan *p) {
    BnCudaCtx *ctx = (BnCudaCtx *)ctx_ptr;
    ResidentRequest *r = (ResidentRequest *)ptr;
    if (!ctx || !r || !p || r->backend_owner != ctx || ctx->compute_capability < 800 ||
        p->n_tokens <= 0 || p->n_tokens > r->workspace.capacity || p->hk != 16 || p->hv != 48 ||
        p->dk != 128 || p->dv != 128 || p->q_scale != 0x1.6a09e6p-4f ||
        (p->output_profile != BN_GPU_GDN_OUTPUT_LOCAL_FMA &&
         p->output_profile != BN_GPU_GDN_OUTPUT_HISTORY_FMA) || set_device(r))
        return -1;
    size_t nq = (size_t)p->n_tokens * 16 * 128, nv = (size_t)p->n_tokens * 48 * 128;
    size_t ng = (size_t)p->n_tokens * 48, ns = 48u * 128 * 128;
    BnGPUResidentView views[] = {
        p->q, p->k, p->v, p->log_decay, p->beta, p->initial_state, p->next_state, p->out};
    size_t counts[] = {nq, nq, nv, ng, ng, ns, ns, nv};
    void *data[8];
    for (unsigned i = 0; i < 8; ++i) {
        data[i] = view_data(r, views[i]);
        int is_bf16 = i < 3 || i == 7;
        if (!data[i] || views[i].count != counts[i] ||
            (is_bf16 ? !bn_quant_format_is_bf16(views[i].type)
                     : !bn_quant_format_is_f32(views[i].type)) ||
            (i < 3 && ((uintptr_t)data[i] & 15u)))
            return -1;
    }
    for (unsigned i = 0; i < 8; ++i) {
        if ((i != 6 && overlap(views[6], views[i])) || (i != 7 && overlap(views[7], views[i])))
            return -1;
    }
    Workspace *w = &r->workspace;
    float *gc = (float *)w->blocks[0], *a = (float *)w->blocks[1], *ai = (float *)w->blocks[2];
    __nv_bfloat16 *ab = (__nv_bfloat16 *)w->blocks[3], *ww = (__nv_bfloat16 *)w->blocks[4];
    __nv_bfloat16 *u = (__nv_bfloat16 *)w->blocks[5], *h = (__nv_bfloat16 *)w->blocks[6];
    __nv_bfloat16 *vn = (__nv_bfloat16 *)w->blocks[7];
    int nt = p->n_tokens;
    dim3 grid((nt + 63) / 64, 48);
    const __nv_bfloat16 *q = (const __nv_bfloat16 *)data[0], *k = (const __nv_bfloat16 *)data[1];
    const __nv_bfloat16 *v = (const __nv_bfloat16 *)data[2];
    const float *g = (const float *)data[3], *beta = (const float *)data[4];
    // Consume stale launch errors before queueing any operation.
    if (cudaGetLastError() != cudaSuccess) {
        r->failed = 1;
        return -1;
    }
    scan<<<grid, 32, 0, r->stream>>>(gc, g, nt, 48);
    kkt_tiled<<<grid, 256, 0, r->stream>>>(a, k, beta, gc, nt);
    inverse<<<grid, 256, 0, r->stream>>>(ai, a, nt, 4);
    size_t na = (size_t)nt * 48 * 64;
    cast_inverse<<<(na + 255) / 256, 256, 0, r->stream>>>(ab, ai, na);
    wy_tiled<<<dim3(grid.x, 48, 2), 256, 0, r->stream>>>(ww, u, ab, k, v, beta, gc, nt);
    fused_sweep<<<dim3(48, 4), 128, 0, r->stream>>>((float *)data[6], h, vn, ww, u, k, gc,
                                                    (const float *)data[5], nt);
    fused_output<<<dim3(grid.x, 48, 2), 256, 0, r->stream>>>((__nv_bfloat16 *)data[7], q, k, h, vn,
                                                             gc, nt, p->output_profile);
    if (cudaGetLastError() == cudaSuccess)
        return 0;
    r->failed = 1;
    return -1;
}

static size_t embedding_workspace_size(void *ctx_ptr) {
    const BnCudaCtx *ctx = (const BnCudaCtx *)ctx_ptr;
    return ctx && ctx->compute_capability >= 800 ? sizeof(int) : 0;
}

static int embedding_reserve(void *ctx_ptr, void *ptr) {
    BnCudaCtx *ctx = (BnCudaCtx *)ctx_ptr;
    ResidentRequest *r = (ResidentRequest *)ptr;
    if (!ctx || !r || r->backend_owner != ctx || ctx->compute_capability < 800 || set_device(r))
        return -1;
    if (r->embedding_error)
        return 0;
    int *status = NULL;
    if (cudaMalloc(&status, sizeof(*status)) != cudaSuccess)
        return -1;
    if (cudaMemsetAsync(status, 0, sizeof(*status), r->stream) != cudaSuccess || synchronize(r)) {
        cudaFree(status);
        return -1;
    }
    r->embedding_error = status;
    return 0;
}

static __global__ void embedding_kernel(__nv_bfloat16 *out, const int32_t *ids,
    const BnBlockQ4K *weight, int *error, size_t count, int dim, int vocab) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    int token = ids[i / (size_t)dim];
    if (token < 0 || token >= vocab) {
        atomicExch(error, 1);
        out[i] = __float2bfloat16_rn(0.0f);
        return;
    }
    size_t offset = (size_t)token * dim + i % (size_t)dim;
    const BnBlockQ4K *w = weight + offset / 256;
    int pos = offset % 256, scale, minimum;
    cuda_kquant_group_scale_min(w->scales, pos / 32, &scale, &minimum);
    int q = cuda_bf16_q4_quant(w, pos / 32, pos % 32);
    /* Pinned CUDA GGUF embedding rounds scale products in HALF and contracts
     * the final multiply/subtract to HALF FMA, then converts to BF16. */
    __half d = __hmul_rn(__ushort_as_half(w->d), __int2half_rn(scale));
    __half m = __hmul_rn(__ushort_as_half(w->dmin), __int2half_rn(minimum));
    out[i] = __float2bfloat16_rn(__half2float(__hfma(d, __int2half_rn(q), __hneg(m))));
}

static __global__ void embedding_q8_kernel(__nv_bfloat16 *out, const int32_t *ids,
    const BnBlockQ8_0 *weight, int *error, size_t count, int dim, int vocab) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int token = ids[i / (size_t)dim];
    if (token < 0 || token >= vocab) {
        atomicExch(error, 1); out[i] = __float2bfloat16_rn(0.0f); return;
    }
    size_t offset = (size_t)token * dim + i % (size_t)dim;
    const BnBlockQ8_0 *w = weight + offset / 32;
    __half value = __hmul_rn(__ushort_as_half(w->d), __int2half_rn(w->qs[offset % 32]));
    out[i] = __float2bfloat16_rn(__half2float(value));
}

static int embedding(void *ctx_ptr, void *ptr, const BnGPUEmbeddingResidentPlan *p) {
    BnCudaCtx *ctx = (BnCudaCtx *)ctx_ptr;
    ResidentRequest *r = (ResidentRequest *)ptr;
    if (!ctx || !r || !p || r->backend_owner != ctx || ctx->compute_capability < 800 ||
        !r->embedding_error || p->n_tokens <= 0 || p->dim <= 0 || p->vocab_size <= 0 ||
        p->n_tokens > INT_MAX / p->dim || set_device(r))
        return -1;
    const BnCudaBuffer *w = (const BnCudaBuffer *)p->weight;
    const BnQuantFormatOps *format = w ? bn_quant_format_ops(w->type) : NULL;
    if (!format || (format->bf16_embedding != BN_QUANT_BF16_EMBEDDING_Q4_K &&
                   format->bf16_embedding != BN_QUANT_BF16_EMBEDDING_Q8_0) || !w->data ||
        w->rows != p->vocab_size || w->cols != p->dim || p->dim % format->block_elems ||
        (size_t)p->vocab_size > SIZE_MAX / (size_t)p->dim)
        return -1;
    size_t blocks = (size_t)p->vocab_size * p->dim / format->block_elems;
    if (blocks > SIZE_MAX / format->bytes_per_block ||
        w->size < blocks * format->bytes_per_block)
        return -1;
    void *ids = view_data(r, p->indices), *out = view_data(r, p->out);
    size_t count = (size_t)p->n_tokens * p->dim;
    if (!ids || !out || p->indices.type != BN_GGUF_TENSOR_I32 ||
        !bn_quant_format_is_bf16(p->out.type) || p->indices.count != (size_t)p->n_tokens ||
        p->out.count != count || overlap(p->indices, p->out))
        return -1;
    if (cudaGetLastError() != cudaSuccess) {
        r->failed = 1;
        return -1;
    }
    if (format->bf16_embedding == BN_QUANT_BF16_EMBEDDING_Q8_0)
        embedding_q8_kernel<<<(unsigned)((count + 255) / 256), 256, 0, r->stream>>>(
            (__nv_bfloat16 *)out, (const int32_t *)ids, (const BnBlockQ8_0 *)w->data,
            r->embedding_error, count, p->dim, p->vocab_size);
    else
        embedding_kernel<<<(unsigned)((count + 255) / 256), 256, 0, r->stream>>>(
            (__nv_bfloat16 *)out, (const int32_t *)ids, (const BnBlockQ4K *)w->data,
            r->embedding_error, count, p->dim, p->vocab_size);
    if (cudaGetLastError() == cudaSuccess)
        return 0;
    r->failed = 1;
    return -1;
}

template <bool Add, int Width, bool Grouped, bool AddThird = false>
static __global__ void contiguous_rmsnorm_kernel(__nv_bfloat16 *out,
    __nv_bfloat16 *residual_out, const __nv_bfloat16 *x,
    const __nv_bfloat16 *residual, const __nv_bfloat16 *weight,
    float eps, int weight_offset, int reduction_threads, int dim, float inverse_dim, int weight_groups,
    const __nv_bfloat16 *addend) {
    const int tid=threadIdx.x;
    const int width=Width?Width:dim;
    const size_t base=(size_t)blockIdx.x*width;
    const size_t weight_base=Grouped?(size_t)(blockIdx.x%weight_groups)*width:0;
    /* Keep the reference logical reduction lanes even when more physical
     * threads parallelize the independent output work. */
    bool split=width>=8192;
    int groups=split?512>>(__ffs(reduction_threads)-1):1;
    int active_threads=groups*reduction_threads;
    int column=split?tid&(reduction_threads-1):tid;
    float acc[4]={0,0,0,0};
    /* Match a materialized F32 square followed by the contiguous four-lane
     * reduction. Contracting square+sum into FMA changes the reference. */
    for(int i=tid*4;tid<active_threads&&i<width;i+=active_threads*4)
        for(int j=0;j<4;j++) {
            float value=__bfloat162float(x[base+i+j]);
            if(Add)value=__fadd_rn(value,__bfloat162float(residual[base+i+j]));
            if(AddThird)value=__fadd_rn(value,__bfloat162float(addend[base+i+j]));
            acc[j]=__fadd_rn(acc[j],__fmul_rn(value,value));
        }
    float sum=__fadd_rn(__fadd_rn(__fadd_rn(acc[0],acc[1]),acc[2]),acc[3]);
    __shared__ float partial[512],total;
    partial[tid]=sum;
    for(int offset=reduction_threads/2;offset>=32;offset>>=1) {
        __syncthreads();
        if(split?column<offset:tid<offset){sum=__fadd_rn(sum,partial[tid+offset]);partial[tid]=sum;}
    }
    __syncthreads();
    if(split?column<32:tid<32) {
        for(int offset=16;offset;offset>>=1)
            sum=__fadd_rn(sum,__shfl_down_sync(0xffffffffu,sum,offset));
        if(column==0){
            if(groups==1)total=sum;
            else partial[tid]=sum;
        }
    }
    __syncthreads();
    /* Torch reduces block X before block Y once wide rows split over warps. */
    if(groups>1) {
        if(tid<32) {
            float group_sum=tid<groups?partial[tid*reduction_threads]:0.0f;
            for(int offset=groups/2;offset;offset>>=1)
                group_sum=__fadd_rn(group_sum,__shfl_down_sync(0xffffffffu,group_sum,offset));
            if(tid==0)total=group_sum;
        }
        __syncthreads();
    }
    float mean=__fmul_rn(total,Width?1.0f/(float)Width:inverse_dim);
    float scale=rsqrtf(__fadd_rn(mean,eps));
    for(int i=tid;i<width;i+=blockDim.x) {
        float value=__bfloat162float(x[base+i]);
        if(Add)value=__fadd_rn(value,__bfloat162float(residual[base+i]));
        if(AddThird)value=__fadd_rn(value,__bfloat162float(addend[base+i]));
        float gamma=__bfloat162float(weight[weight_base+i]);
        if(weight_offset)gamma=__fadd_rn(gamma,1.0f);
        out[base+i]=__float2bfloat16_rn(__fmul_rn(__fmul_rn(value,scale),gamma));
        if(Add)residual_out[base+i]=__float2bfloat16_rn(value);
    }
}

static int contiguous_norm_threads(int tokens) {
    int outputs=1;
    while(outputs<16&&outputs<=tokens/2)outputs*=2;
    return 512/outputs;
}

template <bool Add, bool Grouped, bool AddThird = false>
static void launch_contiguous_norm(__nv_bfloat16 *out, __nv_bfloat16 *residual_out,
    const __nv_bfloat16 *x, const __nv_bfloat16 *residual,
    const __nv_bfloat16 *weight, float eps, int weight_offset,
    int tokens, int dim, int weight_groups, cudaStream_t stream,
    const __nv_bfloat16 *addend = NULL) {
    int logical=contiguous_norm_threads(tokens);
    int physical=tokens<=16||dim>=8192?512:logical;
    float inverse_dim=1.0f/(float)dim;
#define BN_CONTIGUOUS_NORM(W) \
    contiguous_rmsnorm_kernel<Add,W,Grouped,AddThird><<<tokens,physical,0,stream>>>( \
        out,residual_out,x,residual,weight,eps,weight_offset,logical,dim,inverse_dim,weight_groups,addend)
    if(dim==2560) { BN_CONTIGUOUS_NORM(2560); }
    else if(dim==4096) { BN_CONTIGUOUS_NORM(4096); }
    else if(dim==5120) { BN_CONTIGUOUS_NORM(5120); }
    else { BN_CONTIGUOUS_NORM(0); }
#undef BN_CONTIGUOUS_NORM
}

/* Fixed tiled F32 contracts. Preserve the original persistent two-input and
 * looped three-input square pairing; LoopedAdd selects the independently
 * qualified 1024-column two-input reduction (second tile squared before
 * the first tile FMA, followed by three sequential FMAs). General1 selects
 * persistent layouts for all counts (unfused three-input squares); General2
 * selects the looped layout for all counts; General3 selects persistent
 * fused square pairs for all counts. Legacy contracts remain fixed. */
template <int Inputs>
static __device__ float tiled_norm_value(const __nv_bfloat16 *x,
    const __nv_bfloat16 *residual, const __nv_bfloat16 *addend, size_t i) {
    float value = __bfloat162float(x[i]);
    if (Inputs >= 2) value = __fadd_rn(value, __bfloat162float(residual[i]));
    if (Inputs == 3) value = __fadd_rn(value, __bfloat162float(addend[i]));
    return value;
}

template <int Inputs, bool LoopedAdd = false, int General = 0>
static __global__ void tiled_norm_5120_kernel(__nv_bfloat16 *out,
    __nv_bfloat16 *residual_out, const __nv_bfloat16 *x,
    const __nv_bfloat16 *residual, const __nv_bfloat16 *addend,
    const __nv_bfloat16 *weight) {
    const int tid = threadIdx.x;
    const size_t base = (size_t)blockIdx.x * 5120;
    float sum;
    if ((Inputs == 2 && LoopedAdd) || General == 2) {
        float a[4];
        for (int j = 0; j < 4; j++) {
            float v = tiled_norm_value<Inputs>(x,residual,addend,base+tid*4+j);
            float v1 = tiled_norm_value<Inputs>(x,residual,addend,base+tid*4+j+1024);
            float acc = __fmaf_rn(v,v,__fmul_rn(v1,v1));
            for (int k = 2; k < 5; k++) {
                v = tiled_norm_value<Inputs>(x,residual,addend,base+tid*4+j+k*1024);
                acc = __fmaf_rn(v,v,acc);
            }
            a[j] = acc;
        }
        sum = __fadd_rn(__fadd_rn(a[0],a[2]),__fadd_rn(a[1],a[3]));
    } else if (Inputs == 1 && General == 0) {
        float a[4];
        for (int j = 0; j < 4; j++) {
            float v0 = tiled_norm_value<1>(x,residual,addend,base+tid*4+j);
            float v1 = tiled_norm_value<1>(x,residual,addend,base+tid*4+j+1024);
            float acc = __fadd_rn(__fmul_rn(v0,v0),__fmul_rn(v1,v1));
            for (int k = 2; k < 5; k++) {
                float v = tiled_norm_value<1>(x,residual,addend,base+tid*4+j+k*1024);
                acc = __fmaf_rn(v,v,acc);
            }
            a[j] = acc;
        }
        sum = __fadd_rn(__fadd_rn(a[0],a[2]),__fadd_rn(a[1],a[3]));
    } else if (Inputs == 2 || General == 1 || General == 3) {
        float v[16];
        for (int k = 0; k < 2; k++)
            for (int j = 0; j < 8; j++) {
                int i = tid*8+j+k*4096;
                v[k*8+j] = i < 5120 ? tiled_norm_value<Inputs>(x,residual,addend,base+i) : 0.0f;
            }
        if (Inputs == 3 && General == 1) {
        float parity[2];
        for (int j = 0; j < 2; j++) {
            float first = __fadd_rn(
                __fadd_rn(__fmul_rn(v[j],v[j]),__fmul_rn(v[j+2],v[j+2])),
                __fadd_rn(__fmul_rn(v[j+4],v[j+4]),__fmul_rn(v[j+6],v[j+6])));
            float second = __fadd_rn(
                __fadd_rn(__fmul_rn(v[j+8],v[j+8]),__fmul_rn(v[j+10],v[j+10])),
                __fadd_rn(__fmul_rn(v[j+12],v[j+12]),__fmul_rn(v[j+14],v[j+14])));
            parity[j] = __fadd_rn(first,second);
        }
        sum = __fadd_rn(parity[0],parity[1]);
        } else {
        float parity[2];
        for (int j = 0; j < 2; j++) {
            float first = __fadd_rn(
                __fmaf_rn(v[j],v[j],__fmul_rn(v[j+2],v[j+2])),
                __fmaf_rn(v[j+4],v[j+4],__fmul_rn(v[j+6],v[j+6])));
            float second = __fadd_rn(
                __fmaf_rn(v[j+8],v[j+8],__fmul_rn(v[j+10],v[j+10])),
                __fmaf_rn(v[j+12],v[j+12],__fmul_rn(v[j+14],v[j+14])));
            parity[j] = __fadd_rn(first,second);
        }
        sum = __fadd_rn(parity[0],parity[1]);
        }
    } else {
        float a[8];
        for (int j = 0; j < 8; j++) {
            int i = tid*8+j;
            float first = tiled_norm_value<3>(x,residual,addend,base+i);
            float second = i+4096 < 5120
                ? tiled_norm_value<3>(x,residual,addend,base+i+4096) : 0.0f;
            a[j] = __fmaf_rn(second,second,__fmul_rn(first,first));
        }
        sum = __fadd_rn(
            __fadd_rn(__fadd_rn(a[0],a[2]),__fadd_rn(a[4],a[6])),
            __fadd_rn(__fadd_rn(a[1],a[3]),__fadd_rn(a[5],a[7])));
    }
    for (int offset = 16; offset; offset >>= 1)
        sum = __fadd_rn(sum,__shfl_xor_sync(0xffffffffu,sum,offset));
    __shared__ float partial[16];
    if (!(tid & 31)) partial[tid/32] = sum;
    __syncthreads();
    const int warps = (Inputs == 1 && General == 0) || LoopedAdd || General == 2 ? 8 : 16;
    sum = partial[tid & (warps-1)];
    for (int offset = warps/2; offset; offset >>= 1)
        sum = __fadd_rn(sum,__shfl_xor_sync(0xffffffffu,sum,offset));
    float shifted = __fmaf_rn(sum,1.0f/5120.0f,1e-6f);
    float scale = rsqrtf(shifted);
    for (int i = tid; i < 5120; i += blockDim.x) {
        float value = tiled_norm_value<Inputs>(x,residual,addend,base+i);
        float gamma = __fadd_rn(__bfloat162float(weight[i]),1.0f);
        out[base+i] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(value,scale),gamma));
        if (Inputs >= 2) residual_out[base+i] = __float2bfloat16_rn(value);
    }
}

static bool tiled_norm_5120_valid(int dim, float eps, int offset, int groups) {
    return dim == 5120 && eps == 1e-6f && offset == 1 && groups >= 0 && groups <= 1;
}

static __global__ void vector8_rmsnorm_kernel(__nv_bfloat16 *out,const __nv_bfloat16 *x,
    const __nv_bfloat16 *weight,int dim,float eps) {
    int tid=threadIdx.x,lane=tid&31,warp=tid>>5;size_t base=(size_t)blockIdx.x*dim;
    float sum=0;
    for(int j=0;j<8;++j){float v=__bfloat162float(x[base+tid*8+j]);sum=__fmaf_rn(v,v,sum);}
    for(int off=1;off<32;off*=2){float other=__shfl_down_sync(0xffffffffu,sum,off);if(lane+off<32)sum=__fadd_rn(sum,other);}
    __shared__ float partial[6],scale;
    if(!lane)partial[warp]=sum;
    __syncthreads();
    if(!tid){float total=partial[0];for(int j=1;j<blockDim.x/32;++j)total=__fadd_rn(total,partial[j]);scale=rsqrtf(__fadd_rn(__fdiv_rn(total,(float)dim),eps));}
    __syncthreads();
    for(int j=0;j<8;++j){int col=tid*8+j;float norm=__fmul_rn(__bfloat162float(x[base+col]),scale);out[base+col]=__float2bfloat16_rn(__fmul_rn(norm,__bfloat162float(weight[col])));}
}

static __global__ void vector8_4096_rmsnorm_kernel(__nv_bfloat16 *out,const __nv_bfloat16 *x,
    const __nv_bfloat16 *weight) {
    int tid=threadIdx.x,lane=tid&31,warp=tid>>5;size_t base=(size_t)blockIdx.x*4096;
    float sum=0;
    for(int start=tid*8;start<4096;start+=blockDim.x*8)
        for(int j=0;j<8;++j){float v=__bfloat162float(x[base+start+j]);sum=__fmaf_rn(v,v,sum);}
    for(int off=1;off<32;off*=2){float other=__shfl_down_sync(0xffffffffu,sum,off);if(lane+off<32)sum=__fadd_rn(sum,other);}
    __shared__ float partial[16],scale;
    if(!lane)partial[warp]=sum;
    __syncthreads();
    if(!tid){float total=partial[0];for(int j=1;j<blockDim.x/32;++j)total=__fadd_rn(total,partial[j]);scale=rsqrtf(__fadd_rn(__fdiv_rn(total,4096.f),1e-5f));}
    __syncthreads();
    for(int start=tid*8;start<4096;start+=blockDim.x*8)
        for(int j=0;j<8;++j){int col=start+j;float value=__fmul_rn(__bfloat162float(x[base+col]),scale);out[base+col]=__float2bfloat16_rn(__fmul_rn(value,__bfloat162float(weight[col])));}
}

static int rms_norm(void *ctx_ptr, void *ptr, const BnGPURMSNormResidentPlan *p) {
    BnCudaCtx *ctx = (BnCudaCtx *)ctx_ptr;
    ResidentRequest *r = (ResidentRequest *)ptr;
    if (!ctx || !r || !p || r->backend_owner != ctx || ctx->compute_capability < 800 ||
        p->n_tokens <= 0 || p->dim <= 0 || p->n_tokens > INT_MAX / p->dim ||
        (p->reduction != BN_GPU_RMSNORM_REDUCTION_DEFAULT &&
         !(p->reduction==BN_GPU_RMSNORM_REDUCTION_VECTOR8_4096&&p->dim==4096&&p->eps==1e-5f&&p->weight_offset==0&&p->weight_groups<=1&&p->weight_groups>=0&&!(p->x.offset%8)&&!(p->out.offset%8)) &&
         !(p->reduction==BN_GPU_RMSNORM_REDUCTION_VECTOR8_F32&&(p->dim==512||p->dim==1536)&&p->eps==1e-5f&&p->weight_offset==0) &&
         !(p->reduction == BN_GPU_RMSNORM_REDUCTION_CONTIGUOUS_F32 &&
           p->dim >= 2048 && p->dim <= 130560 && !(p->dim & 3)) &&
         !((p->reduction == BN_GPU_RMSNORM_REDUCTION_TILED_5120_FMA ||
            p->reduction == BN_GPU_RMSNORM_REDUCTION_TILED_5120_LOOPED_ADD_FMA ||
            p->reduction == BN_GPU_RMSNORM_REDUCTION_TILED_5120_PERSISTENT ||
            p->reduction == BN_GPU_RMSNORM_REDUCTION_TILED_5120_LOOPED ||
            p->reduction == BN_GPU_RMSNORM_REDUCTION_TILED_5120_PERSISTENT_FUSED_SQUARES) &&
           tiled_norm_5120_valid(p->dim,p->eps,p->weight_offset,p->weight_groups))) ||
        !isfinite(p->eps) || p->eps < 0.0f ||
        (p->weight_offset != 0 && p->weight_offset != 1) || set_device(r))
        return -1;
    int groups=p->weight_groups>0?p->weight_groups:1;
    if(p->weight_groups<0||groups>p->n_tokens||p->n_tokens%groups||
       (groups>1&&p->reduction!=BN_GPU_RMSNORM_REDUCTION_CONTIGUOUS_F32))return -1;
    const BnCudaBuffer *w = (const BnCudaBuffer *)p->weight;
    size_t count = (size_t)p->n_tokens * p->dim;
    void *x = view_data(r, p->x), *out = view_data(r, p->out);
    if (!w || !w->data || !bn_quant_format_is_bf16(w->type) ||
        w->size / sizeof(uint16_t) < (size_t)groups*p->dim || !x || !out ||
        p->x.count != count || p->out.count != count ||
        !bn_quant_format_is_bf16(p->x.type) || !bn_quant_format_is_bf16(p->out.type) ||
        overlap(p->x, p->out))
        return -1;
    if (cudaGetLastError() != cudaSuccess) {
        r->failed = 1;
        return -1;
    }
    if(p->reduction==BN_GPU_RMSNORM_REDUCTION_VECTOR8_4096){
        int threads=p->n_tokens<256?512:256;
        vector8_4096_rmsnorm_kernel<<<p->n_tokens,threads,0,r->stream>>>((__nv_bfloat16*)out,
            (const __nv_bfloat16*)x,(const __nv_bfloat16*)w->data);
        if(cudaGetLastError()==cudaSuccess)return 0;
        r->failed=1;return -1;
    }
    if(p->reduction==BN_GPU_RMSNORM_REDUCTION_VECTOR8_F32){
        vector8_rmsnorm_kernel<<<p->n_tokens,p->dim/8,0,r->stream>>>((__nv_bfloat16*)out,
            (const __nv_bfloat16*)x,(const __nv_bfloat16*)w->data,p->dim,p->eps);
        if(cudaGetLastError()==cudaSuccess)return 0;r->failed=1;return -1;
    }
    if (p->reduction == BN_GPU_RMSNORM_REDUCTION_TILED_5120_PERSISTENT ||
        p->reduction == BN_GPU_RMSNORM_REDUCTION_TILED_5120_LOOPED ||
        p->reduction == BN_GPU_RMSNORM_REDUCTION_TILED_5120_PERSISTENT_FUSED_SQUARES) {
        if (p->reduction != BN_GPU_RMSNORM_REDUCTION_TILED_5120_LOOPED)
            tiled_norm_5120_kernel<1,false,1><<<p->n_tokens,512,0,r->stream>>>(
                (__nv_bfloat16 *)out,NULL,(const __nv_bfloat16 *)x,NULL,NULL,
                (const __nv_bfloat16 *)w->data);
        else tiled_norm_5120_kernel<1,false,2><<<p->n_tokens,256,0,r->stream>>>(
                (__nv_bfloat16 *)out,NULL,(const __nv_bfloat16 *)x,NULL,NULL,
                (const __nv_bfloat16 *)w->data);
        if (cudaGetLastError() == cudaSuccess) return 0;
        r->failed = 1; return -1;
    }
    if (p->reduction == BN_GPU_RMSNORM_REDUCTION_TILED_5120_FMA ||
        p->reduction == BN_GPU_RMSNORM_REDUCTION_TILED_5120_LOOPED_ADD_FMA) {
        tiled_norm_5120_kernel<1><<<p->n_tokens,256,0,r->stream>>>(
            (__nv_bfloat16 *)out,NULL,(const __nv_bfloat16 *)x,NULL,NULL,
            (const __nv_bfloat16 *)w->data);
        if (cudaGetLastError() == cudaSuccess) return 0;
        r->failed = 1; return -1;
    }
    if(p->reduction==BN_GPU_RMSNORM_REDUCTION_CONTIGUOUS_F32) {
        if(groups>1) {
            launch_contiguous_norm<false,true>((__nv_bfloat16*)out,NULL,
                (const __nv_bfloat16*)x,NULL,(const __nv_bfloat16*)w->data,
                p->eps,p->weight_offset,p->n_tokens,p->dim,groups,r->stream);
        } else {
            launch_contiguous_norm<false,false>((__nv_bfloat16*)out,NULL,
                (const __nv_bfloat16*)x,NULL,(const __nv_bfloat16*)w->data,
                p->eps,p->weight_offset,p->n_tokens,p->dim,groups,r->stream);
        }
        if(cudaGetLastError()==cudaSuccess)return 0;
        r->failed=1;return -1;
    }
    int threads = cuda_rmsnorm_threads(p->dim);
    rmsnorm_bf16_batch_kernel<<<p->n_tokens, threads, (size_t)(threads / 32) * sizeof(float),
        r->stream>>>((__nv_bfloat16 *)out, (const __nv_bfloat16 *)x,
                     (const __nv_bfloat16 *)w->data, p->dim, p->eps, p->weight_offset);
    if (cudaGetLastError() == cudaSuccess)
        return 0;
    r->failed = 1;
    return -1;
}

static int add_norm(void *ctx_ptr, void *ptr, const BnGPUAddRMSNormResidentPlan *p) {
    BnCudaCtx *ctx = (BnCudaCtx *)ctx_ptr;
    ResidentRequest *r = (ResidentRequest *)ptr;
    if (!ctx || !r || !p || r->backend_owner != ctx || ctx->compute_capability < 800 ||
        p->n_tokens <= 0 || p->dim <= 0 || p->n_tokens > INT_MAX / p->dim ||
        (p->reduction != BN_GPU_RMSNORM_REDUCTION_DEFAULT &&
         !(p->reduction == BN_GPU_RMSNORM_REDUCTION_CONTIGUOUS_F32 &&
           p->dim >= 2048 && p->dim <= 130560 && !(p->dim & 3)) &&
         !((p->reduction == BN_GPU_RMSNORM_REDUCTION_TILED_5120_FMA ||
            p->reduction == BN_GPU_RMSNORM_REDUCTION_TILED_5120_LOOPED_ADD_FMA ||
            p->reduction == BN_GPU_RMSNORM_REDUCTION_TILED_5120_PERSISTENT ||
            p->reduction == BN_GPU_RMSNORM_REDUCTION_TILED_5120_LOOPED ||
            p->reduction == BN_GPU_RMSNORM_REDUCTION_TILED_5120_PERSISTENT_FUSED_SQUARES) &&
           tiled_norm_5120_valid(p->dim,p->eps,p->weight_offset,p->weight_groups))) ||
        !isfinite(p->eps) || p->eps < 0.0f ||
        (p->weight_offset != 0 && p->weight_offset != 1) || set_device(r))
        return -1;
    int groups=p->weight_groups>0?p->weight_groups:1;
    if(p->weight_groups<0||groups>p->n_tokens||p->n_tokens%groups||
       (groups>1&&p->reduction!=BN_GPU_RMSNORM_REDUCTION_CONTIGUOUS_F32))return -1;
    const BnCudaBuffer *weight = (const BnCudaBuffer *)p->weight;
    if (!weight || !weight->data || !bn_quant_format_is_bf16(weight->type) ||
        weight->size / sizeof(uint16_t) < (size_t)groups*p->dim)
        return -1;
    BnGPUResidentView views[] = {p->x, p->residual, p->out, p->residual_out};
    void *data[4];
    size_t count = (size_t)p->n_tokens * p->dim;
    for (unsigned i = 0; i < 4; i++) {
        data[i] = view_data(r, views[i]);
        if (!data[i] || !bn_quant_format_is_bf16(views[i].type) || views[i].count != count)
            return -1;
    }
    for (unsigned o = 2; o < 4; o++)
        for (unsigned i = 0; i < 4; i++)
            if (o != i && overlap(views[o], views[i]))
                return -1;
    if (cudaGetLastError() != cudaSuccess) {
        r->failed = 1;
        return -1;
    }
    if (p->reduction == BN_GPU_RMSNORM_REDUCTION_TILED_5120_PERSISTENT ||
        p->reduction == BN_GPU_RMSNORM_REDUCTION_TILED_5120_LOOPED ||
        p->reduction == BN_GPU_RMSNORM_REDUCTION_TILED_5120_PERSISTENT_FUSED_SQUARES) {
        if (p->reduction != BN_GPU_RMSNORM_REDUCTION_TILED_5120_LOOPED)
            tiled_norm_5120_kernel<2,false,1><<<p->n_tokens,512,0,r->stream>>>(
                (__nv_bfloat16 *)data[2],(__nv_bfloat16 *)data[3],
                (const __nv_bfloat16 *)data[0],(const __nv_bfloat16 *)data[1],NULL,
                (const __nv_bfloat16 *)weight->data);
        else tiled_norm_5120_kernel<2,false,2><<<p->n_tokens,256,0,r->stream>>>(
                (__nv_bfloat16 *)data[2],(__nv_bfloat16 *)data[3],
                (const __nv_bfloat16 *)data[0],(const __nv_bfloat16 *)data[1],NULL,
                (const __nv_bfloat16 *)weight->data);
        if (cudaGetLastError() == cudaSuccess) return 0;
        r->failed = 1; return -1;
    }
    if (p->reduction == BN_GPU_RMSNORM_REDUCTION_TILED_5120_LOOPED_ADD_FMA) {
        tiled_norm_5120_kernel<2,true><<<p->n_tokens,256,0,r->stream>>>(
            (__nv_bfloat16 *)data[2],(__nv_bfloat16 *)data[3],
            (const __nv_bfloat16 *)data[0],(const __nv_bfloat16 *)data[1],NULL,
            (const __nv_bfloat16 *)weight->data);
        if (cudaGetLastError() == cudaSuccess) return 0;
        r->failed = 1; return -1;
    }
    if (p->reduction == BN_GPU_RMSNORM_REDUCTION_TILED_5120_FMA) {
        tiled_norm_5120_kernel<2><<<p->n_tokens,512,0,r->stream>>>(
            (__nv_bfloat16 *)data[2],(__nv_bfloat16 *)data[3],
            (const __nv_bfloat16 *)data[0],(const __nv_bfloat16 *)data[1],NULL,
            (const __nv_bfloat16 *)weight->data);
        if (cudaGetLastError() == cudaSuccess) return 0;
        r->failed = 1; return -1;
    }
    if(p->reduction==BN_GPU_RMSNORM_REDUCTION_CONTIGUOUS_F32) {
        if(groups>1) {
            launch_contiguous_norm<true,true>((__nv_bfloat16*)data[2],(__nv_bfloat16*)data[3],
                (const __nv_bfloat16*)data[0],(const __nv_bfloat16*)data[1],
                (const __nv_bfloat16*)weight->data,p->eps,p->weight_offset,
                p->n_tokens,p->dim,groups,r->stream);
        } else {
            launch_contiguous_norm<true,false>((__nv_bfloat16*)data[2],(__nv_bfloat16*)data[3],
                (const __nv_bfloat16*)data[0],(const __nv_bfloat16*)data[1],
                (const __nv_bfloat16*)weight->data,p->eps,p->weight_offset,
                p->n_tokens,p->dim,groups,r->stream);
        }
        if(cudaGetLastError()==cudaSuccess)return 0;
        r->failed=1;return -1;
    }
    int threads = cuda_rmsnorm_threads(p->dim);
    add_rmsnorm_bf16_batch_kernel<<<p->n_tokens, threads,
        (size_t)(threads / 32) * sizeof(float), r->stream>>>(
        (__nv_bfloat16 *)data[2], (__nv_bfloat16 *)data[3],
        (const __nv_bfloat16 *)data[0], (const __nv_bfloat16 *)data[1],
        (const __nv_bfloat16 *)weight->data, p->dim, p->eps, p->weight_offset);
    if (cudaGetLastError() == cudaSuccess)
        return 0;
    r->failed = 1;
    return -1;
}

static int add3_norm(void *ctx_ptr, void *ptr, const BnGPUAdd3RMSNormResidentPlan *p) {
    BnCudaCtx *ctx = (BnCudaCtx *)ctx_ptr;
    ResidentRequest *r = (ResidentRequest *)ptr;
    if (!ctx || !r || !p || r->backend_owner != ctx || ctx->compute_capability < 800 ||
        p->n_tokens <= 0 || p->dim < 2048 || p->dim > 130560 || (p->dim & 3) ||
        p->n_tokens > INT_MAX / p->dim ||
        (p->reduction != BN_GPU_RMSNORM_REDUCTION_CONTIGUOUS_F32 &&
         !((p->reduction == BN_GPU_RMSNORM_REDUCTION_TILED_5120_FMA ||
            p->reduction == BN_GPU_RMSNORM_REDUCTION_TILED_5120_LOOPED_ADD_FMA ||
            p->reduction == BN_GPU_RMSNORM_REDUCTION_TILED_5120_PERSISTENT ||
            p->reduction == BN_GPU_RMSNORM_REDUCTION_TILED_5120_LOOPED ||
            p->reduction == BN_GPU_RMSNORM_REDUCTION_TILED_5120_PERSISTENT_FUSED_SQUARES) &&
           tiled_norm_5120_valid(p->dim,p->eps,p->weight_offset,p->weight_groups))) ||
        !isfinite(p->eps) || p->eps < 0.0f ||
        (p->weight_offset != 0 && p->weight_offset != 1) || set_device(r)) return -1;
    int groups = p->weight_groups > 0 ? p->weight_groups : 1;
    if (p->weight_groups < 0 || groups > p->n_tokens || p->n_tokens % groups) return -1;
    const BnCudaBuffer *weight = (const BnCudaBuffer *)p->weight;
    if (!weight || !weight->data || !bn_quant_format_is_bf16(weight->type) ||
        weight->size / sizeof(uint16_t) < (size_t)groups * p->dim) return -1;
    BnGPUResidentView views[] = {p->x,p->residual,p->addend,p->out,p->residual_out};
    void *data[5];
    size_t count = (size_t)p->n_tokens * p->dim;
    for (unsigned i = 0; i < 5; i++) {
        data[i] = view_data(r, views[i]);
        if (!data[i] || !bn_quant_format_is_bf16(views[i].type) || views[i].count != count)
            return -1;
    }
    for (unsigned o = 3; o < 5; o++)
        for (unsigned i = 0; i < 5; i++)
            if (o != i && overlap(views[o], views[i])) return -1;
    if (cudaGetLastError() != cudaSuccess) { r->failed = 1; return -1; }
    if (p->reduction == BN_GPU_RMSNORM_REDUCTION_TILED_5120_PERSISTENT ||
        p->reduction == BN_GPU_RMSNORM_REDUCTION_TILED_5120_LOOPED ||
        p->reduction == BN_GPU_RMSNORM_REDUCTION_TILED_5120_PERSISTENT_FUSED_SQUARES) {
        if (p->reduction == BN_GPU_RMSNORM_REDUCTION_TILED_5120_PERSISTENT)
            tiled_norm_5120_kernel<3,false,1><<<p->n_tokens,512,0,r->stream>>>(
                (__nv_bfloat16 *)data[3],(__nv_bfloat16 *)data[4],
                (const __nv_bfloat16 *)data[0],(const __nv_bfloat16 *)data[1],
                (const __nv_bfloat16 *)data[2],(const __nv_bfloat16 *)weight->data);
        else if (p->reduction == BN_GPU_RMSNORM_REDUCTION_TILED_5120_PERSISTENT_FUSED_SQUARES)
            tiled_norm_5120_kernel<3,false,3><<<p->n_tokens,512,0,r->stream>>>(
                (__nv_bfloat16 *)data[3],(__nv_bfloat16 *)data[4],
                (const __nv_bfloat16 *)data[0],(const __nv_bfloat16 *)data[1],
                (const __nv_bfloat16 *)data[2],(const __nv_bfloat16 *)weight->data);
        else tiled_norm_5120_kernel<3,false,2><<<p->n_tokens,256,0,r->stream>>>(
                (__nv_bfloat16 *)data[3],(__nv_bfloat16 *)data[4],
                (const __nv_bfloat16 *)data[0],(const __nv_bfloat16 *)data[1],
                (const __nv_bfloat16 *)data[2],(const __nv_bfloat16 *)weight->data);
        if (cudaGetLastError() == cudaSuccess) return 0;
        r->failed = 1; return -1;
    }
    if (p->reduction == BN_GPU_RMSNORM_REDUCTION_TILED_5120_FMA ||
        p->reduction == BN_GPU_RMSNORM_REDUCTION_TILED_5120_LOOPED_ADD_FMA) {
        tiled_norm_5120_kernel<3><<<p->n_tokens,512,0,r->stream>>>(
            (__nv_bfloat16 *)data[3],(__nv_bfloat16 *)data[4],
            (const __nv_bfloat16 *)data[0],(const __nv_bfloat16 *)data[1],
            (const __nv_bfloat16 *)data[2],(const __nv_bfloat16 *)weight->data);
        if (cudaGetLastError() == cudaSuccess) return 0;
        r->failed = 1; return -1;
    }
    if (groups > 1) {
        launch_contiguous_norm<true,true,true>((__nv_bfloat16 *)data[3],(__nv_bfloat16 *)data[4],
            (const __nv_bfloat16 *)data[0],(const __nv_bfloat16 *)data[1],
            (const __nv_bfloat16 *)weight->data,p->eps,p->weight_offset,
            p->n_tokens,p->dim,groups,r->stream,(const __nv_bfloat16 *)data[2]);
    } else {
        launch_contiguous_norm<true,false,true>((__nv_bfloat16 *)data[3],(__nv_bfloat16 *)data[4],
            (const __nv_bfloat16 *)data[0],(const __nv_bfloat16 *)data[1],
            (const __nv_bfloat16 *)weight->data,p->eps,p->weight_offset,
            p->n_tokens,p->dim,groups,r->stream,(const __nv_bfloat16 *)data[2]);
    }
    if (cudaGetLastError() == cudaSuccess) return 0;
    r->failed = 1;
    return -1;
}

static int prepare(void *ctx_ptr, void *ptr, const BnGPUGDNPrepareResidentPlan *p) {
    BnCudaCtx *ctx = (BnCudaCtx *)ctx_ptr;
    ResidentRequest *r = (ResidentRequest *)ptr;
    if (!ctx || !r || !p || r->backend_owner != ctx || ctx->compute_capability < 800 ||
        p->n_tokens <= 0 || p->hk <= 0 || p->hv <= 0 || p->dk <= 0 || p->dk > 256 || p->dv <= 0 ||
        p->dv > 256 || !isfinite(p->norm_eps) || p->norm_eps <= 0.0f ||
        !isfinite(p->softplus_threshold) || p->softplus_threshold < 0.0f || set_device(r))
        return -1;
    int bk = 1, bv = 1;
    while (bk < p->dk)
        bk *= 2;
    while (bv < p->dv)
        bv *= 2;
    int64_t dim = 2 * (int64_t)p->hk * p->dk + (int64_t)p->hv * p->dv;
    if (bk != bv || dim > INT_MAX || p->n_tokens > INT_MAX / dim)
        return -1;
    const BnCudaBuffer *a = (const BnCudaBuffer *)p->a_log;
    const BnCudaBuffer *bias = (const BnCudaBuffer *)p->dt_bias;
    if (!a || !a->data || !bias || !bias->data || !bn_quant_format_is_f32(a->type) ||
        !bn_quant_format_is_bf16(bias->type) || a->size / sizeof(float) < (size_t)p->hv ||
        bias->size / sizeof(uint16_t) < (size_t)p->hv)
        return -1;
    size_t nq = (size_t)p->n_tokens * p->hk * p->dk;
    size_t nv = (size_t)p->n_tokens * p->hv * p->dv, ng = (size_t)p->n_tokens * p->hv;
    BnGPUResidentView views[] = {p->x, p->ba, p->q, p->k, p->v, p->log_decay, p->beta};
    size_t counts[] = {(size_t)p->n_tokens * dim, 2 * ng, nq, nq, nv, ng, ng};
    void *data[7];
    for (unsigned i = 0; i < 7; i++) {
        data[i] = view_data(r, views[i]);
        if (!data[i] || views[i].count != counts[i] ||
            (i < 5 ? !bn_quant_format_is_bf16(views[i].type)
                   : !bn_quant_format_is_f32(views[i].type)))
            return -1;
    }
    for (unsigned o = 2; o < 7; o++)
        for (unsigned i = 0; i < 7; i++)
            if (o != i && overlap(views[o], views[i]))
                return -1;
    if (cudaGetLastError() != cudaSuccess) {
        r->failed = 1;
        return -1;
    }
    unsigned blocks = (unsigned)(((size_t)p->n_tokens - 1) / 8 + 1) * (unsigned)(p->hk + p->hv);
#define BN_CUDA_GDN_RESIDENT_PREP(B)                                                               \
    gdn_prefill_prepare_bf16_kernel<B, true><<<blocks, 128, 0, r->stream>>>(                       \
        NULL, (float *)data[5], (float *)data[6], (const __nv_bfloat16 *)data[0],                  \
        (const __nv_bfloat16 *)data[1], (const float *)a->data, (const __nv_bfloat16 *)bias->data, \
        p->n_tokens, p->hk, p->dk, p->hv, p->dv, p->norm_eps, p->softplus_threshold,               \
        (__nv_bfloat16 *)data[2], (__nv_bfloat16 *)data[3], (__nv_bfloat16 *)data[4])
    if (bk <= 32) {
        BN_CUDA_GDN_RESIDENT_PREP(32);
    } else if (bk == 64) {
        BN_CUDA_GDN_RESIDENT_PREP(64);
    } else if (bk == 128) {
        BN_CUDA_GDN_RESIDENT_PREP(128);
    } else {
        BN_CUDA_GDN_RESIDENT_PREP(256);
    }
#undef BN_CUDA_GDN_RESIDENT_PREP
    if (cudaGetLastError() == cudaSuccess)
        return 0;
    r->failed = 1;
    return -1;
}

static int decode_validate(void *ctx_ptr, void *ptr, const BnGPUGDNDecodeResidentPlan *p,
    void **data, size_t *out_no, int *out_sequences,
    ResidentRequest *activation_owner = NULL) {
    BnCudaCtx *ctx = (BnCudaCtx *)ctx_ptr;
    ResidentRequest *r = (ResidentRequest *)ptr;
    if (!ctx || !r || !p || r->backend_owner != ctx || ctx->compute_capability < 800 ||
        p->hk <= 0 || p->hv <= 0 || p->hv % p->hk || p->dk != 128 || p->dv != 128 ||
        (p->profile!=BN_GPU_GDN_DECODE_PACKED&&p->profile!=BN_GPU_GDN_DECODE_RSQRT) ||
        (p->profile==BN_GPU_GDN_DECODE_RSQRT&&ctx->compute_capability!=1200) ||
        p->hv > INT_MAX / 128 || p->n_sequences < 0 || p->n_sequences > 65535 || !isfinite(p->q_scale) || p->q_scale <= 0.0f || set_device(r))
        return -1;
    int64_t dim = 2 * (int64_t)p->hk * 128 + (int64_t)p->hv * 128;
    if (dim > INT_MAX)
        return -1;
    const BnCudaBuffer *a = (const BnCudaBuffer *)p->a_log;
    const BnCudaBuffer *bias = (const BnCudaBuffer *)p->dt_bias;
    if (!a || !a->data || !bias || !bias->data || !bn_quant_format_is_f32(a->type) ||
        !bn_quant_format_is_bf16(bias->type) || a->size / sizeof(float) < (size_t)p->hv ||
        bias->size / sizeof(uint16_t) < (size_t)p->hv)
        return -1;
    size_t no = (size_t)p->hv * 128, ns = no * 128;
    int sequences = p->n_sequences ? p->n_sequences : 1;
    if (ns > SIZE_MAX / (size_t)sequences) return -1;
    BnGPUResidentView views[] = {p->x, p->ba, p->initial_state, p->next_state, p->out};
    size_t counts[] = {(size_t)dim, (size_t)p->hv * 2, ns, ns, no};
    for (unsigned i = 0; i < 5; i++) counts[i] *= (size_t)sequences;
    for (unsigned i = 0; i < 5; i++) {
        ResidentRequest *owner = r;
        if (activation_owner && (i == 0 || i == 1 || i == 4) &&
            views[i].request == activation_owner) owner = activation_owner;
        data[i] = view_data(owner, views[i]);
        if (!data[i] || views[i].count != counts[i] ||
            (i == 2 || i == 3 ? !bn_quant_format_is_f32(views[i].type)
                              : !bn_quant_format_is_bf16(views[i].type)))
            return -1;
    }
    for (unsigned o = 3; o < 5; o++)
        for (unsigned i = 0; i < 5; i++)
            if (o != i && overlap(views[o], views[i]))
                return -1;
    *out_no = no; *out_sequences = sequences;
    return 0;
}

static int decode(void *ctx_ptr, void *ptr, const BnGPUGDNDecodeResidentPlan *p) {
    ResidentRequest *r = (ResidentRequest *)ptr;
    void *data[5]; size_t no; int sequences;
    if (decode_validate(ctx_ptr, ptr, p, data, &no, &sequences)) return -1;
    const BnCudaBuffer *a = (const BnCudaBuffer *)p->a_log;
    const BnCudaBuffer *bias = (const BnCudaBuffer *)p->dt_bias;
    if (cudaGetLastError() != cudaSuccess) {
        r->failed = 1;
        return -1;
    }
#define BN_GDN_RESIDENT_LAUNCH(BATCH,RSQRT) \
    gdn_decode_bf16_kernel<BATCH,false,RSQRT><<<dim3((unsigned)((no+3)/4),(unsigned)sequences),128,0,r->stream>>>( \
        (__nv_bfloat16*)data[4],(float*)data[3],(const float*)data[2], \
        (const __nv_bfloat16*)data[0],(const __nv_bfloat16*)data[1],(const float*)a->data, \
        (const __nv_bfloat16*)bias->data,p->hk,p->hv,p->q_scale)
    if(sequences>1) {
        if(p->profile==BN_GPU_GDN_DECODE_RSQRT){BN_GDN_RESIDENT_LAUNCH(true,true);}
        else {BN_GDN_RESIDENT_LAUNCH(true,false);}
    } else {
        if(p->profile==BN_GPU_GDN_DECODE_RSQRT){BN_GDN_RESIDENT_LAUNCH(false,true);}
        else {BN_GDN_RESIDENT_LAUNCH(false,false);}
    }
#undef BN_GDN_RESIDENT_LAUNCH
    if (cudaGetLastError() == cudaSuccess)
        return 0;
    r->failed = 1;
    return -1;
}

template<typename Row>
static size_t sequence_group_workspace_size(void *ctx_ptr, int capacity, int slots=1) {
    const BnCudaCtx *ctx = (const BnCudaCtx *)ctx_ptr;
    return ctx && ctx->compute_capability >= 800 && capacity > 0 && capacity <= 256 && slots>0 && slots<=64
        ? sizeof(SequenceGroupWorkspace<Row>) + (size_t)capacity *
            ((size_t)(slots+1)*sizeof(Row)+sizeof(cudaEvent_t)) + (size_t)slots*sizeof(cudaEvent_t) : 0;
}
static size_t decode_group_workspace_size(void *ctx_ptr, int capacity) {
    return sequence_group_workspace_size<BnCudaSequenceRow>(ctx_ptr,capacity);
}
template<typename Row>
static int group_reserve(void *ctx_ptr, void *ptr, int capacity, size_t budget, SequenceGroupWorkspace<Row> **storage, int slots=1) {
    ResidentRequest *r = (ResidentRequest *)ptr;
    size_t bytes = sequence_group_workspace_size<Row>(ctx_ptr, capacity, slots);
    if (!r || r->backend_owner != ctx_ptr || !bytes || bytes > budget || set_device(r)) return -1;
    if (*storage) return (*storage)->capacity == capacity && (*storage)->slots == slots ? 0 : -1;
    SequenceGroupWorkspace<Row> *w = (SequenceGroupWorkspace<Row> *)calloc(1, sizeof(*w));
    if (!w) return -1;
    w->capacity = capacity;w->slots=slots;
    w->copied=(cudaEvent_t*)calloc((size_t)slots,sizeof(*w->copied));
    w->ready = (cudaEvent_t *)calloc((size_t)capacity, sizeof(*w->ready));
    cudaError_t error = w->ready&&w->copied ? cudaSuccess : cudaErrorMemoryAllocation;
    if (error == cudaSuccess) error = cudaHostAlloc((void **)&w->host,
        (size_t)capacity * slots * sizeof(*w->host), cudaHostAllocDefault);
    if (error == cudaSuccess) error = cudaMalloc((void **)&w->device,
        (size_t)capacity * sizeof(*w->device));
    for (int i = 0; i < capacity && error == cudaSuccess; i++)
        error = cudaEventCreateWithFlags(&w->ready[i], cudaEventDisableTiming);
    for(int i=0;i<slots&&error==cudaSuccess;i++)
        error=cudaEventCreateWithFlags(&w->copied[i],cudaEventDisableTiming);
    if (error == cudaSuccess) error = cudaEventCreateWithFlags(&w->done, cudaEventDisableTiming);
    if (error != cudaSuccess) {
        free_decode_group(w);
        if (error != cudaErrorMemoryAllocation) r->failed = 1;
        (void)cudaGetLastError(); return -1;
    }
    *storage = w;
    return 0;
}
static int decode_group_reserve(void *ctx, void *ptr, int capacity, size_t budget) {
    ResidentRequest *r = (ResidentRequest *)ptr;
    return r ? group_reserve(ctx,ptr,capacity,budget,&r->decode_group) : -1;
}
static int conv_group_reserve(void *ctx, void *ptr, int capacity, size_t budget) {
    ResidentRequest *r = (ResidentRequest *)ptr;
    return r ? group_reserve(ctx,ptr,capacity,budget,&r->conv_group) : -1;
}
static int group_fail(ResidentRequest *coordinator, ResidentRequest **participants, int count) {
    (void)cudaStreamSynchronize(coordinator->stream);
    coordinator->failed=1;
    for (int i=0;i<count;i++) participants[i]->failed=1;
    return -1;
}
template<typename Row>
static int group_prepare(ResidentRequest *coordinator, SequenceGroupWorkspace<Row> *w,
    ResidentRequest **participants, const Row *rows, int count) {
    if (set_device(coordinator)) return -1;
    int slot=w->cursor;Row *source=w->host+(size_t)slot*w->capacity;
    if (w->filled==w->slots) {
        cudaError_t status = cudaEventQuery(w->copied[slot]);
        if (status == cudaErrorNotReady) {
            cudaError_t last = cudaGetLastError();
            if (last != cudaSuccess && last != cudaErrorNotReady) goto fail;
            return BN_GPU_GROUP_BUSY;
        }
        if (status != cudaSuccess) goto fail;
    }
    if (cudaGetLastError() != cudaSuccess) goto fail;
    memcpy(source,rows,(size_t)count*sizeof(*rows));
    for (int i = 0; i < count; i++) {
        if (cudaEventRecord(w->ready[i],participants[i]->stream) != cudaSuccess ||
            cudaStreamWaitEvent(coordinator->stream,w->ready[i],0) != cudaSuccess) goto fail;
    }
    if (cudaMemcpyAsync(w->device,source,(size_t)count*sizeof(*rows),
        cudaMemcpyHostToDevice,coordinator->stream) != cudaSuccess ||
        cudaEventRecord(w->copied[slot],coordinator->stream) != cudaSuccess) goto fail;
    w->cursor=(slot+1)%w->slots;if(w->filled<w->slots)w->filled++;
    return 0;
fail:
    return group_fail(coordinator,participants,count);
}
template<typename Row>
static int group_finish(ResidentRequest *coordinator, SequenceGroupWorkspace<Row> *w,
    ResidentRequest **participants, int count) {
    if (cudaGetLastError() != cudaSuccess ||
        cudaEventRecord(w->done,coordinator->stream) != cudaSuccess)
        return group_fail(coordinator,participants,count);
    for (int i=0;i<count;i++)
        if (cudaStreamWaitEvent(participants[i]->stream,w->done,0) != cudaSuccess)
            return group_fail(coordinator,participants,count);
    return 0;
}
static int decode_group(void *ctx_ptr, void *ptr, const BnGPUGDNDecodeGroupPlan *p) {
    ResidentRequest *coordinator = (ResidentRequest *)ptr;
    if (!coordinator || coordinator->backend_owner != ctx_ptr || coordinator->failed || !p ||
        !p->sequences || p->n_sequences <= 0 || p->n_sequences > 256 ||
        !coordinator->decode_group || p->n_sequences > coordinator->decode_group->capacity)
        return -1;
    DecodeGroupWorkspace *w = coordinator->decode_group;
    ResidentRequest *participants[256];
    BnCudaSequenceRow rows[256];
    size_t no = 0; int single = 0;
    /* Validate the entire plan before touching pinned descriptors or queues. */
    for (int i = 0; i < p->n_sequences; i++) {
        const BnGPUGDNDecodeSequence *s = &p->sequences[i];
        ResidentRequest *r = (ResidentRequest *)s->request;
        if (!r || r == coordinator || r->backend_owner != ctx_ptr ||
            r->device != coordinator->device) return -1;
        for (int j = 0; j < i; j++) if (participants[j] == r) return -1;
        BnGPUGDNDecodeResidentPlan one = {};
        one.hk=p->hk; one.dk=p->dk; one.hv=p->hv; one.dv=p->dv;
        one.profile=p->profile;one.q_scale=p->q_scale; one.a_log=p->a_log; one.dt_bias=p->dt_bias;
        one.x=s->x; one.ba=s->ba; one.initial_state=s->initial_state;
        one.next_state=s->next_state; one.out=s->out;
        void *data[5];
        if (decode_validate(ctx_ptr,r,&one,data,&no,&single,coordinator)) return -1;
        rows[i].x=(const __nv_bfloat16 *)data[0]; rows[i].ba=(const __nv_bfloat16 *)data[1];
        rows[i].initial_state=(const float *)data[2]; rows[i].state=(float *)data[3];
        rows[i].out=(__nv_bfloat16 *)data[4]; participants[i]=r;
    }
    /* Shared activation buffers can alias across rows. Every write must be
     * disjoint from every other row's inputs and writes before queueing work. */
    for (int i = 0; i < p->n_sequences; i++) {
        const BnGPUGDNDecodeSequence *a = &p->sequences[i];
        BnGPUResidentView writes[] = {a->next_state,a->out};
        for (int j = 0; j < p->n_sequences; j++) if (i != j) {
            const BnGPUGDNDecodeSequence *b = &p->sequences[j];
            BnGPUResidentView all[] = {b->x,b->ba,b->initial_state,b->next_state,b->out};
            for (unsigned w = 0; w < 2; w++)
                for (unsigned v = 0; v < 5; v++)
                    if (overlap(writes[w],all[v])) return -1;
        }
    }
    int status=group_prepare(coordinator,w,participants,rows,p->n_sequences);
    if(status) return status;
#define BN_GDN_GROUP_LAUNCH(RSQRT) \
    gdn_decode_bf16_kernel<false,true,RSQRT><<<dim3((unsigned)((no+3)/4),(unsigned)p->n_sequences),128,0,coordinator->stream>>>( \
        NULL,NULL,NULL,NULL,NULL,(const float*)((const BnCudaBuffer*)p->a_log)->data, \
        (const __nv_bfloat16*)((const BnCudaBuffer*)p->dt_bias)->data,p->hk,p->hv,p->q_scale,w->device)
    if(p->profile==BN_GPU_GDN_DECODE_RSQRT){BN_GDN_GROUP_LAUNCH(true);}
    else {BN_GDN_GROUP_LAUNCH(false);}
#undef BN_GDN_GROUP_LAUNCH
    return group_finish(coordinator,w,participants,p->n_sequences);
}

/* Eight value rows per warp; four contiguous keys per lane. Keep state in
 * registers across the chunk. Q/K sum and contracted dot helpers have distinct
 * arithmetic contracts: rounding all dot products first changes BF16 outputs. */
static __global__ void kda_recurrent_kernel(__nv_bfloat16 *out, float *next,
    const __nv_bfloat16 *x, const __nv_bfloat16 *gate, const __nv_bfloat16 *beta_raw,
    const float *parameter, const float *bias, const float *initial,
    int nt, int heads, float scale, float lower_bound, int storage) {
    int head = blockIdx.x, row0 = blockIdx.y * 8, lane = threadIdx.x;
    float state[8][4];
#pragma unroll
    for (int v = 0; v < 8; v++)
#pragma unroll
        for (int j = 0; j < 4; j++)
            state[v][j] = initial[((size_t)head * 128 + row0 + v) * 128 + lane * 4 + j];
    float amplitude = storage == BN_GPU_KDA_A_LOG ? ex(parameter[head]) : -parameter[head];
    for (int t = 0; t < nt; t++) {
        size_t token = (size_t)t * heads * 128, packed = token * 3;
        float q[4], k[4], q2[4], k2[4], decay[4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            size_t key = (size_t)head * 128 + lane * 4 + j;
            q[j] = __bfloat162float(x[packed + key]);
            k[j] = __bfloat162float(x[packed + (size_t)heads * 128 + key]);
            q2[j] = __fmul_rn(q[j], q[j]); k2[j] = __fmul_rn(k[j], k[j]);
        }
        float qr, kr, qs = __fadd_rn(cuda_gdn_decode_sum(q2), 1e-6f);
        float ks = __fadd_rn(cuda_gdn_decode_sum(k2), 1e-6f);
        asm("sqrt.approx.ftz.f32 %0,%1;" : "=f"(qr) : "f"(qs));
        asm("sqrt.approx.ftz.f32 %0,%1;" : "=f"(kr) : "f"(ks));
#pragma unroll
        for (int j = 0; j < 4; j++) {
            q[j] = __fmul_rn(cuda_gdn_prep_div(q[j], qr), scale);
            k[j] = cuda_gdn_prep_div(k[j], kr);
            size_t key = (size_t)head * 128 + lane * 4 + j;
            float shifted = __fadd_rn(__bfloat162float(gate[token + key]), bias[key]);
            float log_decay = cuda_gdn_prep_div(lower_bound,
                __fadd_rn(1.0f, ex(-__fmul_rn(amplitude, shifted))));
            decay[j] = ex(log_decay);
        }
        float beta = cuda_gdn_prep_div(1.0f,
            __fadd_rn(1.0f, ex(-__bfloat162float(beta_raw[(size_t)t * heads + head]))));
#pragma unroll
        for (int v = 0; v < 8; v++) {
#pragma unroll
            for (int j = 0; j < 4; j++) state[v][j] = __fmul_rn(state[v][j], decay[j]);
            float value = __bfloat162float(x[packed + (size_t)2 * heads * 128 +
                (size_t)head * 128 + row0 + v]);
            float delta = __fmul_rn(__fsub_rn(value, cuda_gdn_decode_dot4(state[v], k)), beta);
#pragma unroll
            for (int j = 0; j < 4; j++) state[v][j] = __fmaf_rn(delta, k[j], state[v][j]);
            float y = cuda_gdn_decode_dot4(state[v], q);
            if (!lane) out[token + (size_t)head * 128 + row0 + v] = __float2bfloat16_rn(y);
        }
    }
#pragma unroll
    for (int v = 0; v < 8; v++)
#pragma unroll
        for (int j = 0; j < 4; j++)
            next[((size_t)head * 128 + row0 + v) * 128 + lane * 4 + j] = state[v][j];
}

static int kda_recurrent(void *ctx_ptr, void *ptr, const BnGPUKDARecurrentResidentPlan *p) {
    BnCudaCtx *ctx = (BnCudaCtx *)ctx_ptr;
    ResidentRequest *r = (ResidentRequest *)ptr;
    if (!ctx || !r || !p || r->backend_owner != ctx || ctx->compute_capability < 800 ||
        p->head_dim != 128 || p->num_heads <= 0 || p->num_heads > INT_MAX / 384 ||
        p->n_tokens <= 0 || p->n_tokens > INT_MAX / (p->num_heads * 384) ||
        (size_t)p->num_heads > SIZE_MAX / 16384 / sizeof(float) ||
        !isfinite(p->q_scale) || p->q_scale <= 0.0f ||
        !isfinite(p->lower_bound) || p->lower_bound >= 0.0f ||
        (p->parameter_storage != BN_GPU_KDA_A_LOG &&
         p->parameter_storage != BN_GPU_KDA_A_NEGATIVE_EXP) || set_device(r)) return -1;
    size_t nh = p->num_heads, n = (size_t)p->n_tokens * nh * 128, ns = nh * 16384;
    const BnCudaBuffer *a = (const BnCudaBuffer *)p->decay_parameter;
    const BnCudaBuffer *bias = (const BnCudaBuffer *)p->gate_bias;
    if (!a || !a->data || !bias || !bias->data ||
        !bn_quant_format_is_f32(a->type) || !bn_quant_format_is_f32(bias->type) ||
        a->rows != p->num_heads || a->cols != 1 || a->size / sizeof(float) < nh ||
        bias->rows != p->num_heads || bias->cols != 128 || bias->size / sizeof(float) < nh * 128)
        return -1;
    BnGPUResidentView views[] = {p->x, p->gate, p->beta_raw, p->initial_state, p->next_state, p->out};
    size_t counts[] = {n * 3, n, (size_t)p->n_tokens * nh, ns, ns, n};
    void *data[6];
    for (unsigned i = 0; i < 6; i++) {
        data[i] = view_data(r, views[i]);
        if (!data[i] || views[i].count != counts[i] ||
            ((i == 3 || i == 4) ? !bn_quant_format_is_f32(views[i].type) :
                                  !bn_quant_format_is_bf16(views[i].type))) return -1;
    }
    for (unsigned o = 4; o < 6; o++)
        for (unsigned i = 0; i < 6; i++)
            if (o != i && overlap(views[o], views[i])) return -1;
    if (cudaGetLastError() != cudaSuccess) { r->failed = 1; return -1; }
    kda_recurrent_kernel<<<dim3((unsigned)nh, 16), 32, 0, r->stream>>>(
        (__nv_bfloat16 *)data[5], (float *)data[4], (const __nv_bfloat16 *)data[0],
        (const __nv_bfloat16 *)data[1], (const __nv_bfloat16 *)data[2],
        (const float *)a->data, (const float *)bias->data, (const float *)data[3],
        p->n_tokens, p->num_heads, p->q_scale, p->lower_bound, p->parameter_storage);
    if (cudaGetLastError() == cudaSuccess) return 0;
    r->failed = 1;
    return -1;
}

static int conv_validate(void *ctx_ptr, void *ptr, const BnGPUCausalConvResidentPlan *p,
    void **data, unsigned *out_tiles, int *out_threads, ResidentRequest *activation_owner = NULL,
    bool f32_weight = false) {
    BnCudaCtx *ctx = (BnCudaCtx *)ctx_ptr;
    ResidentRequest *r = (ResidentRequest *)ptr;
    if (!ctx || !r || !p || r->backend_owner != ctx || ctx->compute_capability < 800 ||
        p->n_tokens <= 0 || p->dim <= 0 || p->kernel < 2 || p->kernel > 4 ||
        p->dim > INT_MAX / p->kernel || p->n_tokens > INT_MAX / p->dim || set_device(r))
        return -1;
    const BnCudaBuffer *w = (const BnCudaBuffer *)p->weight;
    if (!w || !w->data ||
        (f32_weight ? w->type != BN_GGUF_TENSOR_F32 : !bn_quant_format_is_bf16(w->type)) ||
        w->rows != p->dim || w->cols != p->kernel ||
        w->size / (f32_weight ? sizeof(float) : sizeof(uint16_t)) < (size_t)p->dim * p->kernel)
        return -1;
    size_t n = (size_t)p->n_tokens * p->dim, ns = (size_t)(p->kernel - 1) * p->dim;
    BnGPUResidentView views[] = {p->x, p->initial_history, p->next_history, p->out};
    size_t counts[] = {n, ns, ns, n};
    for (unsigned i = 0; i < 4; i++) {
        ResidentRequest *owner=r;
        if(activation_owner&&(i==0||i==3)&&views[i].request==activation_owner) owner=activation_owner;
        data[i] = view_data(owner, views[i]);
        if (!data[i] || views[i].count != counts[i] || !bn_quant_format_is_bf16(views[i].type))
            return -1;
    }
    for (unsigned o = 2; o < 4; o++)
        for (unsigned i = 0; i < 4; i++)
            if (o != i && overlap(views[o], views[i]))
                return -1;
    int threads = p->n_tokens <= 8 ? 64 : 128;
    size_t tiles = (((size_t)p->dim - 1) / threads + 1) * (((size_t)p->n_tokens - 1) / 8 + 1);
    if (tiles > INT_MAX)
        return -1;
    *out_tiles=(unsigned)tiles; *out_threads=threads;
    return 0;
}
static int convolve(void *ctx_ptr, void *ptr, const BnGPUCausalConvResidentPlan *p) {
    ResidentRequest *r=(ResidentRequest *)ptr;void *data[4];unsigned tiles;int threads;
    if(conv_validate(ctx_ptr,ptr,p,data,&tiles,&threads)) return -1;
    const BnCudaBuffer *w=(const BnCudaBuffer *)p->weight;
    if (cudaGetLastError() != cudaSuccess) {
        r->failed = 1;
        return -1;
    }
    causal_conv_silu_bf16_batch_kernel<false><<<(unsigned)tiles, threads, 0, r->stream>>>(
        (__nv_bfloat16 *)data[3], (__nv_bfloat16 *)data[2], (const __nv_bfloat16 *)data[0],
        (const __nv_bfloat16 *)data[1], (const __nv_bfloat16 *)w->data, p->n_tokens, p->dim,
        p->kernel);
    if (cudaGetLastError() == cudaSuccess)
        return 0;
    r->failed = 1;
    return -1;
}

static int convolve_f32_weight(void *ctx_ptr, void *ptr,
    const BnGPUCausalConvResidentPlan *p) {
    ResidentRequest *r = (ResidentRequest *)ptr;
    void *data[4]; unsigned tiles; int threads;
    if (conv_validate(ctx_ptr, ptr, p, data, &tiles, &threads, NULL, true)) return -1;
    const BnCudaBuffer *w = (const BnCudaBuffer *)p->weight;
    if (cudaGetLastError() != cudaSuccess) {
        r->failed = 1;
        return -1;
    }
    causal_conv_silu_bf16_batch_kernel<false,true><<<tiles, threads, 0, r->stream>>>(
        (__nv_bfloat16 *)data[3], (__nv_bfloat16 *)data[2],
        (const __nv_bfloat16 *)data[0], (const __nv_bfloat16 *)data[1],
        NULL, p->n_tokens, p->dim, p->kernel, NULL, (const float *)w->data);
    if (cudaGetLastError() == cudaSuccess) return 0;
    r->failed = 1;
    return -1;
}

static int conv_group_impl(void *ctx_ptr,void *ptr,const BnGPUCausalConvGroupPlan *p,
    const int *token_counts) {
    ResidentRequest *coordinator=(ResidentRequest *)ptr;
    if(!coordinator||coordinator->backend_owner!=ctx_ptr||coordinator->failed||!p||
       !p->sequences||p->n_sequences<=0||p->n_sequences>256||!coordinator->conv_group||
       p->n_sequences>coordinator->conv_group->capacity) return -1;
    DecodeGroupWorkspace *w=coordinator->conv_group;
    ResidentRequest *participants[256];BnCudaSequenceRow rows[256];unsigned tiles[256];int threads[256];bool all_one=true;
    for(int i=0;i<p->n_sequences;i++) {
        const BnGPUCausalConvSequence *v=&p->sequences[i];ResidentRequest *r=(ResidentRequest *)v->request;
        if(!r||r==coordinator||r->backend_owner!=ctx_ptr||r->device!=coordinator->device) return -1;
        for(int j=0;j<i;j++)if(participants[j]==r)return -1;
        BnGPUCausalConvResidentPlan one={};one.n_tokens=token_counts?token_counts[i]:1;one.dim=p->dim;one.kernel=p->kernel;one.weight=p->weight;
        one.x=v->x;one.out=v->out;one.initial_history=v->initial_history;one.next_history=v->next_history;
        void *data[4];if(conv_validate(ctx_ptr,r,&one,data,&tiles[i],&threads[i],coordinator)) return -1;
        if(one.n_tokens!=1)all_one=false;
        rows[i].x=data[0];rows[i].initial_state=data[1];rows[i].state=data[2];rows[i].out=data[3];rows[i].ba=NULL;participants[i]=r;
    }
    for(int i=0;i<p->n_sequences;i++) {
        const BnGPUCausalConvSequence *a=&p->sequences[i];BnGPUResidentView writes[]={a->next_history,a->out};
        for(int j=0;j<p->n_sequences;j++)if(i!=j) {
            const BnGPUCausalConvSequence *b=&p->sequences[j];BnGPUResidentView all[]={b->x,b->initial_history,b->next_history,b->out};
            for(unsigned o=0;o<2;o++)for(unsigned v=0;v<4;v++)if(overlap(writes[o],all[v]))return -1;
        }
    }
    int status=group_prepare(coordinator,w,participants,rows,p->n_sequences);if(status)return status;
    const BnCudaBuffer *weight=(const BnCudaBuffer *)p->weight;
    if(all_one) {
        causal_conv_silu_bf16_batch_kernel<true><<<dim3(tiles[0],(unsigned)p->n_sequences),threads[0],0,coordinator->stream>>>(
            NULL,NULL,NULL,NULL,(const __nv_bfloat16 *)weight->data,1,p->dim,p->kernel,w->device);
    } else for(int i=0;i<p->n_sequences;i++) {
        causal_conv_silu_bf16_batch_kernel<false><<<tiles[i],threads[i],0,coordinator->stream>>>(
            (__nv_bfloat16 *)rows[i].out,(__nv_bfloat16 *)rows[i].state,
            (const __nv_bfloat16 *)rows[i].x,(const __nv_bfloat16 *)rows[i].initial_state,
            (const __nv_bfloat16 *)weight->data,token_counts[i],p->dim,p->kernel);
        if(cudaGetLastError()!=cudaSuccess)return group_fail(coordinator,participants,p->n_sequences);
    }
    return group_finish(coordinator,w,participants,p->n_sequences);
}

static int conv_group(void *ctx,void *ptr,const BnGPUCausalConvGroupPlan *p) {
    return conv_group_impl(ctx,ptr,p,NULL);
}
static int conv_prefill_group(void *ctx,void *ptr,const BnGPUCausalConvPrefillGroupPlan *p) {
    if(!p || !p->token_counts)return -1;
    return conv_group_impl(ctx,ptr,&p->group,p->token_counts);
}

/* Match the reference one-warp row layouts:4 contiguous values/lane for
 * one row,8 values/lane for two rows, with a second row replica for four. */
static __device__ float glm_output_square_sum8(const float *v) {
    float even = __fadd_rn(__fmaf_rn(v[0], v[0], __fmul_rn(v[2], v[2])),
                          __fmaf_rn(v[4], v[4], __fmul_rn(v[6], v[6])));
    float odd = __fadd_rn(__fmaf_rn(v[1], v[1], __fmul_rn(v[3], v[3])),
                         __fmaf_rn(v[5], v[5], __fmul_rn(v[7], v[7])));
    return __fadd_rn(even, odd);
}

template <int R, bool Compiled = false, bool Glm = false>
static __global__ void gated_norm_kernel(__nv_bfloat16 *out, const __nv_bfloat16 *x,
                                         const __nv_bfloat16 *z, const __nv_bfloat16 *weight,
                                         int rows, float eps, int activation,
                                         const float *weight_f32 = NULL) {
    enum { lanes = R == 1 ? 32 : 16, width = 128 / lanes, replicas = R == 4 ? 2 : 1 };
    int lane = threadIdx.x % lanes, row0 = blockIdx.x * R + threadIdx.x / lanes;
#pragma unroll
    for (int rep = 0; rep < replicas; rep++) {
        int row = row0 + rep * 2;
        size_t base = (size_t)row * 128;
        float values[width], even[width / 2], odd[width / 2];
#pragma unroll
        for (int g = 0; g < width / 2; g++) {
            int c = lane * width + g * 2;
            values[g * 2] = row < rows ? __bfloat162float(x[base + c]) : 0.0f;
            values[g * 2 + 1] = row < rows ? __bfloat162float(x[base + c + 1]) : 0.0f;
            even[g] = __fmul_rn(values[g * 2], values[g * 2]);
            odd[g] = __fmul_rn(values[g * 2 + 1], values[g * 2 + 1]);
        }
#pragma unroll
        for (int step = 1; step < width / 2; step *= 2) {
#pragma unroll
            for (int g = 0; g < width / 2; g += step * 2) {
                even[g] = __fadd_rn(even[g], even[g + step]);
                odd[g] = __fadd_rn(odd[g], odd[g + step]);
            }
        }
        float sum = Glm ? glm_output_square_sum8(values) : __fadd_rn(even[0], odd[0]);
#pragma unroll
        for (int delta = lanes / 2; delta; delta >>= 1)
            sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, delta, lanes));
        float variance = __fadd_rn(cuda_gdn_prep_div(sum, 128.0f), eps), inv;
        if (Glm) {
            float root;
            asm("sqrt.approx.ftz.f32 %0,%1;" : "=f"(root) : "f"(variance));
            inv = cuda_gdn_prep_div(1.0f, root);
        } else {
            asm("rsqrt.approx.ftz.f32 %0,%1;" : "=f"(inv) : "f"(variance));
        }
#pragma unroll
        for (int j = 0; j < width; j++) {
            int c = lane * width + j;
            float gate = row < rows ? __bfloat162float(z[base + c]) : 0.0f;
            if (Compiled) {
                /* Preserve the selected compiled FP32 SiLU contract: a
                 * libdevice exponential followed by direct division, rather
                 * than a rounded reciprocal multiplied by the input. */
                gate = cuda_gdn_prep_div(gate, __fadd_rn(1.0f, expf(-gate)));
            } else {
                float sigmoid = cuda_gdn_prep_div(1.0f, __fadd_rn(1.0f, ex(-gate)));
                if (activation == BN_GPU_GATED_NORM_SILU)
                    gate = __fmul_rn(gate, sigmoid);
                else
                    gate = sigmoid;
            }
            float gamma = Glm ? weight_f32[c] : __bfloat162float(weight[c]);
            float value = __fmul_rn(__fmul_rn(values[j], inv), gamma);
            if (row < rows)
                out[base + c] = __float2bfloat16_rn(__fmul_rn(value, gate));
        }
    }
}

template <bool Glm>
static int gated_norm_impl(void *ctx_ptr, void *ptr, const BnGPUGatedRMSNormResidentPlan *p) {
    BnCudaCtx *ctx = (BnCudaCtx *)ctx_ptr;
    ResidentRequest *r = (ResidentRequest *)ptr;
    if (!ctx || !r || !p || r->backend_owner != ctx || ctx->compute_capability < 800 ||
        ctx->row_reduction_max_blocks <= 0 || p->rows <= 0 || p->dim != 128 ||
        p->rows > INT_MAX / 128 || !isfinite(p->eps) || p->eps <= 0.0f ||
        (p->activation != BN_GPU_GATED_NORM_SILU && p->activation != BN_GPU_GATED_NORM_SIGMOID) ||
        (Glm ? (p->profile != BN_GPU_GATED_NORM_GLM_R2 || p->activation != BN_GPU_GATED_NORM_SIGMOID) :
         (p->profile != BN_GPU_GATED_NORM_FLA && p->profile != BN_GPU_GATED_NORM_COMPILED_R2)) ||
        (p->profile == BN_GPU_GATED_NORM_COMPILED_R2 && p->activation != BN_GPU_GATED_NORM_SILU) ||
        set_device(r))
        return -1;
    const BnCudaBuffer *w = (const BnCudaBuffer *)p->weight;
    if (!w || !w->data ||
        (Glm ? (!bn_quant_format_is_f32(w->type) || w->rows != 1 || w->cols != 128) :
               !bn_quant_format_is_bf16(w->type)) ||
        w->size / (Glm ? sizeof(float) : sizeof(uint16_t)) < 128)
        return -1;
    BnGPUResidentView views[] = {p->x, p->z, p->out};
    void *data[3];
    size_t n = (size_t)p->rows * 128;
    for (int i = 0; i < 3; i++) {
        data[i] = view_data(r, views[i]);
        if (!data[i] || !bn_quant_format_is_bf16(views[i].type) || views[i].count != n)
            return -1;
    }
    if (overlap(p->out, p->x) || overlap(p->out, p->z))
        return -1;
    int tile = 1;
    while (tile < 4 && (int64_t)tile * ctx->row_reduction_max_blocks < p->rows)
        tile *= 2;
    if (cudaGetLastError() != cudaSuccess) {
        r->failed = 1;
        return -1;
    }
#define BN_GATED_NORM(R)                                                                           \
    gated_norm_kernel<R><<<(p->rows + R - 1) / R, 32, 0, r->stream>>>(                             \
        (__nv_bfloat16 *)data[2], (const __nv_bfloat16 *)data[0], (const __nv_bfloat16 *)data[1],  \
        (const __nv_bfloat16 *)w->data, p->rows, p->eps, (int)p->activation)
    if (Glm) {
        gated_norm_kernel<2, false, true><<<(p->rows + 1) / 2, 32, 0, r->stream>>>(
            (__nv_bfloat16 *)data[2], (const __nv_bfloat16 *)data[0],
            (const __nv_bfloat16 *)data[1], NULL, p->rows, p->eps,
            (int)p->activation, (const float *)w->data);
    } else if (p->profile == BN_GPU_GATED_NORM_COMPILED_R2) {
        gated_norm_kernel<2, true><<<(p->rows + 1) / 2, 32, 0, r->stream>>>(
            (__nv_bfloat16 *)data[2], (const __nv_bfloat16 *)data[0],
            (const __nv_bfloat16 *)data[1], (const __nv_bfloat16 *)w->data,
            p->rows, p->eps, (int)p->activation);
    } else if (tile == 1) {
        BN_GATED_NORM(1);
    } else if (tile == 2) {
        BN_GATED_NORM(2);
    } else {
        BN_GATED_NORM(4);
    }
#undef BN_GATED_NORM
    if (cudaGetLastError() == cudaSuccess)
        return 0;
    r->failed = 1;
    return -1;
}

static int gated_norm(void *ctx, void *request, const BnGPUGatedRMSNormResidentPlan *p) {
    return gated_norm_impl<false>(ctx, request, p);
}

static int glm_gated_norm(void *ctx, void *request, const BnGPUGatedRMSNormResidentPlan *p) {
    return gated_norm_impl<true>(ctx, request, p);
}

static __global__ void transpose_axes_kernel(__nv_bfloat16 *out, const __nv_bfloat16 *x,
                                             size_t count, int outer, int inner, int width) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    size_t col = i % (size_t)width, head = (i / width) % ((size_t)outer * inner),
           token = i / ((size_t)outer * inner * width);
    size_t b = head / outer, a = head % outer;
    out[i] = x[((token * outer + a) * inner + b) * width + col];
}
static int transpose_axes(void *ctx_ptr, void *ptr, const BnGPUTransposeAxesResidentPlan *p) {
    BnCudaCtx *ctx = (BnCudaCtx *)ctx_ptr;
    ResidentRequest *r = (ResidentRequest *)ptr;
    if (!ctx || !r || !p || r->backend_owner != ctx || p->n_tokens <= 0 || p->outer <= 0 ||
        p->inner <= 0 || p->width <= 0 || set_device(r))
        return -1;
    size_t count = (size_t)p->n_tokens;
    int dims[] = {p->outer, p->inner, p->width};
    for (int i = 0; i < 3; i++) {
        if (count > (size_t)INT_MAX / (size_t)dims[i])
            return -1;
        count *= dims[i];
    }
    void *x = view_data(r, p->x), *out = view_data(r, p->out);
    if (!x || !out || p->x.count != count || p->out.count != count ||
        !bn_quant_format_is_bf16(p->x.type) || !bn_quant_format_is_bf16(p->out.type) ||
        overlap(p->x, p->out))
        return -1;
    if (cudaGetLastError() != cudaSuccess) {
        r->failed = 1;
        return -1;
    }
    transpose_axes_kernel<<<(unsigned)((count + 255) / 256), 256, 0, r->stream>>>(
        (__nv_bfloat16 *)out, (const __nv_bfloat16 *)x, count, p->outer, p->inner, p->width);
    if (cudaGetLastError() == cudaSuccess)
        return 0;
    r->failed = 1;
    return -1;
}
template <typename T>
static __global__ void split_columns_kernel(T *left, T *right, const T *x, size_t count, int cols,
                                            int left_cols) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    size_t row = i / (size_t)cols, col = i % (size_t)cols;
    if (col < (size_t)left_cols)
        left[row * left_cols + col] = x[i];
    else
        right[row * (cols - left_cols) + col - left_cols] = x[i];
}
static int split_columns(void *ctx_ptr, void *ptr, const BnGPUSplitColumnsResidentPlan *p) {
    BnCudaCtx *ctx = (BnCudaCtx *)ctx_ptr;
    ResidentRequest *r = (ResidentRequest *)ptr;
    if (!ctx || !r || !p || r->backend_owner != ctx || p->rows <= 0 || p->cols <= 1 ||
        p->left_cols <= 0 || p->left_cols >= p->cols || p->rows > INT_MAX / p->cols ||
        set_device(r))
        return -1;
    size_t count = (size_t)p->rows * p->cols;
    void *x = view_data(r, p->x), *left = view_data(r, p->left), *right = view_data(r, p->right);
    int f32 = bn_quant_format_is_f32(p->x.type);
    if (!x || !left || !right || (!f32 && !bn_quant_format_is_bf16(p->x.type)) ||
        p->x.type != p->left.type || p->x.type != p->right.type || p->x.count != count ||
        p->left.count != (size_t)p->rows * p->left_cols ||
        p->right.count != (size_t)p->rows * (p->cols - p->left_cols) || overlap(p->x, p->left) ||
        overlap(p->x, p->right) || overlap(p->left, p->right))
        return -1;
    if (cudaGetLastError() != cudaSuccess) {
        r->failed = 1;
        return -1;
    }
    unsigned blocks = (unsigned)((count + 255) / 256);
    if (f32)
        split_columns_kernel<<<blocks, 256, 0, r->stream>>>(
            (uint32_t *)left, (uint32_t *)right, (const uint32_t *)x, count, p->cols, p->left_cols);
    else
        split_columns_kernel<<<blocks, 256, 0, r->stream>>>(
            (uint16_t *)left, (uint16_t *)right, (const uint16_t *)x, count, p->cols, p->left_cols);
    if (cudaGetLastError() == cudaSuccess)
        return 0;
    r->failed = 1;
    return -1;
}

static int gather_rows(void *ctx,void *ptr,const BnGPUGatherRowsResidentPlan *p) {
    ResidentRequest *r=(ResidentRequest *)ptr;
    if(!ctx || !r || !p || r->backend_owner!=ctx || p->source_rows<=0 ||
       p->row_width<=0 || p->n_rows<=0 || !p->rows || set_device(r))return -1;
    const int f32=bn_quant_format_is_f32(p->x.type);
    const size_t width=f32?sizeof(uint32_t):sizeof(uint16_t);
    if((!f32 && !bn_quant_format_is_bf16(p->x.type)) || p->x.type!=p->out.type ||
       (size_t)p->source_rows>SIZE_MAX/(size_t)p->row_width/width ||
       (size_t)p->n_rows>SIZE_MAX/(size_t)p->row_width/width ||
       p->x.count!=(size_t)p->source_rows*p->row_width ||
       p->out.count!=(size_t)p->n_rows*p->row_width || overlap(p->x,p->out))return -1;
    const unsigned char *x=(const unsigned char *)view_data(r,p->x);
    unsigned char *out=(unsigned char *)view_data(r,p->out);
    if(!x || !out)return -1;
    for(int i=0;i<p->n_rows;i++)if(p->rows[i]<0 || p->rows[i]>=p->source_rows)return -1;
    if(cudaGetLastError()!=cudaSuccess){r->failed=1;return -1;}
    const size_t bytes=(size_t)p->row_width*width;
    /* Host indices are consumed here; only device addresses enter the queue.
     * No staging allocation, metadata upload or tensor readback is required. */
    for(int i=0;i<p->n_rows;i++)
        if(cudaMemcpyAsync(out+(size_t)i*bytes,x+(size_t)p->rows[i]*bytes,
            bytes,cudaMemcpyDeviceToDevice,r->stream)!=cudaSuccess){r->failed=1;return -1;}
    if(cudaGetLastError()==cudaSuccess)return 0;
    r->failed=1;return -1;
}

template <typename T>
static __global__ void join_columns_kernel(T *out,const T *left,const T *right,
    size_t count,int cols,int left_cols) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) {
        size_t row=i/cols,col=i%cols;
        out[i]=col<(size_t)left_cols?left[row*left_cols+col]:
            right[row*(cols-left_cols)+col-left_cols];
    }
}
static int join_columns(void *ctx_ptr,void *ptr,const BnGPUJoinColumnsResidentPlan *p) {
    BnCudaCtx *ctx=(BnCudaCtx*)ctx_ptr;ResidentRequest*r=(ResidentRequest*)ptr;
    if(!ctx||!r||!p||r->backend_owner!=ctx||ctx->compute_capability<800||
       p->rows<=0||p->cols<=1||p->left_cols<=0||p->left_cols>=p->cols||
       p->rows>INT_MAX/p->cols||set_device(r))return -1;
    size_t count=(size_t)p->rows*p->cols;
    void *x=view_data(r,p->x),*left=view_data(r,p->left),*right=view_data(r,p->right);
    int f32=bn_quant_format_is_f32(p->x.type);
    if(!x||!left||!right||(!f32&&!bn_quant_format_is_bf16(p->x.type))||
       p->x.type!=p->left.type||p->x.type!=p->right.type||p->x.count!=count||
       p->left.count!=(size_t)p->rows*p->left_cols||
       p->right.count!=(size_t)p->rows*(p->cols-p->left_cols)||
       overlap(p->x,p->left)||overlap(p->x,p->right))return -1;
    if(cudaGetLastError()!=cudaSuccess){r->failed=1;return -1;}
    unsigned blocks=(unsigned)((count+255)/256);
    if(f32)join_columns_kernel<<<blocks,256,0,r->stream>>>((uint32_t*)x,
        (const uint32_t*)left,(const uint32_t*)right,count,p->cols,p->left_cols);
    else join_columns_kernel<<<blocks,256,0,r->stream>>>((uint16_t*)x,
        (const uint16_t*)left,(const uint16_t*)right,count,p->cols,p->left_cols);
    if(cudaGetLastError()==cudaSuccess)return 0;
    r->failed=1;return -1;
}

static size_t linear_input_workspace_size(void *ctx_ptr, size_t max_elements) {
    const BnCudaCtx *ctx = (const BnCudaCtx *)ctx_ptr;
    if (!ctx || !max_elements || max_elements > INT_MAX || max_elements % 32 ||
        max_elements / 32 > SIZE_MAX / sizeof(BnCudaBlockQ8_1)) return 0;
    return (max_elements / 32) * sizeof(BnCudaBlockQ8_1);
}

static int reserve_linear(void *ctx_ptr, void *ptr, size_t max_elements) {
    BnCudaCtx *ctx = (BnCudaCtx *)ctx_ptr;
    ResidentRequest *r = (ResidentRequest *)ptr;
    if (!ctx || !r || r->backend_owner != ctx || max_elements == 0 || max_elements > INT_MAX ||
        max_elements % 32 || set_device(r))
        return -1;
    size_t bytes = (max_elements / 32) * sizeof(BnCudaBlockQ8_1);
    if (bytes <= r->linear_input_bytes)
        return 0;
    if (synchronize(r))
        return -1;
    BnCudaBlockQ8_1 *next = NULL;
    if (cudaMalloc(&next, bytes) != cudaSuccess)
        return -1;
    cudaFree(r->linear_input);
    r->linear_input = next;
    r->linear_input_bytes = bytes;
    return 0;
}
static int linear_a8(void *ctx_ptr, void *ptr, const BnGPUA8LinearResidentPlan *p) {
    BnCudaCtx *ctx = (BnCudaCtx *)ctx_ptr;
    ResidentRequest *r = (ResidentRequest *)ptr;
    if (!ctx || !r || !p || r->backend_owner != ctx || ctx->compute_capability < 800 ||
        set_device(r))
        return -1;
    BnCudaBF16Linear op;
    if (cuda_bf16_linear_plan(&p->weight, p->rows, p->cols, p->n_tokens, &op) ||
        op.plan.kind == BN_BACKEND_BF16_MATMUL_DECODED_GEMM)
        return -1;
    size_t nx = (size_t)p->n_tokens * p->cols, ny = (size_t)p->n_tokens * p->rows;
    void *x = view_data(r, p->x), *out = view_data(r, p->out);
    if (!x || !out || p->x.count != nx || p->out.count != ny ||
        !bn_quant_format_is_bf16(p->x.type) || !bn_quant_format_is_bf16(p->out.type) ||
        overlap(p->x, p->out) || !r->linear_input ||
        nx / 32 > r->linear_input_bytes / sizeof(BnCudaBlockQ8_1))
        return -1;
    if (cudaGetLastError() != cudaSuccess) {
        r->failed = 1;
        return -1;
    }
    unsigned blocks = (unsigned)(nx / 32);
    if (cuda_bf16_linear_needs_affine_sum(&op))
        quantize_bf16_q8_affine_batch_kernel<<<blocks, 32, 0, r->stream>>>(
            r->linear_input, (const __nv_bfloat16 *)x);
    else
        quantize_bf16_q8_batch_kernel<<<blocks, 32, 0, r->stream>>>(r->linear_input,
                                                                    (const __nv_bfloat16 *)x);
    if (cudaGetLastError() == cudaSuccess &&
        cuda_bf16_linear_device(ctx, (__nv_bfloat16 *)out, &op, (const __nv_bfloat16 *)x,
                                r->linear_input, p->rows, p->cols, p->n_tokens, r->stream,
                                r->linear_workspace,r->linear_workspace_bytes) == 0)
        return 0;
    r->failed = 1;
    return -1;
}

/* NVIDIA recommends 32MiB for sm90+, 4MiB for older GPUs. Setting a stream
 * resets the user workspace, so bind it before attaching the allocation.
 * https://docs.nvidia.com/cuda/cublas/index.html#cublassetworkspace */
static size_t bf16_linear_workspace_size(void *ctx_ptr) {
    const BnCudaCtx *ctx = (const BnCudaCtx *)ctx_ptr;
    return !ctx || ctx->compute_capability < 800 ? 0 :
        (ctx->compute_capability >= 900 ? 32u : 4u) * 1024u * 1024u;
}

static int reserve_bf16_linear(void *ctx_ptr, void *ptr, size_t max_elements) {
    BnCudaCtx *ctx = (BnCudaCtx *)ctx_ptr;
    ResidentRequest *r = (ResidentRequest *)ptr;
    if (!ctx || !r || r->backend_owner != ctx || ctx->compute_capability < 800 || set_device(r))
        return -1;
    if (max_elements && reserve_linear(ctx_ptr, ptr, max_elements))
        return -1;
    if (r->linear_cublas)
        return 0;
    if (synchronize(r))
        return -1;
    cublasHandle_t handle = NULL;
    void *workspace = NULL;
    size_t bytes = bf16_linear_workspace_size(ctx);
    cublasMath_t math = (cublasMath_t)((unsigned)CUBLAS_TENSOR_OP_MATH |
                                       (unsigned)CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS ||
        cudaMalloc(&workspace, bytes) != cudaSuccess ||
        cublasSetStream(handle, r->stream) != CUBLAS_STATUS_SUCCESS ||
        cublasSetWorkspace(handle, workspace, bytes) != CUBLAS_STATUS_SUCCESS ||
        cublasSetMathMode(handle, math) != CUBLAS_STATUS_SUCCESS) {
        if (handle)
            cublasDestroy(handle);
        cudaFree(workspace);
        fprintf(stderr, "CUDA resident BF16 GEMM setup failed\n");
        return -1;
    }
    r->linear_cublas = handle;
    r->linear_workspace = workspace;
    r->linear_workspace_bytes = bytes;
    return 0;
}
static int linear_bf16(void *ctx_ptr, void *ptr, const BnGPUBF16LinearResidentPlan *p) {
    BnCudaCtx *ctx = (BnCudaCtx *)ctx_ptr;
    ResidentRequest *r = (ResidentRequest *)ptr;
    if (!ctx || !r || !p || r->backend_owner != ctx || ctx->compute_capability < 800 ||
        set_device(r))
        return -1;
    BnCudaBF16Linear op;
    if (cuda_bf16_linear_plan(&p->weight, p->rows, p->cols, p->n_tokens, &op))
        return -1;
    if (op.plan.kind != BN_BACKEND_BF16_MATMUL_DECODED_GEMM)
        return linear_a8(ctx_ptr, ptr, p);
    size_t nx = (size_t)p->n_tokens * p->cols, ny = (size_t)p->n_tokens * p->rows;
    void *x = view_data(r, p->x), *out = view_data(r, p->out);
    if (!x || !out || p->x.count != nx || p->out.count != ny ||
        !bn_quant_format_is_bf16(p->x.type) || !bn_quant_format_is_bf16(p->out.type) ||
        overlap(p->x, p->out) || (((uintptr_t)x | (uintptr_t)out) & 15u) || !r->linear_cublas ||
        !r->linear_workspace)
        return -1;
    if (cudaGetLastError() != cudaSuccess) {
        r->failed = 1;
        return -1;
    }
    if (cuda_bf16_gemm_device(r->linear_cublas, (__nv_bfloat16 *)out, op.decoded->data,
                              (const __nv_bfloat16 *)x, p->rows, p->cols,
                              p->n_tokens) == CUBLAS_STATUS_SUCCESS &&
        cudaGetLastError() == cudaSuccess)
        return 0;
    r->failed = 1;
    return -1;
}

static int head_linear_bf16(void *ctx_ptr,void *ptr,const BnGPUHeadLinearBF16ResidentPlan *p) {
    BnCudaCtx *ctx=(BnCudaCtx*)ctx_ptr;ResidentRequest *r=(ResidentRequest*)ptr;
    if(!ctx||!r||!p||r->backend_owner!=ctx||ctx->compute_capability!=1200||p->n_tokens<1||
       p->heads<1||p->heads>128||p->rows<16||p->cols<16||p->rows%16||p->cols%16||
       p->rows>INT_MAX/p->heads||p->cols>INT_MAX/p->heads||
       p->n_tokens>INT_MAX/(p->heads*p->rows)||p->n_tokens>INT_MAX/(p->heads*p->cols)||
       (size_t)p->heads>SIZE_MAX/2/(size_t)p->rows/(size_t)p->cols||
       !r->linear_cublas||!r->linear_workspace||set_device(r))return -1;
    const BnCudaBuffer *w=(const BnCudaBuffer*)p->weight;
    size_t nx=(size_t)p->n_tokens*p->heads*p->cols,ny=(size_t)p->n_tokens*p->heads*p->rows;
    void *x=view_data(r,p->x),*out=view_data(r,p->out);
    if(!w||!w->data||!bn_quant_format_is_bf16(w->type)||w->rows!=p->heads*p->rows||w->cols!=p->cols||
       w->size!=(size_t)p->heads*p->rows*p->cols*2||!x||!out||!bn_quant_format_is_bf16(p->x.type)||
       !bn_quant_format_is_bf16(p->out.type)||p->x.count!=nx||p->out.count!=ny||overlap(p->x,p->out)||
       (((uintptr_t)x|(uintptr_t)out|(uintptr_t)w->data)&15u))return -1;
    if(cudaGetLastError()!=cudaSuccess){r->failed=1;return -1;}
    float alpha=1,beta=0;
    cublasStatus_t status=cublasGemmStridedBatchedEx(r->linear_cublas,CUBLAS_OP_T,CUBLAS_OP_N,
        p->rows,p->n_tokens,p->cols,&alpha,w->data,CUDA_R_16BF,p->cols,(long long)p->rows*p->cols,
        x,CUDA_R_16BF,p->heads*p->cols,p->cols,&beta,out,CUDA_R_16BF,p->heads*p->rows,p->rows,
        p->heads,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if(status==CUBLAS_STATUS_SUCCESS&&cudaGetLastError()==cudaSuccess)return 0;
    r->failed=1;return -1;
}

static __global__ void add_bf16_kernel(__nv_bfloat16*out,const __nv_bfloat16*x,const __nv_bfloat16*y,size_t n) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=__float2bfloat16_rn(__fadd_rn(__bfloat162float(x[i]),__bfloat162float(y[i])));
}
static int add_bf16(void*ctx_ptr,void*ptr,const BnGPUAddBF16ResidentPlan*p) {
    BnCudaCtx*ctx=(BnCudaCtx*)ctx_ptr;ResidentRequest*r=(ResidentRequest*)ptr;
    if(!ctx||!r||!p||r->backend_owner!=ctx||!p->count||p->count>INT_MAX||set_device(r))return -1;
    BnGPUResidentView v[]={p->x,p->y,p->out};void*data[3];
    for(int i=0;i<3;++i){data[i]=view_data(r,v[i]);if(!data[i]||v[i].count!=p->count||!bn_quant_format_is_bf16(v[i].type))return -1;}
    if(overlap(v[0],v[2])||overlap(v[1],v[2]))return -1;
    if(cudaGetLastError()!=cudaSuccess){r->failed=1;return -1;}
    add_bf16_kernel<<<(unsigned)((p->count+255)/256),256,0,r->stream>>>((__nv_bfloat16*)data[2],(const __nv_bfloat16*)data[0],(const __nv_bfloat16*)data[1],p->count);
    if(cudaGetLastError()==cudaSuccess)return 0;r->failed=1;return -1;
}

static __global__ void add_f32_kernel(float *out,const float *x,const float *y,size_t n) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)out[i]=__fadd_rn(x[i],y[i]);
}
static int add_f32(void *ctx_ptr,void *ptr,const BnGPUAddF32ResidentPlan *p) {
    BnCudaCtx *ctx=(BnCudaCtx*)ctx_ptr;
    ResidentRequest *r=(ResidentRequest*)ptr;
    if(!ctx||!r||!p||r->backend_owner!=ctx||!p->count||
       p->count>INT_MAX||p->count>SIZE_MAX/sizeof(float)||r->failed)return -1;
    BnGPUResidentView v[]={p->x,p->y,p->out};void *data[3];
    for(int i=0;i<3;i++) {
        if(v[i].count!=p->count||v[i].type!=BN_GGUF_TENSOR_F32)return -1;
        data[i]=view_data(r,v[i]);
        if(!data[i]||((uintptr_t)data[i]&3u))return -1;
    }
    if(overlap(v[0],v[2])||overlap(v[1],v[2])||set_device(r))return -1;
    if(cudaGetLastError()!=cudaSuccess){r->failed=1;return -1;}
    add_f32_kernel<<<(unsigned)((p->count+255)/256),256,0,r->stream>>>(
        (float*)data[2],(const float*)data[0],(const float*)data[1],p->count);
    if(cudaGetLastError()==cudaSuccess)return 0;
    r->failed=1;return -1;
}

static __global__ void cast_bf16_f32(float*out,const __nv_bfloat16*x,size_t n) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=__bfloat162float(x[i]);
}
static int linear_bf16_f32_weight(void *ctx_ptr,void *ptr,const BnGPUBF16F32LinearResidentPlan*p) {
    BnCudaCtx*ctx=(BnCudaCtx*)ctx_ptr;ResidentRequest*r=(ResidentRequest*)ptr;
    if(!ctx||!r||!p||r->backend_owner!=ctx||ctx->compute_capability!=1200||
       p->n_tokens<1||p->rows<1||p->cols<1||p->n_tokens>INT_MAX/p->rows||p->n_tokens>INT_MAX/p->cols||
       !r->linear_cublas||!r->linear_workspace||set_device(r))return -1;
    const BnCudaBuffer*w=(const BnCudaBuffer*)p->weight;
    if(!w||!w->data||w->type!=BN_GGUF_TENSOR_F32||w->rows!=p->rows||w->cols!=p->cols||
       (size_t)p->rows>SIZE_MAX/sizeof(float)/(size_t)p->cols||w->size!=(size_t)p->rows*p->cols*sizeof(float))return -1;
    size_t nx=(size_t)p->n_tokens*p->cols,ny=(size_t)p->n_tokens*p->rows;
    BnGPUResidentView v[]={p->x,p->cast,p->out};size_t counts[]={nx,nx,ny};void*data[3];
    for(int i=0;i<3;++i){data[i]=view_data(r,v[i]);
        if(!data[i]||v[i].count!=counts[i]||v[i].type!=(i?BN_GGUF_TENSOR_F32:BN_GGUF_TENSOR_BF16))return -1;
        for(int j=0;j<i;++j)if(overlap(v[i],v[j]))return -1;}
    if(cudaGetLastError()!=cudaSuccess){r->failed=1;return -1;}
    cast_bf16_f32<<<(unsigned)((nx+255)/256),256,0,r->stream>>>((float*)data[1],(const __nv_bfloat16*)data[0],nx);
    float alpha=1,beta=0;
    if(cudaGetLastError()==cudaSuccess&&cublasGemmEx(r->linear_cublas,CUBLAS_OP_T,CUBLAS_OP_N,
       p->rows,p->n_tokens,p->cols,&alpha,w->data,CUDA_R_32F,p->cols,data[1],CUDA_R_32F,p->cols,
       &beta,data[2],CUDA_R_32F,p->rows,CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT)==CUBLAS_STATUS_SUCCESS&&
       cudaGetLastError()==cudaSuccess)return 0;
    r->failed=1;return -1;
}

static int ffn_linear(BnCudaCtx *ctx, ResidentRequest *r, __nv_bfloat16 *out,
                      const __nv_bfloat16 *x, const BnCudaBF16Linear *op,
                      int rows, int cols, int nt) {
    if (op->plan.kind == BN_BACKEND_BF16_MATMUL_DECODED_GEMM)
        return cuda_bf16_gemm_device(r->linear_cublas, out, op->decoded->data, x,
                                     rows, cols, nt) == CUBLAS_STATUS_SUCCESS ? 0 : -1;
    return cuda_bf16_linear_device(ctx, out, op, x, r->linear_input, rows, cols, nt, r->stream,
        r->linear_workspace,r->linear_workspace_bytes);
}

static int ffn_prepare(ResidentRequest *r, const __nv_bfloat16 *x, size_t count,
                       int affine) {
    unsigned blocks = (unsigned)(count / 32);
    if (affine)
        quantize_bf16_q8_affine_batch_kernel<<<blocks, 32, 0, r->stream>>>(r->linear_input, x);
    else
        quantize_bf16_q8_batch_kernel<<<blocks, 32, 0, r->stream>>>(r->linear_input, x);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

static __global__ void clamped_swiglu_kernel(__nv_bfloat16 *out,
    const __nv_bfloat16 *x,int nt,int hidden,float limit) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=nt*hidden)return;
    int row=i/hidden,col=i%hidden;
    float gate=fminf(__bfloat162float(x[(size_t)row*hidden*2+col]),limit);
    float up=fmaxf(-limit,fminf(__bfloat162float(x[(size_t)row*hidden*2+hidden+col]),limit));
    float silu=gate/(1.0f+expf(-gate));
    silu=__bfloat162float(__float2bfloat16_rn(silu));
    out[i]=__float2bfloat16_rn(__fmul_rn(silu,__fadd_rn(up,0.0f)));
}
static int clamped_swiglu(void *ctx_ptr,void *ptr,const BnGPUClampedSwiGLUResidentPlan *p) {
    BnCudaCtx *ctx=(BnCudaCtx*)ctx_ptr;ResidentRequest*r=(ResidentRequest*)ptr;
    if(!ctx||!r||!p||r->backend_owner!=ctx||ctx->compute_capability<800||
        p->n_tokens<=0||p->hidden_dim<=0||p->hidden_dim>INT_MAX/2||
        p->n_tokens>INT_MAX/(p->hidden_dim*2)||!isfinite(p->limit)||p->limit<0||set_device(r))return -1;
    size_t n=(size_t)p->n_tokens*p->hidden_dim;
    void *x=view_data(r,p->gate_up),*out=view_data(r,p->out);
    if(!x||!out||p->gate_up.count!=n*2||p->out.count!=n||
        !bn_quant_format_is_bf16(p->gate_up.type)||!bn_quant_format_is_bf16(p->out.type)||overlap(p->gate_up,p->out))return -1;
    if(cudaGetLastError()!=cudaSuccess){r->failed=1;return -1;}
    clamped_swiglu_kernel<<<(unsigned)((n+255)/256),256,0,r->stream>>>((__nv_bfloat16*)out,(const __nv_bfloat16*)x,p->n_tokens,p->hidden_dim,p->limit);
    if(cudaGetLastError()==cudaSuccess)return 0;
    r->failed=1;return -1;
}

static int dense_ffn(void *ctx_ptr, void *ptr, const BnGPUFFNResidentPlan *p) {
    BnCudaCtx *ctx = (BnCudaCtx *)ctx_ptr;
    ResidentRequest *r = (ResidentRequest *)ptr;
    if (!ctx || !r || !p || r->backend_owner != ctx || ctx->compute_capability < 800 ||
        p->n_tokens <= 0 || p->dim <= 0 || p->hidden_dim <= 0 ||
        (p->activation_profile != BN_GPU_BF16_FFN_SILU_SEPARATE &&
         p->activation_profile != BN_GPU_BF16_FFN_SILU_FUSED &&
         p->activation_profile != BN_GPU_BF16_FFN_SILU_COMPILED) ||
        (p->layout != BN_GPU_BF16_FFN_SEPARATE && p->layout != BN_GPU_BF16_FFN_STACKED_DECODED) ||
        p->n_tokens > INT_MAX / p->dim || p->n_tokens > INT_MAX / p->hidden_dim ||
        set_device(r))
        return -1;
    BnCudaBF16Linear ops[3];
    if (cuda_bf16_linear_plan(&p->weights.gate, p->hidden_dim, p->dim, p->n_tokens, &ops[0]) ||
        cuda_bf16_linear_plan(&p->weights.up, p->hidden_dim, p->dim, p->n_tokens, &ops[1]) ||
        cuda_bf16_linear_plan(&p->weights.down, p->dim, p->hidden_dim, p->n_tokens, &ops[2]))
        return -1;
    size_t count = (size_t)p->n_tokens * p->dim, full = (size_t)p->n_tokens * p->hidden_dim;
    BnGPUResidentView views[] = {p->x, p->gate, p->up, p->out};
    size_t counts[] = {count, full, full, count};
    void *data[4];
    for (unsigned i = 0; i < 4; i++) {
        data[i] = view_data(r, views[i]);
        if (!data[i] || views[i].count != counts[i] ||
            !bn_quant_format_is_bf16(views[i].type) || ((uintptr_t)data[i] & 15u))
            return -1;
        for (unsigned j = 0; j < i; j++)
            if (overlap(views[i], views[j]))
                return -1;
    }
    const bool stacked=p->layout==BN_GPU_BF16_FFN_STACKED_DECODED;
    const BnCudaBuffer *joined=(const BnCudaBuffer *)p->gate_up;
    void *merged=NULL;
    if(stacked) {
        if(p->activation_profile!=BN_GPU_BF16_FFN_SILU_COMPILED||
           p->hidden_dim>INT_MAX/2||p->n_tokens>INT_MAX/(2*p->hidden_dim)||
           ops[0].plan.kind!=BN_BACKEND_BF16_MATMUL_DECODED_GEMM||
           ops[1].plan.kind!=BN_BACKEND_BF16_MATMUL_DECODED_GEMM||
           p->weights.gate.type!=p->weights.up.type||!joined||!joined->data||
           joined->type!=BN_GGUF_TENSOR_BF16||joined->rows!=2*p->hidden_dim||joined->cols!=p->dim||
           (size_t)joined->rows>SIZE_MAX/(size_t)p->dim/sizeof(uint16_t)||
           joined->size<(size_t)joined->rows*p->dim*sizeof(uint16_t)||
           ops[0].decoded->data_parent!=joined||ops[1].decoded->data_parent!=joined||
           ops[0].decoded->data!=joined->data||
           ops[1].decoded->data!=(const char *)joined->data+(size_t)p->hidden_dim*p->dim*sizeof(uint16_t)||
           p->gate_up_out.count!=full*2||!bn_quant_format_is_bf16(p->gate_up_out.type)||
           p->gate_up_out.slot!=p->gate.slot||p->gate_up_out.offset!=p->gate.offset||
           !(merged=view_data(r,p->gate_up_out))||((uintptr_t)merged&15u))return -1;
        for(unsigned i=0;i<4;i++)if(i!=1&&overlap(p->gate_up_out,views[i]))return -1;
    } else if(joined||p->gate_up_out.count)return -1;
    int a8[3];
    size_t needed = 0;
    for (unsigned i = 0; i < 3; i++) {
        a8[i] = ops[i].plan.kind != BN_BACKEND_BF16_MATMUL_DECODED_GEMM;
        if (a8[i]) {
            size_t n = i == 2 ? full : count;
            if (n > needed) needed = n;
        } else if (!r->linear_cublas || !r->linear_workspace)
            return -1;
    }
    if (needed && (!r->linear_input ||
        needed / 32 > r->linear_input_bytes / sizeof(BnCudaBlockQ8_1)))
        return -1;
    if (cudaGetLastError() != cudaSuccess) {
        r->failed = 1;
        return -1;
    }
    const __nv_bfloat16 *x = (const __nv_bfloat16 *)data[0];
    __nv_bfloat16 *g = (__nv_bfloat16 *)data[1], *u = (__nv_bfloat16 *)data[2];
    if(stacked) {
        if(cuda_bf16_gemm_device(r->linear_cublas,(__nv_bfloat16 *)merged,joined->data,x,
            2*p->hidden_dim,p->dim,p->n_tokens)!=CUBLAS_STATUS_SUCCESS) {
            r->failed=1;return -1;
        }
        silu_mul_stacked_compiled_bf16_kernel<<<(unsigned)((full+255)/256),256,0,r->stream>>>(
            u,(const __nv_bfloat16 *)merged,full,p->hidden_dim);
        g=u;
    } else {
    if (((a8[0] || a8[1]) && ffn_prepare(r, x, count,
            cuda_bf16_linear_needs_affine_sum(&ops[0]) ||
            cuda_bf16_linear_needs_affine_sum(&ops[1]))) ||
        ffn_linear(ctx, r, g, x, &ops[0], p->hidden_dim, p->dim, p->n_tokens) ||
        ffn_linear(ctx, r, u, x, &ops[1], p->hidden_dim, p->dim, p->n_tokens) ||
        cudaGetLastError() != cudaSuccess) {
        r->failed = 1;
        return -1;
    }
    if (p->activation_profile == BN_GPU_BF16_FFN_SILU_COMPILED)
        silu_mul_compiled_bf16_kernel<<<(unsigned)((full + 255) / 256), 256, 0, r->stream>>>(g, u, full);
    else if (p->activation_profile == BN_GPU_BF16_FFN_SILU_FUSED)
        silu_mul_fused_bf16_kernel<<<(unsigned)((full + 255) / 256), 256, 0, r->stream>>>(g, u, full);
    else
        silu_mul_bf16_kernel<<<(unsigned)((full + 255) / 256), 256, 0, r->stream>>>(g, u, full);
    }
    if (cudaGetLastError() == cudaSuccess &&
        (!a8[2] || ffn_prepare(r, g, full, cuda_bf16_linear_needs_affine_sum(&ops[2])) == 0) &&
        ffn_linear(ctx, r, (__nv_bfloat16 *)data[3], g, &ops[2], p->dim,
                   p->hidden_dim, p->n_tokens) == 0 && cudaGetLastError() == cudaSuccess)
        return 0;
    r->failed = 1;
    return -1;
}

} // namespace bn_cuda_gdn

#endif
