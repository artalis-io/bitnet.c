/* Original exllamav3 bd1f5191 CUDA arithmetic; MIT third_party/exllamav3/LICENSE. */
#pragma once
#define NUM_THREADS_STORE 256
__device__ __forceinline__ void sincos_accurate(float x, float* s, float* c)
{
    const float k = rintf(x * 0.15915494309189535f);
    float r = fmaf(-k, 6.28318548202514648f, x);
    r = fmaf(-k, -1.74845553e-7f, r);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    __sincosf(r, s, c);
#else
    // Host pass of a .cpp that includes this header: the body is parsed, never run, and the
    // intrinsic is only declared under the device compilers (glibc declares a __sincosf of its
    // own, MSVC's CRT does not)
    *s = sinf(r);
    *c = cosf(r);
#endif
}

__global__ __launch_bounds__(1024)
void dsv4_compress_windows_kernel
(
    const half* __restrict__ kv_new,     // (seq, W)
    const half* __restrict__ gate_new,   // (seq, W)
    const half* __restrict__ ring_kv,    // (buf_rows, W)
    const half* __restrict__ ring_gate,  // (buf_rows, W)
    float* __restrict__ ovl,             // (depth, 2, m, hd) or nullptr
    const float* __restrict__ ape,       // (m, W)
    const half* __restrict__ norm_w,     // (hd)
    const float eps,
    const float* __restrict__ inv_freq,  // (rd / 2)
    half* __restrict__ dest_a,           // (cap, Wa)
    half* __restrict__ dest_b,           // (cap, hd - Wa) or nullptr
    const int* __restrict__ pos_ptr,     // device position override or nullptr
    const int pos_base,
    const int seq,
    const int m,
    const int buf_rows,
    const int ovl_depth,
    const int W,
    const int hd,
    const int rd,
    const int Wa,
    const bool overlap,
    const int* __restrict__ slot_ids,    // batched: (B,) state slot per job, else nullptr
    const int ring_stride,               // batched: per-slot strides in elements
    const int ovl_stride,
    const int da_stride,
    const int db_stride,
    const int* __restrict__ pool_bt,     // paged pools: block table (one row per job), else nullptr
    const int bt_stride,                 // paged pools, batched: block table row stride
    const int epp,                       // paged pools: entries per page
    const bool stage_rel                 // dest_a is a per-JOB staging buffer of this step's
                                         // entries at rows [0, nw) (packed-pool quantization
                                         // follows); dest_b / pool_bt unused
)
{
    extern __shared__ float sh[];        // comp[hd] + reduce[hd / 32]
    float* sh_comp = sh;
    float* sh_red = sh + hd;

    // Batched mode: grid.y = job; chunk rows are job-sliced, state buffers slot-indexed,
    // position read per job. Single-job launches have grid.y = 1 and null slot_ids.
    // Pool destinations are shared paged tensors addressed via the per-job block table
    // when pool_bt is given, per-slot strided otherwise (contiguous scratch)
    const int job = blockIdx.y;
    if (slot_ids)
    {
        int slot = slot_ids[job];
        kv_new += (size_t) job * seq * W;
        gate_new += (size_t) job * seq * W;
        ring_kv += (size_t) slot * ring_stride;
        ring_gate += (size_t) slot * ring_stride;
        if (ovl) ovl += (size_t) slot * ovl_stride;
        if (!pool_bt)
        {
            dest_a += (size_t) (stage_rel ? job : slot) * da_stride;
            if (dest_b) dest_b += (size_t) slot * db_stride;
        }
    }
    if (pool_bt) pool_bt += (size_t) job * bt_stride;
    int pos0 = pos_ptr ? pos_ptr[job] : pos_base;
    int ec0 = pos0 / m;
    int nw = (pos0 + seq) / m - ec0;

    int w = blockIdx.x;
    if (w >= nw) return;
    int c = threadIdx.x;                 // column; blockDim.x = hd padded to a warp multiple
    bool active = c < hd;
    int t_warps = blockDim.x / 32;

    // Softmax-pooled window entry, online, fp32
    int E = overlap ? 2 * m : m;
    float mx = -1e30f;
    float l = 0.0f;
    float acc = 0.0f;

    for (int e = 0; active && e < E; ++e)
    {
        float kvv, gv;
        if (overlap && e < m && w == 0)
        {
            // Previous window's Ca slice from the snapshot ring (ape already applied)
            if (ec0 == 0) continue;
            const float* snap = ovl + (size_t) ((ec0 - 1) % ovl_depth) * 2 * m * hd;
            kvv = snap[e * hd + c];
            gv = snap[(m + e) * hd + c];
        }
        else
        {
            int abs_pos, sc;
            if (overlap && e < m)
            {
                abs_pos = (ec0 + w - 1) * m + e;
                sc = c;
            }
            else
            {
                int r = overlap ? e - m : e;
                abs_pos = (ec0 + w) * m + r;
                sc = overlap ? hd + c : c;
            }
            if (abs_pos >= pos0)
            {
                kvv = __half2float(kv_new[(size_t) (abs_pos - pos0) * W + sc]);
                gv = __half2float(gate_new[(size_t) (abs_pos - pos0) * W + sc]);
            }
            else
            {
                size_t row = (size_t) (abs_pos % buf_rows) * W;
                kvv = __half2float(ring_kv[row + sc]);
                gv = __half2float(ring_gate[row + sc]);
            }
            gv += ape[(abs_pos % m) * W + sc];
        }
        float nm = fmaxf(mx, gv);
        float alpha = __expf(mx - nm);
        float p = __expf(gv - nm);
        l = l * alpha + p;
        acc = acc * alpha + p * kvv;
        mx = nm;
    }
    float comp = active ? acc / l : 0.0f;

    // Weighted RMS norm over the hd columns
    float sq = comp * comp;
    for (int offset = 16; offset > 0; offset >>= 1)
        sq += __shfl_down_sync(0xffffffffu, sq, offset);
    if ((c % 32) == 0) sh_red[c / 32] = sq;
    __syncthreads();
    if (c < 32)
    {
        sq = c < t_warps ? sh_red[c] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sq += __shfl_down_sync(0xffffffffu, sq, offset);
        if (c == 0) sh_red[0] = sq;
    }
    __syncthreads();
    float rmr = rsqrtf(sh_red[0] / (float) hd + eps);
    float normed = comp * rmr * (active ? __half2float(norm_w[c]) : 0.0f);
    if (active) sh_comp[c] = normed;
    __syncthreads();
    if (!active) return;

    // GPT-J rope on the trailing rd columns at the window position
    float out = normed;
    int c0 = hd - rd;
    if (c >= c0)
    {
        int p = (c - c0) / 2;
        float theta = inv_freq[p] * (float) ((ec0 + w) * m);
        float cs, sn;
        sincos_accurate(theta, &sn, &cs);
        float v_e = sh_comp[c0 + p * 2];
        float v_o = sh_comp[c0 + p * 2 + 1];
        out = ((c - c0) & 1) ? (v_o * cs + v_e * sn) : (v_e * cs - v_o * sn);
    }

    size_t drow = stage_rel ? (size_t) w : (size_t) (ec0 + w);
    if (pool_bt && !stage_rel)
        drow = (size_t) pool_bt[drow / epp] * epp + drow % epp;
    if (!dest_b || c < Wa)
        dest_a[drow * Wa + c] = __float2half_rn(out);
    else
        dest_b[drow * (size_t) (hd - Wa) + (c - Wa)] = __float2half_rn(out);

    // Overlap snapshot for the next call: last window's Ca slice, written by block 0 AFTER
    // its restore read (see header comment). Sources are kv_new / pre-pos0 ring rows only,
    // both stable within this kernel
    if (overlap && w == 0)
    {
        float* snap = ovl + (size_t) ((ec0 + nw - 1) % ovl_depth) * 2 * m * hd;
        for (int e = 0; e < m; ++e)
        {
            int abs_pos = (ec0 + nw - 1) * m + e;
            float kvv, gv;
            if (abs_pos >= pos0)
            {
                kvv = __half2float(kv_new[(size_t) (abs_pos - pos0) * W + c]);
                gv = __half2float(gate_new[(size_t) (abs_pos - pos0) * W + c]);
            }
            else
            {
                size_t row = (size_t) (abs_pos % buf_rows) * W;
                kvv = __half2float(ring_kv[row + c]);
                gv = __half2float(ring_gate[row + c]);
            }
            snap[e * hd + c] = kvv;
            snap[(m + e) * hd + c] = gv + ape[(abs_pos % m) * W + c];
        }
    }
}


__global__ __launch_bounds__(NUM_THREADS_STORE)
void dsv4_compress_store_kernel
(
    const half* __restrict__ kv_new,     // (seq, W)
    const half* __restrict__ gate_new,   // (seq, W)
    half* __restrict__ ring_kv,          // (buf_rows, W)
    half* __restrict__ ring_gate,        // (buf_rows, W)
    const int* __restrict__ pos_ptr,
    const int pos_base,
    const int seq,
    const int buf_rows,
    const int W,
    const int j0,                        // first chunk row to store (seq - buf_rows clamp)
    const int* __restrict__ slot_ids,
    const int ring_stride
)
{
    const int job = blockIdx.y;
    if (slot_ids)
    {
        int slot = slot_ids[job];
        kv_new += (size_t) job * seq * W;
        gate_new += (size_t) job * seq * W;
        ring_kv += (size_t) slot * ring_stride;
        ring_gate += (size_t) slot * ring_stride;
    }
    int pos0 = pos_ptr ? pos_ptr[job] : pos_base;
    size_t i = (size_t) blockIdx.x * NUM_THREADS_STORE + threadIdx.x;
    size_t total = (size_t) (seq - j0) * W;
    if (i >= total) return;
    int j = j0 + (int) (i / W);
    int c = (int) (i % W);
    size_t src = (size_t) j * W + c;
    size_t dst = (size_t) ((pos0 + j) % buf_rows) * W + c;
    ring_kv[dst] = kv_new[src];
    ring_gate[dst] = gate_new[src];
}


