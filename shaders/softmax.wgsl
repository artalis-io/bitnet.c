// Per-head softmax over attention scores
// Dispatch: (n_heads, 1, 1) — one workgroup per head

struct Uniforms {
    p0: u32, p1: u32, p2: u32, p3: u32,
    p4: u32, p5: u32, p6: u32, p7: u32,
}

@group(0) @binding(0) var<storage, read_write> att: array<f32>;
@group(0) @binding(1) var<uniform> u: Uniforms;

// p0 = n_heads, p1 = n_kv (valid positions), p2 = seq_len

var<workgroup> shared_mem: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let h = wid.x;
    let tid = lid.x;
    let n_heads = u.p0;
    let n_kv = u.p1;
    let seq_len = u.p2;

    if (h >= n_heads) {
        return;
    }

    let base = h * seq_len;

    // Phase 1: find max (parallel reduction)
    var local_max = -3.402823e+38f;  // -FLT_MAX
    var i = tid;
    while (i < n_kv) {
        local_max = max(local_max, att[base + i]);
        i += 256u;
    }

    shared_mem[tid] = local_max;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            shared_mem[tid] = max(shared_mem[tid], shared_mem[tid + s]);
        }
        workgroupBarrier();
    }
    let max_val = shared_mem[0];
    workgroupBarrier();

    // Phase 2: exp(x - max) and sum
    var local_sum = 0.0;
    var comp = 0.0;
    i = tid;
    while (i < n_kv) {
        let e = exp(att[base + i] - max_val);
        att[base + i] = e;
        let y = e - comp;
        let t = local_sum + y;
        comp = (t - local_sum) - y;
        local_sum = t;
        i += 256u;
    }

    shared_mem[tid] = local_sum;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        workgroupBarrier();
    }
    let sum_val = shared_mem[0];
    workgroupBarrier();

    // Phase 3: normalize
    let inv_sum = 1.0 / sum_val;
    i = tid;
    while (i < n_kv) {
        att[base + i] *= inv_sum;
        i += 256u;
    }
}
