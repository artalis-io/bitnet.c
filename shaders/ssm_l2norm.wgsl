// Per-head L2 normalization of Q and K vectors.
// One workgroup per head. Workgroup reduction for norm computation.
// Dispatch: (num_k_heads, 1, 1)

struct Uniforms {
    p0: u32, p1: u32, p2: u32, p3: u32,
    p4: u32, p5: u32, p6: u32, p7: u32,
}

@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read_write> k: array<f32>;
@group(0) @binding(2) var<uniform> u: Uniforms;

// p0 = head_dim, p1 = q offset, p2 = k offset (float indices)

var<workgroup> shared_q: array<f32, 256>;
var<workgroup> shared_k: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let head = wid.x;
    let tid = lid.x;
    let hd = u.p0;
    let q_base = u.p1 + head * hd;
    let k_base = u.p2 + head * hd;

    // Accumulate squared norm
    var qn: f32 = 0.0;
    var kn: f32 = 0.0;
    var d = tid;
    while (d < hd) {
        let qv = q[q_base + d];
        let kv = k[k_base + d];
        qn += qv * qv;
        kn += kv * kv;
        d += 256u;
    }
    shared_q[tid] = qn;
    shared_k[tid] = kn;
    workgroupBarrier();

    // Reduction
    var stride: u32 = 128u;
    while (stride > 0u) {
        if (tid < stride) {
            shared_q[tid] += shared_q[tid + stride];
            shared_k[tid] += shared_k[tid + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }

    let inv_qn = 1.0 / (sqrt(shared_q[0]) + 1e-6);
    let inv_kn = 1.0 / (sqrt(shared_k[0]) + 1e-6);
    workgroupBarrier();

    // Normalize
    d = tid;
    while (d < hd) {
        q[q_base + d] *= inv_qn;
        k[k_base + d] *= inv_kn;
        d += 256u;
    }
}
