// Per-head RMS normalization: for each head h, normalize x[h*hs..(h+1)*hs].
// Supports shared weights (all heads same vector) or per-head weights.
// Dispatch: (n_heads, 1, 1) — one workgroup per head.

struct Uniforms {
    p0: u32, p1: u32, p2: u32, p3: u32,
    p4: u32, p5: u32, p6: u32, p7: u32,
}

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<uniform> u: Uniforms;

// p0 = head_size, p1 = eps (bitcast to f32), p2 = per_head (1=per-head weights, 0=shared)

var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let head = wid.x;
    let tid = lid.x;
    let hs = u.p0;
    let eps = bitcast<f32>(u.p1);
    let per_head = u.p2;
    let x_base = head * hs;
    let w_base = select(0u, head * hs, per_head != 0u);

    // Accumulate sum of squares
    var ss: f32 = 0.0;
    var d = tid;
    while (d < hs) {
        let v = x[x_base + d];
        ss += v * v;
        d += 256u;
    }
    shared_sum[tid] = ss;
    workgroupBarrier();

    // Reduction
    var stride: u32 = 128u;
    while (stride > 0u) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }

    let scale = 1.0 / sqrt(shared_sum[0] / f32(hs) + eps);
    workgroupBarrier();

    // Normalize in-place
    d = tid;
    while (d < hs) {
        x[x_base + d] = x[x_base + d] * weight[w_base + d] * scale;
        d += 256u;
    }
}
