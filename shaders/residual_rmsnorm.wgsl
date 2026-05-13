// Fused residual addition + RMS normalization:
//   x[i] += r[i]                                    (in-place residual)
//   out[i] = x[i] * weight[i] / sqrt(mean(x²) + eps)  (RMSNorm)
// Dispatch: (1, 1, 1)
//
// p0 = dim, p1 = eps (bitcast f32)

struct Uniforms {
    p0: u32, p1: u32, p2: u32, p3: u32,
    p4: u32, p5: u32, p6: u32, p7: u32,
}

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> r: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> u: Uniforms;

var<workgroup> shared_mem: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let dim = u.p0;
    let eps = bitcast<f32>(u.p1);

    // Phase 1: x[i] += r[i], accumulate sum of squares. Use compensated local
    // accumulation to limit early CPU/GPU drift from strided reduction order.
    var sum_sq = 0.0;
    var comp = 0.0;
    var i = tid;
    while (i < dim) {
        let xr = x[i] + r[i];
        x[i] = xr;
        let y = xr * xr - comp;
        let t = sum_sq + y;
        comp = (t - sum_sq) - y;
        sum_sq = t;
        i += 256u;
    }

    // Workgroup reduce sum
    shared_mem[tid] = sum_sq;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        workgroupBarrier();
    }

    // Thread 0 computes scale
    if (tid == 0u) {
        shared_mem[0] = 1.0 / sqrt(shared_mem[0] / f32(dim) + eps);
    }
    workgroupBarrier();

    let scale = shared_mem[0];

    // Phase 2: normalize — read x[i] (already updated), write out[i]
    i = tid;
    while (i < dim) {
        out[i] = x[i] * weight[i] * scale;
        i += 256u;
    }
}
