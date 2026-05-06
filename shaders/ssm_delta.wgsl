// Delta rule recurrence (one workgroup per V-head):
//   1. Decay: S *= alpha
//   2. sk = S @ k
//   3. S += k outer (beta * (v - sk))
//   4. out = S^T @ (q * q_scale)
// State S is [head_k_dim x head_v_dim] per V-head, persistent across tokens.
// Dispatch: (num_v_heads, 1, 1)

struct Uniforms {
    p0: u32, p1: u32, p2: u32, p3: u32,
    p4: u32, p5: u32, p6: u32, p7: u32,
}

@group(0) @binding(0) var<storage, read_write> state: array<f32>;   // [num_v_heads * hk * hv]
@group(0) @binding(1) var<storage, read_write> out: array<f32>;     // [num_v_heads * hv]
@group(0) @binding(2) var<storage, read> q: array<f32>;             // [num_k_heads * hk]
@group(0) @binding(3) var<storage, read> k: array<f32>;             // [num_k_heads * hk]
@group(0) @binding(4) var<storage, read> v: array<f32>;             // [num_v_heads * hv]
@group(0) @binding(5) var<storage, read> alpha: array<f32>;         // [num_v_heads]
@group(0) @binding(6) var<storage, read> beta: array<f32>;          // [num_v_heads]
@group(0) @binding(7) var<uniform> u: Uniforms;

// p0 = head_k_dim, p1 = head_v_dim, p2 = num_k_heads,
// p3 = q_scale (bitcast to f32), p4 = state_offset (bytes), p5 = state_layer_size (bytes)
// p6 = q offset, p7 = k offset (float indices); v offset is 2 * num_k_heads * head_k_dim.

// Shared memory for sk. Must be >= head_v_dim. 512 covers all known models
// (typical: 128, Qwen3.5: 128, max practical: ~256).
var<workgroup> sk: array<f32, 512>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let hv_idx = wid.x;
    let tid = lid.x;
    let hk = u.p0;
    let hv = u.p1;
    let num_k_heads = u.p2;
    let q_scale = bitcast<f32>(u.p3);
    let state_layer_off = u.p4 / 4u;  // byte offset → float offset
    let q_off = u.p6;
    let k_off = u.p7;
    let v_off = 2u * num_k_heads * hk;

    let hk_idx = hv_idx % num_k_heads;
    let state_base = state_layer_off + hv_idx * hk * hv;
    let decay = alpha[hv_idx];
    let b = beta[hv_idx];

    // Step 1: Decay state S *= alpha (each thread handles a slice)
    let total = hk * hv;
    var i = tid;
    while (i < total) {
        state[state_base + i] *= decay;
        i += 256u;
    }
    workgroupBarrier();

    // Step 2: sk[v] = sum_k S[k,v] * k[k] (Kahan compensated summation for precision)
    var vi = tid;
    while (vi < hv) {
        var sum: f32 = 0.0;
        var comp: f32 = 0.0;
        for (var ki: u32 = 0u; ki < hk; ki++) {
            let y = state[state_base + ki * hv + vi] * k[k_off + hk_idx * hk + ki] - comp;
            let t = sum + y;
            comp = (t - sum) - y;
            sum = t;
        }
        sk[vi] = sum;
        vi += 256u;
    }
    workgroupBarrier();

    // Step 3: S += k outer (beta * (v - sk))
    // Each thread handles a slice of the k x v state matrix
    i = tid;
    while (i < total) {
        let ki = i / hv;
        let vi2 = i % hv;
        let kk = k[k_off + hk_idx * hk + ki];
        state[state_base + i] += kk * b * (v[v_off + hv_idx * hv + vi2] - sk[vi2]);
        i += 256u;
    }
    workgroupBarrier();

    // Step 4: out[v] = sum_k S[k,v] * q[k] * q_scale (Kahan compensated)
    vi = tid;
    while (vi < hv) {
        var sum: f32 = 0.0;
        var comp: f32 = 0.0;
        for (var ki: u32 = 0u; ki < hk; ki++) {
            let y = state[state_base + ki * hv + vi] * q[q_off + hk_idx * hk + ki] - comp;
            let t = sum + y;
            comp = (t - sum) - y;
            sum = t;
        }
        out[hv_idx * hv + vi] = sum * q_scale;
        vi += 256u;
    }
}
