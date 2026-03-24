// GQA weighted sum: xb[h] = sum_i(att[h][i] * V[i]) for all KV positions
// Dispatch: (n_heads, 1, 1) — one workgroup per head

struct Uniforms {
    p0: u32, p1: u32, p2: u32, p3: u32,
    p4: u32, p5: u32, p6: u32, p7: u32,
}

@group(0) @binding(0) var<storage, read> att: array<f32>;
@group(0) @binding(1) var<storage, read> value_cache: array<f32>;
@group(0) @binding(2) var<storage, read_write> xb: array<f32>;
@group(0) @binding(3) var<uniform> u: Uniforms;

// p0 = n_heads, p1 = head_size, p2 = n_kv, p3 = kv_mul
// p4 = kv_dim, p5 = seq_len, p6 = loff (layer offset in f32 elements)

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let h = wid.x;
    let tid = lid.x;
    let n_heads = u.p0;
    let head_size = u.p1;
    let n_kv = u.p2;
    let kv_mul = u.p3;
    let kv_dim = u.p4;
    let seq_len = u.p5;
    let loff = u.p6;

    if (h >= n_heads) {
        return;
    }

    let kv_h = h / kv_mul;
    let out_base = h * head_size;

    // Each thread handles a subset of dimensions
    var d = tid;
    while (d < head_size) {
        var acc = 0.0f;
        for (var i = 0u; i < n_kv; i++) {
            let t = i;  // linear index into cache slots
            let v_base = loff + t * kv_dim + kv_h * head_size;
            let a = att[h * seq_len + i];
            acc += a * value_cache[v_base + d];
        }
        xb[out_base + d] = acc;
        d += 256u;
    }
}
