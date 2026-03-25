// GQA attention scores: Q·K dot products for all valid KV positions
// Dispatch: (n_heads, 1, 1) — one workgroup per head

struct Uniforms {
    p0: u32, p1: u32, p2: u32, p3: u32,
    p4: u32, p5: u32, p6: u32, p7: u32,
}

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> key_cache: array<f32>;
@group(0) @binding(2) var<storage, read_write> att: array<f32>;
@group(0) @binding(3) var<uniform> u: Uniforms;

// p0 = n_heads, p1 = head_size, p2 = n_kv (valid positions), p3 = kv_mul
// p4 = kv_dim, p5 = seq_len, p6 = loff (layer offset in f32 elements), p7 = inv_sqrt_hs bits

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
    let scale = bitcast<f32>(u.p7);

    if (h >= n_heads) {
        return;
    }

    let kv_h = h / kv_mul;
    let q_base = h * head_size;

    // Each thread handles a range of KV positions
    var i = tid;
    while (i < n_kv) {
        // Ring buffer: oldest entry at (pos+1)%seq_len when cache is full
        // Caller passes n_kv = min(pos+1, seq_len), and we index linearly
        // with wrap: start = (pos - n_kv + 1 + seq_len) % seq_len is handled by caller
        // passing loff already adjusted, and t = i maps to sequential positions.
        // Actually: the host sets start in the cache. We just iterate 0..n_kv
        // and compute t = (start + i) % seq_len where start is embedded in loff logic.
        // Simplification: key_cache is laid out [seq_len * kv_dim] per layer.
        // Position t in cache = i (the host ensures linear layout or we wrap).
        // For ring buffer: t = i (positions stored at pos % seq_len).
        // We iterate all seq_len slots if n_kv == seq_len, or just 0..pos if n_kv < seq_len.

        let t = i;  // linear index into cache slots
        let k_base = loff + t * kv_dim + kv_h * head_size;

        var dot = 0.0;
        for (var d = 0u; d < head_size; d++) {
            dot += q[q_base + d] * key_cache[k_base + d];
        }
        att[h * seq_len + i] = dot * scale;

        i += 256u;
    }
}
