// Fused Rotary Position Encoding: apply RoPE to K and Q in one dispatch.
// Dispatch: (n_kv_heads + n_q_heads, 1, 1).
// Workgroups 0..n_kv_heads-1 process K, remaining workgroups process Q.

struct Uniforms {
    p0: u32, p1: u32, p2: u32, p3: u32,
    p4: u32, p5: u32, p6: u32, p7: u32,
}

@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read_write> k: array<f32>;
@group(0) @binding(2) var<storage, read> freq: array<f32>;
@group(0) @binding(3) var<uniform> u: Uniforms;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let h = wid.x;
    let tid = lid.x;
    let n_q_heads = u.p0;
    let head_size = u.p1;
    let pos = u.p2;
    let rope_dims = u.p3;
    let n_kv_heads = u.p4;
    let k_input_offset = u.p5;
    let total_heads = n_kv_heads + n_q_heads;

    if (h >= total_heads) {
        return;
    }

    var base: u32;
    var is_k = h < n_kv_heads;
    if (is_k) {
        base = k_input_offset + h * head_size;
    } else {
        base = (h - n_kv_heads) * head_size;
    }

    let half_rope = rope_dims / 2u;
    var i = tid;
    while (i < half_rope) {
        let angle = f32(pos) * freq[i];
        let cos_a = cos(angle);
        let sin_a = sin(angle);
        let idx = base + i * 2u;

        if (is_k) {
            let v0 = k[idx];
            let v1 = k[idx + 1u];
            k[idx] = v0 * cos_a - v1 * sin_a;
            k[idx + 1u] = v0 * sin_a + v1 * cos_a;
        } else {
            let v0 = q[idx];
            let v1 = q[idx + 1u];
            q[idx] = v0 * cos_a - v1 * sin_a;
            q[idx + 1u] = v0 * sin_a + v1 * cos_a;
        }

        i += 256u;
    }
}
