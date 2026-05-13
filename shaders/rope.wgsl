// Rotary Position Encoding: rotate Q or K vector by position-dependent cos/sin
// Dispatch: (n_heads, 1, 1) — one workgroup per head

struct Uniforms {
    p0: u32, p1: u32, p2: u32, p3: u32,
    p4: u32, p5: u32, p6: u32, p7: u32,
}

@group(0) @binding(0) var<storage, read_write> vec: array<f32>;
@group(0) @binding(1) var<storage, read> freq: array<f32>;
@group(0) @binding(2) var<uniform> u: Uniforms;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let h = wid.x;
    let tid = lid.x;
    let n_heads = u.p0;
    let head_size = u.p1;
    let pos = u.p2;
    let rope_dims = u.p3;
    let input_offset = u.p4;

    if (h >= n_heads) {
        return;
    }

    let base = input_offset + h * head_size;
    let half_rope = rope_dims / 2u;

    // Each thread handles pairs at stride 256
    var i = tid;
    while (i < half_rope) {
        let angle = f32(pos) * freq[i];
        let cos_a = cos(angle);
        let sin_a = sin(angle);
        let idx0 = base + i;
        let idx1 = idx0 + half_rope;
        let v0 = vec[idx0];
        let v1 = vec[idx1];
        vec[idx0] = v0 * cos_a - v1 * sin_a;
        vec[idx1] = v0 * sin_a + v1 * cos_a;
        i += 256u;
    }
}
