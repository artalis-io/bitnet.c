// In-place SiLU activation: x[i] = silu(x[i])

struct Uniforms {
    p0: u32, p1: u32, p2: u32, p3: u32,
    p4: u32, p5: u32, p6: u32, p7: u32,
}

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<uniform> u: Uniforms;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let gid = wid.x * 256u + lid.x;
    let dim = u.p0;
    if (gid >= dim) {
        return;
    }
    let v = x[gid];
    x[gid] = v / (1.0 + exp(-v));
}
