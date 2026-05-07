// In-place ReLU^2 activation: x[i] = max(0, x[i])^2

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
    let v = max(0.0, x[gid]);
    x[gid] = v * v;
}
