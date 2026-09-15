// Weighted addition: x[i] = clear ? weight * r[i] : x[i] + weight * r[i]
// Used for MoE expert accumulation.
// Dispatch: (ceil(dim/256), 1, 1)

struct Uniforms {
    p0: u32, p1: u32, p2: u32, p3: u32,
    p4: u32, p5: u32, p6: u32, p7: u32,
}

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> r: array<f32>;
@group(0) @binding(2) var<uniform> u: Uniforms;

// p0 = dim, p1 = weight (bitcast to f32), p2 = clear

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let gid = wid.x * 256u + lid.x;
    let dim = u.p0;

    if (gid >= dim) {
        return;
    }

    let weight = bitcast<f32>(u.p1);
    let weighted = weight * r[gid];
    if (u.p2 != 0u) {
        x[gid] = weighted;
    } else {
        x[gid] += weighted;
    }
}
