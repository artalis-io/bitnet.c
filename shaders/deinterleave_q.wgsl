// Extract Q from interleaved [Q0, Gate0, Q1, Gate1, ...] layout.
// Copies head_size elements per head from stride-2*head_size source to contiguous dest.
// Dispatch: (ceil(q_dim/256), 1, 1)

struct Uniforms {
    p0: u32, p1: u32, p2: u32, p3: u32,
    p4: u32, p5: u32, p6: u32, p7: u32,
}

@group(0) @binding(0) var<storage, read> src: array<f32>;   // interleaved [2*q_dim]
@group(0) @binding(1) var<storage, read_write> dst: array<f32>; // contiguous [q_dim]
@group(0) @binding(2) var<uniform> u: Uniforms;

// p0 = q_dim (n_heads * head_size), p1 = head_size

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let gid = wid.x * 256u + lid.x;
    let q_dim = u.p0;
    let hs = u.p1;

    if (gid >= q_dim) {
        return;
    }

    // Map flat index to (head, offset_within_head)
    let head = gid / hs;
    let d = gid % hs;
    // Source: interleaved at head * 2 * hs + d (Q portion, not gate)
    dst[gid] = src[head * 2u * hs + d];
}
