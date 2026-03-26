// Sigmoid gate: out[i] *= sigmoid(gate[offset + i])
// Used for Q-gated attention: gate values are interleaved in the QKV buffer
// at positions head * 2 * head_size + head_size + d.
// Dispatch: (ceil(q_dim/256), 1, 1)

struct Uniforms {
    p0: u32, p1: u32, p2: u32, p3: u32,
    p4: u32, p5: u32, p6: u32, p7: u32,
}

@group(0) @binding(0) var<storage, read_write> out: array<f32>;  // attention output [q_dim]
@group(0) @binding(1) var<storage, read> gate: array<f32>;       // interleaved QKV [2*q_dim]
@group(0) @binding(2) var<uniform> u: Uniforms;

// p0 = q_dim, p1 = head_size

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let gid = wid.x * 256u + lid.x;
    let q_dim = u.p0;
    let hs = u.p1;

    if (gid >= q_dim) {
        return;
    }

    // Map flat index to gate position in interleaved layout
    let head = gid / hs;
    let d = gid % hs;
    let gate_idx = head * 2u * hs + hs + d;  // gate portion starts at offset hs within each head

    let g = gate[gate_idx];
    out[gid] *= 1.0 / (1.0 + exp(-g));
}
