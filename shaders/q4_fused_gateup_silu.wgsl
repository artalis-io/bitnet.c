// Q4_0 repacked fused gate/up matvec with SiLU activation.

const TILE_ROWS: u32 = 32u;
const THREADS_PER_ROW: u32 = 8u;

struct Uniforms {
    total_rows: u32,
    cols: u32,
    gate_rows: u32,
    _pad3: u32,
    bias_offset: u32,
    _pad5: u32,
    _pad6: u32,
    _pad7: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> u: Uniforms;

var<workgroup> gate_reduce: array<f32, 256>;
var<workgroup> up_reduce: array<f32, 256>;

fn dq4(w: u32, sh: u32, s: f32) -> vec4<f32> {
    return s * vec4<f32>(
        f32(i32((w >> sh) & 0xFu) - 8),
        f32(i32((w >> (sh + 4u)) & 0xFu) - 8),
        f32(i32((w >> (sh + 8u)) & 0xFu) - 8),
        f32(i32((w >> (sh + 12u)) & 0xFu) - 8));
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let row_lane = tid & 7u;
    let local_row = tid >> 3u;
    let global_row = wid.x * TILE_ROWS + local_row;
    let blocks_per_row = u.cols >> 5u;
    let total_blocks = u.total_rows * blocks_per_row;

    var gate_acc: f32 = 0.0;
    var up_acc: f32 = 0.0;
    if (global_row < u.gate_rows) {
        let gate_row_base = global_row * blocks_per_row;
        let up_row_base = (global_row + u.gate_rows) * blocks_per_row;
        for (var b = row_lane; b < blocks_per_row; b += THREADS_PER_ROW) {
            let gate_block = gate_row_base + b;
            let up_block = up_row_base + b;
            let gate_s = bitcast<f32>(weights[gate_block]);
            let up_s = bitcast<f32>(weights[up_block]);
            let gate_nib = total_blocks + gate_block * 4u;
            let up_nib = total_blocks + up_block * 4u;

            let gw0 = weights[gate_nib];
            let gw1 = weights[gate_nib + 1u];
            let gw2 = weights[gate_nib + 2u];
            let gw3 = weights[gate_nib + 3u];
            let uw0 = weights[up_nib];
            let uw1 = weights[up_nib + 1u];
            let uw2 = weights[up_nib + 2u];
            let uw3 = weights[up_nib + 3u];
            let elem = b * 32u;

            let x0 = vec4<f32>(x[elem +  0u], x[elem +  1u], x[elem +  2u], x[elem +  3u]);
            let x1 = vec4<f32>(x[elem +  4u], x[elem +  5u], x[elem +  6u], x[elem +  7u]);
            let x2 = vec4<f32>(x[elem +  8u], x[elem +  9u], x[elem + 10u], x[elem + 11u]);
            let x3 = vec4<f32>(x[elem + 12u], x[elem + 13u], x[elem + 14u], x[elem + 15u]);
            let x4 = vec4<f32>(x[elem + 16u], x[elem + 17u], x[elem + 18u], x[elem + 19u]);
            let x5 = vec4<f32>(x[elem + 20u], x[elem + 21u], x[elem + 22u], x[elem + 23u]);
            let x6 = vec4<f32>(x[elem + 24u], x[elem + 25u], x[elem + 26u], x[elem + 27u]);
            let x7 = vec4<f32>(x[elem + 28u], x[elem + 29u], x[elem + 30u], x[elem + 31u]);

            gate_acc += dot(dq4(gw0,  0u, gate_s), x0);
            gate_acc += dot(dq4(gw0, 16u, gate_s), x1);
            gate_acc += dot(dq4(gw1,  0u, gate_s), x2);
            gate_acc += dot(dq4(gw1, 16u, gate_s), x3);
            gate_acc += dot(dq4(gw2,  0u, gate_s), x4);
            gate_acc += dot(dq4(gw2, 16u, gate_s), x5);
            gate_acc += dot(dq4(gw3,  0u, gate_s), x6);
            gate_acc += dot(dq4(gw3, 16u, gate_s), x7);

            up_acc += dot(dq4(uw0,  0u, up_s), x0);
            up_acc += dot(dq4(uw0, 16u, up_s), x1);
            up_acc += dot(dq4(uw1,  0u, up_s), x2);
            up_acc += dot(dq4(uw1, 16u, up_s), x3);
            up_acc += dot(dq4(uw2,  0u, up_s), x4);
            up_acc += dot(dq4(uw2, 16u, up_s), x5);
            up_acc += dot(dq4(uw3,  0u, up_s), x6);
            up_acc += dot(dq4(uw3, 16u, up_s), x7);
        }
    }

    gate_reduce[tid] = gate_acc;
    up_reduce[tid] = up_acc;
    workgroupBarrier();

    let row_base = local_row * THREADS_PER_ROW;
    if (row_lane < 4u) {
        gate_reduce[row_base + row_lane] += gate_reduce[row_base + row_lane + 4u];
        up_reduce[row_base + row_lane] += up_reduce[row_base + row_lane + 4u];
    }
    workgroupBarrier();
    if (row_lane < 2u) {
        gate_reduce[row_base + row_lane] += gate_reduce[row_base + row_lane + 2u];
        up_reduce[row_base + row_lane] += up_reduce[row_base + row_lane + 2u];
    }
    workgroupBarrier();
    if (row_lane < 1u) {
        gate_reduce[row_base] += gate_reduce[row_base + 1u];
        up_reduce[row_base] += up_reduce[row_base + 1u];
    }
    workgroupBarrier();

    if (row_lane == 0u && global_row < u.gate_rows) {
        var gate = gate_reduce[row_base];
        var up = up_reduce[row_base];
        if (u.bias_offset > 0u) {
            gate += bitcast<f32>(weights[u.bias_offset + global_row]);
            up += bitcast<f32>(weights[u.bias_offset + global_row + u.gate_rows]);
        }
        out[global_row] = (gate / (1.0 + exp(-gate))) * up;
    }
}
