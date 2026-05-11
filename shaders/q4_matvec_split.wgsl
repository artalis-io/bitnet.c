// Q4_0 repacked split matvec for stacked projection buffers.

const TILE_ROWS: u32 = 32u;
const THREADS_PER_ROW: u32 = 8u;

struct Uniforms {
    rows: u32,
    cols: u32,
    split1: u32,
    split2: u32,
    bias_offset: u32,
    off0: u32,
    off1: u32,
    off2: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out0: array<f32>;
@group(0) @binding(3) var<storage, read_write> out1: array<f32>;
@group(0) @binding(4) var<storage, read_write> out2: array<f32>;
@group(0) @binding(5) var<uniform> u: Uniforms;

var<workgroup> reduce_buf: array<f32, 256>;

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
    let total_blocks = u.rows * blocks_per_row;
    var acc: f32 = 0.0;

    if (global_row < u.rows) {
        let row_block_base = global_row * blocks_per_row;

        for (var b = row_lane; b < blocks_per_row; b += THREADS_PER_ROW) {
            let block_idx = row_block_base + b;
            let s = bitcast<f32>(weights[block_idx]);
            let nib_base = total_blocks + block_idx * 4u;
            let w0 = weights[nib_base];
            let w1 = weights[nib_base + 1u];
            let w2 = weights[nib_base + 2u];
            let w3 = weights[nib_base + 3u];
            let elem = b * 32u;

            let x0 = vec4<f32>(x[elem +  0u], x[elem +  1u], x[elem +  2u], x[elem +  3u]);
            let x1 = vec4<f32>(x[elem +  4u], x[elem +  5u], x[elem +  6u], x[elem +  7u]);
            let x2 = vec4<f32>(x[elem +  8u], x[elem +  9u], x[elem + 10u], x[elem + 11u]);
            let x3 = vec4<f32>(x[elem + 12u], x[elem + 13u], x[elem + 14u], x[elem + 15u]);
            let x4 = vec4<f32>(x[elem + 16u], x[elem + 17u], x[elem + 18u], x[elem + 19u]);
            let x5 = vec4<f32>(x[elem + 20u], x[elem + 21u], x[elem + 22u], x[elem + 23u]);
            let x6 = vec4<f32>(x[elem + 24u], x[elem + 25u], x[elem + 26u], x[elem + 27u]);
            let x7 = vec4<f32>(x[elem + 28u], x[elem + 29u], x[elem + 30u], x[elem + 31u]);

            acc += dot(dq4(w0,  0u, s), x0);
            acc += dot(dq4(w0, 16u, s), x1);
            acc += dot(dq4(w1,  0u, s), x2);
            acc += dot(dq4(w1, 16u, s), x3);
            acc += dot(dq4(w2,  0u, s), x4);
            acc += dot(dq4(w2, 16u, s), x5);
            acc += dot(dq4(w3,  0u, s), x6);
            acc += dot(dq4(w3, 16u, s), x7);
        }
    }

    reduce_buf[tid] = acc;
    workgroupBarrier();

    let row_base = local_row * THREADS_PER_ROW;
    if (row_lane < 4u) { reduce_buf[row_base + row_lane] += reduce_buf[row_base + row_lane + 4u]; }
    workgroupBarrier();
    if (row_lane < 2u) { reduce_buf[row_base + row_lane] += reduce_buf[row_base + row_lane + 2u]; }
    workgroupBarrier();
    if (row_lane < 1u) { reduce_buf[row_base] += reduce_buf[row_base + 1u]; }
    workgroupBarrier();

    if (row_lane == 0u && global_row < u.rows) {
        var result = reduce_buf[row_base];
        if (u.bias_offset > 0u) {
            result += bitcast<f32>(weights[u.bias_offset + global_row]);
        }
        if (u.split2 > 0u && global_row >= u.split2) {
            out2[u.off2 + global_row - u.split2] = result;
        } else if (global_row >= u.split1) {
            out1[u.off1 + global_row - u.split1] = result;
        } else {
            out0[u.off0 + global_row] = result;
        }
    }
}
