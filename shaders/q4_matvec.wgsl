// Q4_0 REPACKED matvec — vec4 vectorized, 32 rows/tile, 8 threads/row
//
// GPU buffer layout: [f32 scales: n_blocks][nibble u32s: n_blocks * 4]
//
// 8 threads per row, each processes blocks_per_row/8 blocks.
// Each block: 8 dot(vec4) operations instead of 32 scalar multiply-adds.
//
// Dispatch: (ceil(rows/32), n_tokens, 1)

const TILE_ROWS: u32 = 32u;
const WG_SIZE: u32 = 256u;
const THREADS_PER_ROW: u32 = 8u;

struct Uniforms {
    rows: u32,
    cols: u32,
    n_tokens: u32,
    extra: u32,
    bias_offset: u32,
    out_offset: u32,
    _pad6: u32,
    _pad7: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

var<workgroup> reduce_buf: array<f32, 256>;

// Dequantize 4 nibbles from u32 word at bit offset sh
fn dq4(w: u32, sh: u32, s: f32) -> vec4<f32> {
    return s * vec4<f32>(
        f32(i32((w >> sh)        & 0xFu) - 8),
        f32(i32((w >> (sh + 4u)) & 0xFu) - 8),
        f32(i32((w >> (sh + 8u)) & 0xFu) - 8),
        f32(i32((w >> (sh + 12u))& 0xFu) - 8));
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let tile_start = select(wid.x * TILE_ROWS, (wid.x + wid.y * uniforms.extra) * TILE_ROWS, uniforms.extra > 0u);
    let token = select(wid.y, 0u, uniforms.extra > 0u);
    let tid = lid.x;

    let row_lane = tid & 7u;
    let local_row = tid >> 3u;
    let global_row = tile_start + local_row;

    let cols = uniforms.cols;
    let blocks_per_row = cols >> 5u;
    let total_blocks = uniforms.rows * blocks_per_row;
    let x_base = token * cols;

    var acc: f32 = 0.0;

    if (global_row < uniforms.rows) {
        let row_block_base = global_row * blocks_per_row;

        for (var b = row_lane; b < blocks_per_row; b += 8u) {
            let block_idx = row_block_base + b;
            let s = bitcast<f32>(weights[block_idx]);

            let nib_base = total_blocks + block_idx * 4u;
            let w0 = weights[nib_base];
            let w1 = weights[nib_base + 1u];
            let w2 = weights[nib_base + 2u];
            let w3 = weights[nib_base + 3u];

            let elem = x_base + b * 32u;

            // vec4 loads + dot products (8 instead of 32 scalar ops)
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
    if (row_lane < 4u) {
        reduce_buf[row_base + row_lane] += reduce_buf[row_base + row_lane + 4u];
    }
    workgroupBarrier();
    if (row_lane < 2u) {
        reduce_buf[row_base + row_lane] += reduce_buf[row_base + row_lane + 2u];
    }
    workgroupBarrier();
    if (row_lane < 1u) {
        reduce_buf[row_base + row_lane] += reduce_buf[row_base + row_lane + 1u];
    }
    workgroupBarrier();

    if (row_lane == 0u && global_row < uniforms.rows) {
        var result = reduce_buf[row_base];
        if (uniforms.bias_offset > 0u) {
            result += bitcast<f32>(weights[uniforms.bias_offset + global_row]);
        }
        out[uniforms.out_offset + token * uniforms.rows + global_row] = result;
    }
}
