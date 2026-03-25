// Q4_0 REPACKED matvec — simdgroup-coalesced, no x_cache barriers
//
// GPU buffer layout: [f32 scales: n_blocks][nibble u32s: n_blocks * 4]
//
// THREADS_PER_ROW=32 matches Apple GPU simdgroup size for coalesced loads.
// No shared x_cache: GPU L1 cache handles x reuse across rows.
// This eliminates ~16 workgroup barriers per dispatch for dim=2048.
// Only barriers: final reduction (5 steps).

const TILE_ROWS: u32 = 8u;
const WG_SIZE: u32 = 256u;
const THREADS_PER_ROW: u32 = 32u;

struct Uniforms {
    rows: u32,
    cols: u32,
    n_tokens: u32,
    extra: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

var<workgroup> reduce_buf: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let tile_start = select(wid.x * TILE_ROWS, (wid.x + wid.y * uniforms.extra) * TILE_ROWS, uniforms.extra > 0u);
    let token = select(wid.y, 0u, uniforms.extra > 0u);
    let tid = lid.x;

    let local_row = tid / THREADS_PER_ROW;
    let lane = tid % THREADS_PER_ROW;
    let global_row = tile_start + local_row;

    let cols = uniforms.cols;
    let blocks_per_row = cols / 32u;
    let total_blocks = uniforms.rows * blocks_per_row;
    let x_base = token * cols;

    var acc: f32 = 0.0;

    if (global_row < uniforms.rows) {
        let row_block_base = global_row * blocks_per_row;

        for (var b = 0u; b < blocks_per_row; b++) {
            let block_idx = row_block_base + b;
            let scale = bitcast<f32>(weights[block_idx]);

            let nib_base = total_blocks + block_idx * 4u;
            let word_idx = lane / 8u;
            let shift = (lane % 8u) * 4u;
            let nibble = (weights[nib_base + word_idx] >> shift) & 0xFu;

            let elem = b * 32u + lane;
            acc += scale * f32(i32(nibble) - 8) * x[x_base + elem];
        }
    }

    reduce_buf[tid] = acc;
    workgroupBarrier();

    let row_base = local_row * THREADS_PER_ROW;
    for (var s = THREADS_PER_ROW / 2u; s > 0u; s >>= 1u) {
        if (lane < s) {
            reduce_buf[row_base + lane] += reduce_buf[row_base + lane + s];
        }
        workgroupBarrier();
    }

    if (lane == 0u && global_row < uniforms.rows) {
        out[token * uniforms.rows + global_row] = reduce_buf[row_base];
    }
}
