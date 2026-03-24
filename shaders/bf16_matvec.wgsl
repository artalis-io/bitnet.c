// BF16 TILED matvec — bfloat16 (no block structure)
// Process in tiles of 256 elements.
//
// Tiled: TILE_ROWS=32, 8 threads per row, async (no per-block barriers).
// Dispatch: (ceil(rows / TILE_ROWS), n_tokens, 1)

const TILE_ROWS: u32 = 32u;
const WG_SIZE: u32 = 256u;
const THREADS_PER_ROW: u32 = 8u;
const ELEMS_PER_THREAD: u32 = 256u / THREADS_PER_ROW;
const BLOCK_SIZE: u32 = 256u;

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

fn bf16_to_f32(bits: u32) -> f32 {
    return bitcast<f32>(bits << 16u);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let tile_start = select(wid.x * TILE_ROWS, (wid.x + wid.y * uniforms.extra) * TILE_ROWS, uniforms.extra > 0u);
    let token = select(wid.y, 0u, uniforms.extra > 0u);
    let tid = lid.x;

    let local_row = tid / THREADS_PER_ROW;
    let local_elem = tid % THREADS_PER_ROW;
    let global_row = tile_start + local_row;

    let cols = uniforms.cols;
    let blocks_per_row = cols / BLOCK_SIZE;
    let x_base = token * cols;

    // BF16: 2 bytes per element, packed as u32 (2 values per word)
    // Row offset in u32s: row * cols / 2
    let row_u32_offset = global_row * (cols / 2u);

    var acc: f32 = 0.0;

    if (global_row < uniforms.rows) {
        for (var b = 0u; b < blocks_per_row; b++) {
            let elem_base = b * BLOCK_SIZE;
            // Each thread handles 32 elements (256 / 8)
            let my_start = local_elem * ELEMS_PER_THREAD;
            for (var i = 0u; i < ELEMS_PER_THREAD; i += 2u) {
                let elem = my_start + i;
                let pair_idx = (b * BLOCK_SIZE + elem) / 2u;
                let word = weights[row_u32_offset + pair_idx];
                let v0 = bf16_to_f32(word & 0xFFFFu);
                let v1 = bf16_to_f32((word >> 16u) & 0xFFFFu);
                acc += v0 * x[x_base + elem_base + elem];
                acc += v1 * x[x_base + elem_base + elem + 1u];
            }
        }
    }

    reduce_buf[tid] = acc;
    workgroupBarrier();

    let row_base = local_row * THREADS_PER_ROW;
    if (local_elem < 4u) { reduce_buf[row_base + local_elem] += reduce_buf[row_base + local_elem + 4u]; }
    workgroupBarrier();
    if (local_elem < 2u) { reduce_buf[row_base + local_elem] += reduce_buf[row_base + local_elem + 2u]; }
    workgroupBarrier();
    if (local_elem < 1u) { reduce_buf[row_base + local_elem] += reduce_buf[row_base + local_elem + 1u]; }
    workgroupBarrier();

    if (local_elem == 0u && global_row < uniforms.rows) {
        out[token * uniforms.rows + global_row] = reduce_buf[row_base];
    }
}
