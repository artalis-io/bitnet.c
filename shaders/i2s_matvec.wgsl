// I2_S TILED matvec — 2-bit interleaved ternary quantization
// 128-element chunks, 32 bytes per chunk
// Per-tensor scale stored at end of weight data
//
// Tiled: TILE_ROWS=32, 8 threads per row, async (no per-block barriers).
// Dispatch: (ceil(rows / TILE_ROWS), n_tokens, 1)

const TILE_ROWS: u32 = 32u;
const WG_SIZE: u32 = 256u;
const THREADS_PER_ROW: u32 = 8u;
const ELEMS_PER_THREAD: u32 = 128u / THREADS_PER_ROW;
const CHUNK_SIZE: u32 = 128u;

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
    let local_elem = tid % THREADS_PER_ROW;
    let global_row = tile_start + local_row;

    let cols = uniforms.cols;
    let chunks_per_row = cols / CHUNK_SIZE;
    let x_base = token * cols;

    // Per-tensor scale at end of weight data
    let scale_offset = uniforms.rows * cols / 16u;
    let scale = bitcast<f32>(weights[scale_offset]);

    // I2_S: 128 elements per chunk, 32 bytes = 8 u32s per chunk
    let u32s_per_chunk = 8u;
    let row_offset_u32 = global_row * chunks_per_row * u32s_per_chunk;

    var acc: f32 = 0.0;

    if (global_row < uniforms.rows) {
        for (var c = 0u; c < chunks_per_row; c++) {
            let chunk_u32_offset = row_offset_u32 + c * u32s_per_chunk;
            let elem_base = c * CHUNK_SIZE;

            // Each thread handles 16 elements (128 / 8)
            let my_start = local_elem * ELEMS_PER_THREAD;

            for (var i = 0u; i < ELEMS_PER_THREAD; i++) {
                let elem = my_start + i;
                // Which u32 and which 2-bit position within that u32
                let w_idx = elem / 16u;  // which u32 (0..7)
                let byte_in_word = (elem / 4u) % 4u;
                let bit_in_byte = elem % 4u;

                let word = weights[chunk_u32_offset + w_idx];
                let byte_val = (word >> (byte_in_word * 8u)) & 0xFFu;
                let v = (byte_val >> (bit_in_byte * 2u)) & 3u;

                // Map: 0→-1, 1→0, 2→+1, 3→0
                let dv = select(f32(i32(v) - 1), 0.0, v == 3u);
                acc += dv * x[x_base + elem_base + elem];
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
        out[token * uniforms.rows + global_row] = reduce_buf[row_base] * scale;
    }
}
