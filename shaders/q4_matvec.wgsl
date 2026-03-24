// Q4_0 TILED matvec — branchless decode, no per-block barriers
//
// Each thread processes all blocks for its row independently.
// No shared x_cache, no synchronous iteration.
// Only barriers: final reduction.

const TILE_ROWS: u32 = 32u;
const WG_SIZE: u32 = 256u;
const THREADS_PER_ROW: u32 = 8u;
const ELEMS_PER_THREAD: u32 = 32u / THREADS_PER_ROW;

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

fn fp16_to_f32(bits: u32) -> f32 {
    let sign = (bits >> 15u) & 1u;
    let exp = (bits >> 10u) & 0x1Fu;
    let mant = bits & 0x3FFu;
    if (exp == 0u && mant == 0u) {
        return select(0.0, -0.0, sign == 1u);
    }
    let f_bits = (sign << 31u) | ((exp + 112u) << 23u) | (mant << 13u);
    return bitcast<f32>(f_bits);
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
    let blocks_per_row = cols / 32u;
    let x_base = token * cols;
    let row_byte_base = global_row * blocks_per_row * 18u;
    let my_start = local_elem * ELEMS_PER_THREAD;

    var acc: f32 = 0.0;

    if (global_row < uniforms.rows) {
        for (var b = 0u; b < blocks_per_row; b++) {
            let block_byte = row_byte_base + b * 18u;

            // Read FP16 scale
            let sw = weights[block_byte >> 2u];
            let ss = (block_byte & 3u) * 8u;
            let sbits = select(
                (sw >> ss) & 0xFFFFu,
                (sw >> 24u) | ((weights[(block_byte >> 2u) + 1u] & 0xFFu) << 8u),
                ss > 16u
            );
            let scale = fp16_to_f32(sbits);

            let qs_base = block_byte + 2u;
            let elem_base = b * 32u;

            for (var i = 0u; i < ELEMS_PER_THREAD; i++) {
                let elem = my_start + i;
                let byte_idx = elem & 15u;
                let addr = qs_base + byte_idx;
                let byte_val = (weights[addr >> 2u] >> ((addr & 3u) * 8u)) & 0xFFu;
                let nibble = select(byte_val & 0xFu, (byte_val >> 4u) & 0xFu, elem >= 16u);
                acc += scale * f32(i32(nibble) - 8) * x[x_base + elem_base + elem];
            }
        }
    }

    reduce_buf[tid] = acc;
    workgroupBarrier();

    let row_base = local_row * THREADS_PER_ROW;
    for (var s = THREADS_PER_ROW / 2u; s > 0u; s >>= 1u) {
        if (local_elem < s) {
            reduce_buf[row_base + local_elem] += reduce_buf[row_base + local_elem + s];
        }
        workgroupBarrier();
    }

    if (local_elem == 0u && global_row < uniforms.rows) {
        out[token * uniforms.rows + global_row] = reduce_buf[row_base];
    }
}
