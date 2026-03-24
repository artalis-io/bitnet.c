// Q2_K TILED matvec — 2-bit k-quant, 256 elements per block, 84 bytes/block
// Layout: scales[16] (bytes 0-15), qs[64] (bytes 16-79), d FP16 (bytes 80-81), dmin FP16 (bytes 82-83)
//
// Tiled: TILE_ROWS=32, 8 threads per row, async (no per-block barriers).
// Dispatch: (ceil(rows / TILE_ROWS), n_tokens, 1)

const TILE_ROWS: u32 = 32u;
const WG_SIZE: u32 = 256u;
const THREADS_PER_ROW: u32 = 8u;
const ELEMS_PER_THREAD: u32 = 256u / THREADS_PER_ROW;
const QK_K: u32 = 256u;
const BLOCK_BYTES: u32 = 84u;

struct Uniforms { rows: u32, cols: u32, n_tokens: u32, extra: u32 }

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> u: Uniforms;

var<workgroup> reduce_buf: array<f32, 256>;

fn fp16_to_f32(bits: u32) -> f32 {
    let sign = (bits >> 15u) & 1u;
    let exp = (bits >> 10u) & 0x1Fu;
    let mant = bits & 0x3FFu;
    if (exp == 0u && mant == 0u) { return select(0.0, -0.0, sign == 1u); }
    let f_exp = f32(i32(exp) - 15 + 127);
    let f_bits = (sign << 31u) | (u32(f_exp) << 23u) | (mant << 13u);
    return bitcast<f32>(f_bits);
}

fn read_u8(offset: u32) -> u32 {
    let word = weights[offset >> 2u];
    return (word >> ((offset & 3u) * 8u)) & 0xFFu;
}

fn read_u16(offset: u32) -> u32 {
    let word = weights[offset >> 2u];
    let shift = (offset & 2u) * 8u;
    return (word >> shift) & 0xFFFFu;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let tile_start = select(wid.x * TILE_ROWS, (wid.x + wid.y * u.extra) * TILE_ROWS, u.extra > 0u);
    let token = select(wid.y, 0u, u.extra > 0u);
    let tid = lid.x;

    let local_row = tid / THREADS_PER_ROW;
    let local_elem = tid % THREADS_PER_ROW;
    let global_row = tile_start + local_row;

    let cols = u.cols;
    let n_blocks = cols / QK_K;
    let x_base = token * cols;
    let row_byte = global_row * n_blocks * BLOCK_BYTES;

    var acc: f32 = 0.0;

    if (global_row < u.rows) {
        for (var bi = 0u; bi < n_blocks; bi++) {
            let base = row_byte + bi * BLOCK_BYTES;
            let d    = fp16_to_f32(read_u16(base + 80u));
            let dmin = fp16_to_f32(read_u16(base + 82u));
            let qs_base = base + 16u;
            let elem_base = bi * QK_K;

            // Each thread handles 32 elements (256 / 8)
            let my_start = local_elem * ELEMS_PER_THREAD;

            for (var i = 0u; i < ELEMS_PER_THREAD; i++) {
                let elem = my_start + i;
                // Determine which 128-element half, sub-block, and position
                let half = elem / 128u;       // 0 or 1
                let in_half = elem % 128u;
                let sub_block = in_half / 16u; // 0..7 within half
                let pos = in_half % 16u;       // 0..15 within sub-block

                // Scale index: sequential through sub-blocks
                let is_idx = half * 8u + sub_block;
                let sc = read_u8(base + is_idx);
                let dl = d * f32(sc & 0xFu);
                let ml = dmin * f32(sc >> 4u);

                // Bit shift for 2-bit extraction depends on which quarter within half
                let shift = (sub_block / 2u) * 2u;

                // qs position: first or second 16 bytes of the half's 32 qs bytes
                let q_off = qs_base + half * 32u + (sub_block & 1u) * 16u + pos;
                let qbyte = read_u8(q_off);
                let qval = (qbyte >> shift) & 3u;

                acc += (dl * f32(qval) - ml) * x[x_base + elem_base + elem];
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

    if (local_elem == 0u && global_row < u.rows) {
        out[token * u.rows + global_row] = reduce_buf[row_base];
    }
}
