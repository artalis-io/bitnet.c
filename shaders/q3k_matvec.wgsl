// Q3_K TILED matvec — 3-bit k-quant, 256 elements per block, 110 bytes/block
// Layout: hmask[32] (bytes 0-31), qs[64] (bytes 32-95), scales[12] (bytes 96-107), d FP16 (bytes 108-109)
//
// Tiled: TILE_ROWS=32, 8 threads per row, async (no per-block barriers).
// Dispatch: (ceil(rows / TILE_ROWS), n_tokens, 1)

const TILE_ROWS: u32 = 32u;
const WG_SIZE: u32 = 256u;
const THREADS_PER_ROW: u32 = 8u;
const ELEMS_PER_THREAD: u32 = 256u / THREADS_PER_ROW;
const QK_K: u32 = 256u;
const BLOCK_BYTES: u32 = 110u;

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

fn unpack_q3k_scale(scales_base: u32, idx: u32) -> u32 {
    let b0 = read_u8(scales_base + 0u);
    let b1 = read_u8(scales_base + 1u);
    let b2 = read_u8(scales_base + 2u);
    let b3 = read_u8(scales_base + 3u);
    let aux0 = b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);

    let b4 = read_u8(scales_base + 4u);
    let b5 = read_u8(scales_base + 5u);
    let b6 = read_u8(scales_base + 6u);
    let b7 = read_u8(scales_base + 7u);
    let aux1 = b4 | (b5 << 8u) | (b6 << 16u) | (b7 << 24u);

    let b8 = read_u8(scales_base + 8u);
    let b9 = read_u8(scales_base + 9u);
    let b10 = read_u8(scales_base + 10u);
    let b11 = read_u8(scales_base + 11u);
    let tmp = b8 | (b9 << 8u) | (b10 << 16u) | (b11 << 24u);

    var r: array<u32, 4>;
    r[2] = ((aux0 >> 4u) & 0x0F0F0F0Fu) | (((tmp >> 4u) & 0x03030303u) << 4u);
    r[3] = ((aux1 >> 4u) & 0x0F0F0F0Fu) | (((tmp >> 6u) & 0x03030303u) << 4u);
    r[0] = (aux0 & 0x0F0F0F0Fu)          | (((tmp >> 0u) & 0x03030303u) << 4u);
    r[1] = (aux1 & 0x0F0F0F0Fu)          | (((tmp >> 2u) & 0x03030303u) << 4u);

    let word_idx = idx / 4u;
    let byte_idx = idx % 4u;
    return (r[word_idx] >> (byte_idx * 8u)) & 0x3Fu;
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
            let hmask_base  = base;
            let qs_base     = base + 32u;
            let scales_base = base + 96u;
            let d = fp16_to_f32(read_u16(base + 108u));
            let elem_base = bi * QK_K;

            // Each thread handles 32 elements (256 / 8)
            let my_start = local_elem * ELEMS_PER_THREAD;

            for (var i = 0u; i < ELEMS_PER_THREAD; i++) {
                let elem = my_start + i;
                // Determine half (0 or 1), sub-block position
                let half = elem / 128u;
                let in_half = elem % 128u;
                let sub_block = in_half / 16u; // 0..7
                let pos = in_half % 16u;

                // Scale index
                let is_idx = half * 8u + sub_block;
                let sc = unpack_q3k_scale(scales_base, is_idx);
                let dl = d * f32(i32(sc) - 32);

                // 2-bit shift
                let shift = (sub_block / 2u) * 2u;

                // qs position
                let q_off = qs_base + half * 32u + (sub_block & 1u) * 16u + pos;
                let low2 = (read_u8(q_off) >> shift) & 3u;

                // hmask bit: which bit within the byte
                let hmask_byte_pos = (sub_block & 1u) * 16u + pos;
                let m = 1u << (half * 4u + sub_block / 2u);
                let hbit = read_u8(hmask_base + hmask_byte_pos) & m;

                var q3 = i32(low2);
                if (hbit == 0u) { q3 -= 4; }

                acc += dl * f32(q3) * x[x_base + elem_base + elem];
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
