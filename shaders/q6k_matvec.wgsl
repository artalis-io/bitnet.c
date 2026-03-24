// Q6_K TILED matvec — 6-bit k-quant, 256 elements per block, 210 bytes/block
// Layout: ql[128] (bytes 0-127), qh[64] (bytes 128-191), scales[16] (bytes 192-207), d FP16 (bytes 208-209)
//
// Tiled: TILE_ROWS=32, 8 threads per row, async (no per-block barriers).
// Dispatch: (ceil(rows / TILE_ROWS), n_tokens, 1)

const TILE_ROWS: u32 = 32u;
const WG_SIZE: u32 = 256u;
const THREADS_PER_ROW: u32 = 8u;
const ELEMS_PER_THREAD: u32 = 256u / THREADS_PER_ROW;
const QK_K: u32 = 256u;
const BLOCK_BYTES: u32 = 210u;

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

fn read_i8(offset: u32) -> i32 {
    let v = read_u8(offset);
    return select(i32(v), i32(v) - 256, v >= 128u);
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
            let ql_base = base;
            let qh_base = base + 128u;
            let sc_base = base + 192u;
            let d = fp16_to_f32(read_u16(base + 208u));
            let elem_base = bi * QK_K;

            let my_start = local_elem * ELEMS_PER_THREAD;

            for (var i = 0u; i < ELEMS_PER_THREAD; i++) {
                let elem = my_start + i;
                // 256 elements in 2 halves of 128. Each half: 4 sub-groups of 32.
                // Element mapping: half*128 + subgroup -> maps to ql/qh/scale
                let half = elem / 128u;
                let in_half = elem % 128u;
                // Q6_K: within each 128-element half, elements map to (l, l+32, l+64, l+96)
                // where l = in_half % 32
                let quarter = in_half / 32u;   // 0..3
                let l = in_half % 32u;

                let ql_off = ql_base + half * 64u;
                let qh_off = qh_base + half * 32u;
                let sc_off = sc_base + half * 8u;

                let ql0 = read_u8(ql_off + l);
                let ql1 = read_u8(ql_off + l + 32u);
                let qh_val = read_u8(qh_off + l);

                var q6: i32;
                var s_idx: u32;
                switch quarter {
                    case 0u: {
                        q6 = i32((ql0 & 0xFu) | (((qh_val >> 0u) & 3u) << 4u)) - 32;
                        s_idx = l / 16u;
                    }
                    case 1u: {
                        q6 = i32((ql1 & 0xFu) | (((qh_val >> 2u) & 3u) << 4u)) - 32;
                        s_idx = l / 16u + 2u;
                    }
                    case 2u: {
                        q6 = i32((ql0 >> 4u) | (((qh_val >> 4u) & 3u) << 4u)) - 32;
                        s_idx = l / 16u + 4u;
                    }
                    default: { // case 3u
                        q6 = i32((ql1 >> 4u) | (((qh_val >> 6u) & 3u) << 4u)) - 32;
                        s_idx = l / 16u + 6u;
                    }
                }

                let sc = f32(read_i8(sc_off + s_idx));
                acc += d * sc * f32(q6) * x[x_base + elem_base + elem];
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
