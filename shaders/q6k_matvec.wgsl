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

struct Uniforms {
    rows: u32,
    cols: u32,
    n_tokens: u32,
    extra: u32,
    out_offset: u32,
    _pad5: u32,
    _pad6: u32,
    _pad7: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> u: Uniforms;

var<workgroup> reduce_buf: array<f32, 256>;

fn fp16_to_f32(bits: u32) -> f32 {
    let sign = (bits >> 15u) & 1u;
    let exp = (bits >> 10u) & 0x1Fu;
    let mant = bits & 0x3FFu;
    if (exp == 0u) {
        if (mant == 0u) { return select(0.0, -0.0, sign == 1u); }
        let val = f32(mant) * 5.9604644775390625e-8;
        return select(val, -val, sign == 1u);
    }
    let f_bits = (sign << 31u) | ((exp + 112u) << 23u) | (mant << 13u);
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

fn q6_value(ql: u32, qh: u32, qh_shift: u32, ql_high: bool) -> f32 {
    let qlo = select(ql & 0xFu, ql >> 4u, ql_high);
    let q6 = i32(qlo | (((qh >> qh_shift) & 3u) << 4u)) - 32;
    return f32(q6);
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
    let my_start = local_elem * ELEMS_PER_THREAD;
    let half_idx = my_start / 128u;
    let quarter = (my_start % 128u) / 32u;
    let ql_off = half_idx * 64u;
    let qh_off = half_idx * 32u;
    let sc_off = half_idx * 8u;
    var s_base: u32 = 0u;
    var qh_shift: u32 = 0u;
    var ql_add: u32 = 0u;
    var ql_high: bool = false;
    switch quarter {
        case 0u: {
            s_base = 0u; qh_shift = 0u; ql_add = 0u; ql_high = false;
        }
        case 1u: {
            s_base = 2u; qh_shift = 2u; ql_add = 32u; ql_high = false;
        }
        case 2u: {
            s_base = 4u; qh_shift = 4u; ql_add = 0u; ql_high = true;
        }
        default: {
            s_base = 6u; qh_shift = 6u; ql_add = 32u; ql_high = true;
        }
    }

    var acc: f32 = 0.0;

    if (global_row < u.rows) {
        for (var bi = 0u; bi < n_blocks; bi++) {
            let base = row_byte + bi * BLOCK_BYTES;
            let ql_base = base;
            let qh_base = base + 128u;
            let sc_base = base + 192u;
            let d = fp16_to_f32(read_u16(base + 208u));
            let elem_base = bi * QK_K;
            let qlp = ql_base + ql_off + ql_add;
            let qhp = qh_base + qh_off;
            let scp = sc_base + sc_off;
            let scale0 = d * f32(read_i8(scp + s_base));
            let scale1 = d * f32(read_i8(scp + s_base + 1u));
            let xb = x_base + elem_base + my_start;

            let q0 = vec4<f32>(
                q6_value(read_u8(qlp + 0u), read_u8(qhp + 0u), qh_shift, ql_high),
                q6_value(read_u8(qlp + 1u), read_u8(qhp + 1u), qh_shift, ql_high),
                q6_value(read_u8(qlp + 2u), read_u8(qhp + 2u), qh_shift, ql_high),
                q6_value(read_u8(qlp + 3u), read_u8(qhp + 3u), qh_shift, ql_high));
            let q1 = vec4<f32>(
                q6_value(read_u8(qlp + 4u), read_u8(qhp + 4u), qh_shift, ql_high),
                q6_value(read_u8(qlp + 5u), read_u8(qhp + 5u), qh_shift, ql_high),
                q6_value(read_u8(qlp + 6u), read_u8(qhp + 6u), qh_shift, ql_high),
                q6_value(read_u8(qlp + 7u), read_u8(qhp + 7u), qh_shift, ql_high));
            let q2 = vec4<f32>(
                q6_value(read_u8(qlp + 8u), read_u8(qhp + 8u), qh_shift, ql_high),
                q6_value(read_u8(qlp + 9u), read_u8(qhp + 9u), qh_shift, ql_high),
                q6_value(read_u8(qlp + 10u), read_u8(qhp + 10u), qh_shift, ql_high),
                q6_value(read_u8(qlp + 11u), read_u8(qhp + 11u), qh_shift, ql_high));
            let q3 = vec4<f32>(
                q6_value(read_u8(qlp + 12u), read_u8(qhp + 12u), qh_shift, ql_high),
                q6_value(read_u8(qlp + 13u), read_u8(qhp + 13u), qh_shift, ql_high),
                q6_value(read_u8(qlp + 14u), read_u8(qhp + 14u), qh_shift, ql_high),
                q6_value(read_u8(qlp + 15u), read_u8(qhp + 15u), qh_shift, ql_high));
            let q4 = vec4<f32>(
                q6_value(read_u8(qlp + 16u), read_u8(qhp + 16u), qh_shift, ql_high),
                q6_value(read_u8(qlp + 17u), read_u8(qhp + 17u), qh_shift, ql_high),
                q6_value(read_u8(qlp + 18u), read_u8(qhp + 18u), qh_shift, ql_high),
                q6_value(read_u8(qlp + 19u), read_u8(qhp + 19u), qh_shift, ql_high));
            let q5 = vec4<f32>(
                q6_value(read_u8(qlp + 20u), read_u8(qhp + 20u), qh_shift, ql_high),
                q6_value(read_u8(qlp + 21u), read_u8(qhp + 21u), qh_shift, ql_high),
                q6_value(read_u8(qlp + 22u), read_u8(qhp + 22u), qh_shift, ql_high),
                q6_value(read_u8(qlp + 23u), read_u8(qhp + 23u), qh_shift, ql_high));
            let q6 = vec4<f32>(
                q6_value(read_u8(qlp + 24u), read_u8(qhp + 24u), qh_shift, ql_high),
                q6_value(read_u8(qlp + 25u), read_u8(qhp + 25u), qh_shift, ql_high),
                q6_value(read_u8(qlp + 26u), read_u8(qhp + 26u), qh_shift, ql_high),
                q6_value(read_u8(qlp + 27u), read_u8(qhp + 27u), qh_shift, ql_high));
            let q7 = vec4<f32>(
                q6_value(read_u8(qlp + 28u), read_u8(qhp + 28u), qh_shift, ql_high),
                q6_value(read_u8(qlp + 29u), read_u8(qhp + 29u), qh_shift, ql_high),
                q6_value(read_u8(qlp + 30u), read_u8(qhp + 30u), qh_shift, ql_high),
                q6_value(read_u8(qlp + 31u), read_u8(qhp + 31u), qh_shift, ql_high));

            let x0 = vec4<f32>(x[xb + 0u], x[xb + 1u], x[xb + 2u], x[xb + 3u]);
            let x1 = vec4<f32>(x[xb + 4u], x[xb + 5u], x[xb + 6u], x[xb + 7u]);
            let x2 = vec4<f32>(x[xb + 8u], x[xb + 9u], x[xb + 10u], x[xb + 11u]);
            let x3 = vec4<f32>(x[xb + 12u], x[xb + 13u], x[xb + 14u], x[xb + 15u]);
            let x4 = vec4<f32>(x[xb + 16u], x[xb + 17u], x[xb + 18u], x[xb + 19u]);
            let x5 = vec4<f32>(x[xb + 20u], x[xb + 21u], x[xb + 22u], x[xb + 23u]);
            let x6 = vec4<f32>(x[xb + 24u], x[xb + 25u], x[xb + 26u], x[xb + 27u]);
            let x7 = vec4<f32>(x[xb + 28u], x[xb + 29u], x[xb + 30u], x[xb + 31u]);

            acc += scale0 * (dot(q0, x0) + dot(q1, x1) + dot(q2, x2) + dot(q3, x3)) +
                   scale1 * (dot(q4, x4) + dot(q5, x5) + dot(q6, x6) + dot(q7, x7));
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
        out[u.out_offset + token * u.rows + global_row] = reduce_buf[row_base];
    }
}
