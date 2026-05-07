// Q5_K split matvec for packed QKV projection buffers.

const TILE_ROWS: u32 = 32u;
const THREADS_PER_ROW: u32 = 8u;
const ELEMS_PER_THREAD: u32 = 256u / THREADS_PER_ROW;
const QK_K: u32 = 256u;
const BLOCK_BYTES: u32 = 176u;

struct Uniforms {
    rows: u32,
    cols: u32,
    split1: u32,
    split2: u32,
    _pad4: u32,
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

fn read_u16(offset: u32) -> u32 {
    let word = weights[offset >> 2u];
    let shift = (offset & 2u) * 8u;
    return (word >> shift) & 0xFFFFu;
}

fn get_scale_min(j: u32, scales_base: u32) -> vec2<u32> {
    var sc: u32;
    var m: u32;
    if (j < 4u) {
        sc = read_u8(scales_base + j) & 63u;
        m  = read_u8(scales_base + j + 4u) & 63u;
    } else {
        sc = (read_u8(scales_base + j + 4u) & 0xFu) | ((read_u8(scales_base + j - 4u) >> 6u) << 4u);
        m  = (read_u8(scales_base + j + 4u) >> 4u)   | ((read_u8(scales_base + j) >> 6u) << 4u);
    }
    return vec2<u32>(sc, m);
}

fn q5_value(qbyte: u32, hbyte: u32, bit: u32, is_high: bool) -> f32 {
    let nibble = select(qbyte >> 4u, qbyte & 0xFu, !is_high);
    let hi = (hbyte >> bit) & 1u;
    return f32(nibble | (hi << 4u));
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let tile_start = wid.x * TILE_ROWS;
    let tid = lid.x;
    let local_row = tid / THREADS_PER_ROW;
    let local_elem = tid % THREADS_PER_ROW;
    let global_row = tile_start + local_row;
    let n_blocks = u.cols / QK_K;
    let row_byte = global_row * n_blocks * BLOCK_BYTES;
    let my_start = local_elem * ELEMS_PER_THREAD;
    let group = my_start / 64u;
    let is_high = (my_start % 64u) / 32u;
    let bit = group * 2u + is_high;
    let q_off_base = group * 32u;

    var acc: f32 = 0.0;
    if (global_row < u.rows) {
        for (var bi = 0u; bi < n_blocks; bi++) {
            let base = row_byte + bi * BLOCK_BYTES;
            let d = fp16_to_f32(read_u16(base));
            let dmin = fp16_to_f32(read_u16(base + 2u));
            let scales_base = base + 4u;
            let qh_base = base + 16u;
            let qs_base = base + 48u;
            let elem_base = bi * QK_K;
            let sm = get_scale_min(bit, scales_base);
            let ds = d * f32(sm.x);
            let dm = dmin * f32(sm.y);
            let q_off = qs_base + q_off_base;
            let is_hi = is_high != 0u;
            let xb = elem_base + my_start;

            let q0 = vec4<f32>(
                q5_value(read_u8(q_off + 0u), read_u8(qh_base + 0u), bit, is_hi),
                q5_value(read_u8(q_off + 1u), read_u8(qh_base + 1u), bit, is_hi),
                q5_value(read_u8(q_off + 2u), read_u8(qh_base + 2u), bit, is_hi),
                q5_value(read_u8(q_off + 3u), read_u8(qh_base + 3u), bit, is_hi));
            let q1 = vec4<f32>(
                q5_value(read_u8(q_off + 4u), read_u8(qh_base + 4u), bit, is_hi),
                q5_value(read_u8(q_off + 5u), read_u8(qh_base + 5u), bit, is_hi),
                q5_value(read_u8(q_off + 6u), read_u8(qh_base + 6u), bit, is_hi),
                q5_value(read_u8(q_off + 7u), read_u8(qh_base + 7u), bit, is_hi));
            let q2 = vec4<f32>(
                q5_value(read_u8(q_off + 8u), read_u8(qh_base + 8u), bit, is_hi),
                q5_value(read_u8(q_off + 9u), read_u8(qh_base + 9u), bit, is_hi),
                q5_value(read_u8(q_off + 10u), read_u8(qh_base + 10u), bit, is_hi),
                q5_value(read_u8(q_off + 11u), read_u8(qh_base + 11u), bit, is_hi));
            let q3 = vec4<f32>(
                q5_value(read_u8(q_off + 12u), read_u8(qh_base + 12u), bit, is_hi),
                q5_value(read_u8(q_off + 13u), read_u8(qh_base + 13u), bit, is_hi),
                q5_value(read_u8(q_off + 14u), read_u8(qh_base + 14u), bit, is_hi),
                q5_value(read_u8(q_off + 15u), read_u8(qh_base + 15u), bit, is_hi));
            let q4 = vec4<f32>(
                q5_value(read_u8(q_off + 16u), read_u8(qh_base + 16u), bit, is_hi),
                q5_value(read_u8(q_off + 17u), read_u8(qh_base + 17u), bit, is_hi),
                q5_value(read_u8(q_off + 18u), read_u8(qh_base + 18u), bit, is_hi),
                q5_value(read_u8(q_off + 19u), read_u8(qh_base + 19u), bit, is_hi));
            let q5 = vec4<f32>(
                q5_value(read_u8(q_off + 20u), read_u8(qh_base + 20u), bit, is_hi),
                q5_value(read_u8(q_off + 21u), read_u8(qh_base + 21u), bit, is_hi),
                q5_value(read_u8(q_off + 22u), read_u8(qh_base + 22u), bit, is_hi),
                q5_value(read_u8(q_off + 23u), read_u8(qh_base + 23u), bit, is_hi));
            let q6 = vec4<f32>(
                q5_value(read_u8(q_off + 24u), read_u8(qh_base + 24u), bit, is_hi),
                q5_value(read_u8(q_off + 25u), read_u8(qh_base + 25u), bit, is_hi),
                q5_value(read_u8(q_off + 26u), read_u8(qh_base + 26u), bit, is_hi),
                q5_value(read_u8(q_off + 27u), read_u8(qh_base + 27u), bit, is_hi));
            let q7 = vec4<f32>(
                q5_value(read_u8(q_off + 28u), read_u8(qh_base + 28u), bit, is_hi),
                q5_value(read_u8(q_off + 29u), read_u8(qh_base + 29u), bit, is_hi),
                q5_value(read_u8(q_off + 30u), read_u8(qh_base + 30u), bit, is_hi),
                q5_value(read_u8(q_off + 31u), read_u8(qh_base + 31u), bit, is_hi));

            let x0 = vec4<f32>(x[xb + 0u], x[xb + 1u], x[xb + 2u], x[xb + 3u]);
            let x1 = vec4<f32>(x[xb + 4u], x[xb + 5u], x[xb + 6u], x[xb + 7u]);
            let x2 = vec4<f32>(x[xb + 8u], x[xb + 9u], x[xb + 10u], x[xb + 11u]);
            let x3 = vec4<f32>(x[xb + 12u], x[xb + 13u], x[xb + 14u], x[xb + 15u]);
            let x4 = vec4<f32>(x[xb + 16u], x[xb + 17u], x[xb + 18u], x[xb + 19u]);
            let x5 = vec4<f32>(x[xb + 20u], x[xb + 21u], x[xb + 22u], x[xb + 23u]);
            let x6 = vec4<f32>(x[xb + 24u], x[xb + 25u], x[xb + 26u], x[xb + 27u]);
            let x7 = vec4<f32>(x[xb + 28u], x[xb + 29u], x[xb + 30u], x[xb + 31u]);

            let sx0 = x0.x + x0.y + x0.z + x0.w;
            let sx1 = x1.x + x1.y + x1.z + x1.w;
            let sx2 = x2.x + x2.y + x2.z + x2.w;
            let sx3 = x3.x + x3.y + x3.z + x3.w;
            let sx4 = x4.x + x4.y + x4.z + x4.w;
            let sx5 = x5.x + x5.y + x5.z + x5.w;
            let sx6 = x6.x + x6.y + x6.z + x6.w;
            let sx7 = x7.x + x7.y + x7.z + x7.w;

            acc += ds * (dot(q0, x0) + dot(q1, x1) + dot(q2, x2) + dot(q3, x3) +
                         dot(q4, x4) + dot(q5, x5) + dot(q6, x6) + dot(q7, x7))
                 - dm * (sx0 + sx1 + sx2 + sx3 + sx4 + sx5 + sx6 + sx7);
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
        if (u.split2 > 0u && global_row >= u.split2) {
            out2[u.off2 + global_row - u.split2] = reduce_buf[row_base];
        } else if (global_row >= u.split1) {
            out1[u.off1 + global_row - u.split1] = reduce_buf[row_base];
        } else {
            out0[u.off0 + global_row] = reduce_buf[row_base];
        }
    }
}
