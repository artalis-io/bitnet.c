// Q4_K two-output matvec split for stacked gate+up buffers.

const TILE_ROWS: u32 = 32u;
const THREADS_PER_ROW: u32 = 8u;
const ELEMS_PER_THREAD: u32 = 256u / THREADS_PER_ROW;
const QK_K: u32 = 256u;
const BLOCK_BYTES: u32 = 144u;

struct Uniforms {
    rows: u32,
    cols: u32,
    split: u32,
    _pad3: u32,
    _pad4: u32,
    _pad5: u32,
    _pad6: u32,
    _pad7: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out0: array<f32>;
@group(0) @binding(3) var<storage, read_write> out1: array<f32>;
@group(0) @binding(4) var<uniform> u: Uniforms;

var<workgroup> reduce_buf: array<f32, 256>;
var<workgroup> reduce_buf2: array<f32, 256>;

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

fn q4_value(qbyte: u32, is_high: bool) -> f32 {
    return select(f32(qbyte >> 4u), f32(qbyte & 0xFu), !is_high);
}

fn byte_at(word: u32, idx: u32) -> u32 {
    return (word >> (idx * 8u)) & 0xFFu;
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

    var acc: f32 = 0.0;
    var acc2: f32 = 0.0;

    if (global_row < u.rows) {
        for (var bi = 0u; bi < n_blocks; bi++) {
            let base = row_byte + bi * BLOCK_BYTES;
            let d    = fp16_to_f32(read_u16(base));
            let dmin = fp16_to_f32(read_u16(base + 2u));
            let scales_base = base + 4u;
            let qs_base = base + 16u;
            let elem_base = bi * QK_K;

            let group = my_start / 64u;
            let is_high = (my_start % 64u) / 32u;
            let sub = group * 2u + is_high;
            let sm = get_scale_min(sub, scales_base);
            let q_off_base = qs_base + group * 32u;
            let is_hi = is_high != 0u;
            let xb = elem_base + my_start;
            let qw0 = weights[(q_off_base + 0u) >> 2u];
            let qw1 = weights[(q_off_base + 4u) >> 2u];
            let qw2 = weights[(q_off_base + 8u) >> 2u];
            let qw3 = weights[(q_off_base + 12u) >> 2u];
            let qw4 = weights[(q_off_base + 16u) >> 2u];
            let qw5 = weights[(q_off_base + 20u) >> 2u];
            let qw6 = weights[(q_off_base + 24u) >> 2u];
            let qw7 = weights[(q_off_base + 28u) >> 2u];

            let q0 = vec4<f32>(
                q4_value(byte_at(qw0, 0u), is_hi),
                q4_value(byte_at(qw0, 1u), is_hi),
                q4_value(byte_at(qw0, 2u), is_hi),
                q4_value(byte_at(qw0, 3u), is_hi));
            let q1 = vec4<f32>(
                q4_value(byte_at(qw1, 0u), is_hi),
                q4_value(byte_at(qw1, 1u), is_hi),
                q4_value(byte_at(qw1, 2u), is_hi),
                q4_value(byte_at(qw1, 3u), is_hi));
            let q2 = vec4<f32>(
                q4_value(byte_at(qw2, 0u), is_hi),
                q4_value(byte_at(qw2, 1u), is_hi),
                q4_value(byte_at(qw2, 2u), is_hi),
                q4_value(byte_at(qw2, 3u), is_hi));
            let q3 = vec4<f32>(
                q4_value(byte_at(qw3, 0u), is_hi),
                q4_value(byte_at(qw3, 1u), is_hi),
                q4_value(byte_at(qw3, 2u), is_hi),
                q4_value(byte_at(qw3, 3u), is_hi));
            let q4 = vec4<f32>(
                q4_value(byte_at(qw4, 0u), is_hi),
                q4_value(byte_at(qw4, 1u), is_hi),
                q4_value(byte_at(qw4, 2u), is_hi),
                q4_value(byte_at(qw4, 3u), is_hi));
            let q5 = vec4<f32>(
                q4_value(byte_at(qw5, 0u), is_hi),
                q4_value(byte_at(qw5, 1u), is_hi),
                q4_value(byte_at(qw5, 2u), is_hi),
                q4_value(byte_at(qw5, 3u), is_hi));
            let q6 = vec4<f32>(
                q4_value(byte_at(qw6, 0u), is_hi),
                q4_value(byte_at(qw6, 1u), is_hi),
                q4_value(byte_at(qw6, 2u), is_hi),
                q4_value(byte_at(qw6, 3u), is_hi));
            let q7 = vec4<f32>(
                q4_value(byte_at(qw7, 0u), is_hi),
                q4_value(byte_at(qw7, 1u), is_hi),
                q4_value(byte_at(qw7, 2u), is_hi),
                q4_value(byte_at(qw7, 3u), is_hi));

            let x0 = vec4<f32>(x[xb + 0u], x[xb + 1u], x[xb + 2u], x[xb + 3u]);
            let x1 = vec4<f32>(x[xb + 4u], x[xb + 5u], x[xb + 6u], x[xb + 7u]);
            let x2 = vec4<f32>(x[xb + 8u], x[xb + 9u], x[xb + 10u], x[xb + 11u]);
            let x3 = vec4<f32>(x[xb + 12u], x[xb + 13u], x[xb + 14u], x[xb + 15u]);
            let x4 = vec4<f32>(x[xb + 16u], x[xb + 17u], x[xb + 18u], x[xb + 19u]);
            let x5 = vec4<f32>(x[xb + 20u], x[xb + 21u], x[xb + 22u], x[xb + 23u]);
            let x6 = vec4<f32>(x[xb + 24u], x[xb + 25u], x[xb + 26u], x[xb + 27u]);
            let x7 = vec4<f32>(x[xb + 28u], x[xb + 29u], x[xb + 30u], x[xb + 31u]);

            let sum_x = x0.x + x0.y + x0.z + x0.w +
                        x1.x + x1.y + x1.z + x1.w +
                        x2.x + x2.y + x2.z + x2.w +
                        x3.x + x3.y + x3.z + x3.w +
                        x4.x + x4.y + x4.z + x4.w +
                        x5.x + x5.y + x5.z + x5.w +
                        x6.x + x6.y + x6.z + x6.w +
                        x7.x + x7.y + x7.z + x7.w;
            let sum_qx = dot(q0, x0) + dot(q1, x1) + dot(q2, x2) + dot(q3, x3) +
                         dot(q4, x4) + dot(q5, x5) + dot(q6, x6) + dot(q7, x7);
            acc += d * f32(sm.x) * sum_qx - dmin * f32(sm.y) * sum_x;
        }
    }

    if (u._pad3 != 0u && global_row < u.split) {
        let up_row = global_row + u.split;
        let up_row_byte = up_row * n_blocks * BLOCK_BYTES;
        for (var bi = 0u; bi < n_blocks; bi++) {
            let base = up_row_byte + bi * BLOCK_BYTES;
            let d    = fp16_to_f32(read_u16(base));
            let dmin = fp16_to_f32(read_u16(base + 2u));
            let scales_base = base + 4u;
            let qs_base = base + 16u;
            let elem_base = bi * QK_K;

            let group = my_start / 64u;
            let is_high = (my_start % 64u) / 32u;
            let sub = group * 2u + is_high;
            let sm = get_scale_min(sub, scales_base);
            let q_off_base = qs_base + group * 32u;
            let is_hi = is_high != 0u;
            let xb = elem_base + my_start;
            let qw0 = weights[(q_off_base + 0u) >> 2u];
            let qw1 = weights[(q_off_base + 4u) >> 2u];
            let qw2 = weights[(q_off_base + 8u) >> 2u];
            let qw3 = weights[(q_off_base + 12u) >> 2u];
            let qw4 = weights[(q_off_base + 16u) >> 2u];
            let qw5 = weights[(q_off_base + 20u) >> 2u];
            let qw6 = weights[(q_off_base + 24u) >> 2u];
            let qw7 = weights[(q_off_base + 28u) >> 2u];

            let q0 = vec4<f32>(
                q4_value(byte_at(qw0, 0u), is_hi),
                q4_value(byte_at(qw0, 1u), is_hi),
                q4_value(byte_at(qw0, 2u), is_hi),
                q4_value(byte_at(qw0, 3u), is_hi));
            let q1 = vec4<f32>(
                q4_value(byte_at(qw1, 0u), is_hi),
                q4_value(byte_at(qw1, 1u), is_hi),
                q4_value(byte_at(qw1, 2u), is_hi),
                q4_value(byte_at(qw1, 3u), is_hi));
            let q2 = vec4<f32>(
                q4_value(byte_at(qw2, 0u), is_hi),
                q4_value(byte_at(qw2, 1u), is_hi),
                q4_value(byte_at(qw2, 2u), is_hi),
                q4_value(byte_at(qw2, 3u), is_hi));
            let q3 = vec4<f32>(
                q4_value(byte_at(qw3, 0u), is_hi),
                q4_value(byte_at(qw3, 1u), is_hi),
                q4_value(byte_at(qw3, 2u), is_hi),
                q4_value(byte_at(qw3, 3u), is_hi));
            let q4 = vec4<f32>(
                q4_value(byte_at(qw4, 0u), is_hi),
                q4_value(byte_at(qw4, 1u), is_hi),
                q4_value(byte_at(qw4, 2u), is_hi),
                q4_value(byte_at(qw4, 3u), is_hi));
            let q5 = vec4<f32>(
                q4_value(byte_at(qw5, 0u), is_hi),
                q4_value(byte_at(qw5, 1u), is_hi),
                q4_value(byte_at(qw5, 2u), is_hi),
                q4_value(byte_at(qw5, 3u), is_hi));
            let q6 = vec4<f32>(
                q4_value(byte_at(qw6, 0u), is_hi),
                q4_value(byte_at(qw6, 1u), is_hi),
                q4_value(byte_at(qw6, 2u), is_hi),
                q4_value(byte_at(qw6, 3u), is_hi));
            let q7 = vec4<f32>(
                q4_value(byte_at(qw7, 0u), is_hi),
                q4_value(byte_at(qw7, 1u), is_hi),
                q4_value(byte_at(qw7, 2u), is_hi),
                q4_value(byte_at(qw7, 3u), is_hi));

            let x0 = vec4<f32>(x[xb + 0u], x[xb + 1u], x[xb + 2u], x[xb + 3u]);
            let x1 = vec4<f32>(x[xb + 4u], x[xb + 5u], x[xb + 6u], x[xb + 7u]);
            let x2 = vec4<f32>(x[xb + 8u], x[xb + 9u], x[xb + 10u], x[xb + 11u]);
            let x3 = vec4<f32>(x[xb + 12u], x[xb + 13u], x[xb + 14u], x[xb + 15u]);
            let x4 = vec4<f32>(x[xb + 16u], x[xb + 17u], x[xb + 18u], x[xb + 19u]);
            let x5 = vec4<f32>(x[xb + 20u], x[xb + 21u], x[xb + 22u], x[xb + 23u]);
            let x6 = vec4<f32>(x[xb + 24u], x[xb + 25u], x[xb + 26u], x[xb + 27u]);
            let x7 = vec4<f32>(x[xb + 28u], x[xb + 29u], x[xb + 30u], x[xb + 31u]);

            let sum_x = x0.x + x0.y + x0.z + x0.w +
                        x1.x + x1.y + x1.z + x1.w +
                        x2.x + x2.y + x2.z + x2.w +
                        x3.x + x3.y + x3.z + x3.w +
                        x4.x + x4.y + x4.z + x4.w +
                        x5.x + x5.y + x5.z + x5.w +
                        x6.x + x6.y + x6.z + x6.w +
                        x7.x + x7.y + x7.z + x7.w;
            let sum_qx = dot(q0, x0) + dot(q1, x1) + dot(q2, x2) + dot(q3, x3) +
                         dot(q4, x4) + dot(q5, x5) + dot(q6, x6) + dot(q7, x7);
            acc2 += d * f32(sm.x) * sum_qx - dmin * f32(sm.y) * sum_x;
        }
    }

    reduce_buf[tid] = acc;
    reduce_buf2[tid] = acc2;
    workgroupBarrier();

    let row_base = local_row * THREADS_PER_ROW;
    if (local_elem < 4u) {
        reduce_buf[row_base + local_elem] += reduce_buf[row_base + local_elem + 4u];
        reduce_buf2[row_base + local_elem] += reduce_buf2[row_base + local_elem + 4u];
    }
    workgroupBarrier();
    if (local_elem < 2u) {
        reduce_buf[row_base + local_elem] += reduce_buf[row_base + local_elem + 2u];
        reduce_buf2[row_base + local_elem] += reduce_buf2[row_base + local_elem + 2u];
    }
    workgroupBarrier();
    if (local_elem < 1u) {
        reduce_buf[row_base + local_elem] += reduce_buf[row_base + local_elem + 1u];
        reduce_buf2[row_base + local_elem] += reduce_buf2[row_base + local_elem + 1u];
    }
    workgroupBarrier();

    if (local_elem == 0u && global_row < u.rows) {
        if (u._pad3 != 0u) {
            if (global_row < u.split) {
                let gate = reduce_buf[row_base];
                out0[global_row] = (gate / (1.0 + exp(-gate))) * reduce_buf2[row_base];
            }
        } else if (global_row >= u.split) {
            out1[global_row - u.split] = reduce_buf[row_base];
        } else {
            out0[global_row] = reduce_buf[row_base];
        }
    }
}
