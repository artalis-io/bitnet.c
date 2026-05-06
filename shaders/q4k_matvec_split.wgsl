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

            var sum_qx: f32 = 0.0;
            var sum_x: f32 = 0.0;
            for (var i = 0u; i < 32u; i++) {
                let xv = x[elem_base + my_start + i];
                let qbyte = read_u8(q_off_base + i);
                let qval = select(f32(qbyte >> 4u), f32(qbyte & 0xFu), is_high == 0u);
                sum_qx += qval * xv;
                sum_x += xv;
            }
            acc += d * f32(sm.x) * sum_qx - dmin * f32(sm.y) * sum_x;
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
        if (global_row >= u.split) {
            out1[global_row - u.split] = reduce_buf[row_base];
        } else {
            out0[global_row] = reduce_buf[row_base];
        }
    }
}
