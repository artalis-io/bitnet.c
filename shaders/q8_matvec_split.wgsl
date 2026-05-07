// Q8_0 split matvec for stacked projection buffers.

const TILE_ROWS: u32 = 32u;
const THREADS_PER_ROW: u32 = 8u;
const ELEMS_PER_THREAD: u32 = 32u / THREADS_PER_ROW;

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

fn read_byte(byte_addr: u32) -> u32 {
    let word_idx = byte_addr / 4u;
    let byte_off = byte_addr % 4u;
    return (weights[word_idx] >> (byte_off * 8u)) & 0xFFu;
}

fn read_fp16(byte_addr: u32) -> f32 {
    let lo = read_byte(byte_addr);
    let hi = read_byte(byte_addr + 1u);
    return fp16_to_f32(lo | (hi << 8u));
}

fn sign_extend_i8(val: u32) -> i32 {
    return i32(val << 24u) >> 24;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let tile_start = wid.x * TILE_ROWS;
    let tid = lid.x;
    let local_row = tid / THREADS_PER_ROW;
    let local_elem = tid % THREADS_PER_ROW;
    let global_row = tile_start + local_row;
    let blocks_per_row = u.cols / 32u;
    let row_byte_base = global_row * blocks_per_row * 34u;

    var acc: f32 = 0.0;
    if (global_row < u.rows) {
        for (var b = 0u; b < blocks_per_row; b++) {
            let block_byte = row_byte_base + b * 34u;
            let scale = read_fp16(block_byte);
            let qs_byte_start = block_byte + 2u;
            let elem_base = b * 32u;
            let my_start = local_elem * ELEMS_PER_THREAD;
            for (var i = 0u; i < ELEMS_PER_THREAD; i++) {
                let elem = my_start + i;
                let byte_val = read_byte(qs_byte_start + elem);
                acc += scale * f32(sign_extend_i8(byte_val)) *
                       x[elem_base + elem];
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
        if (u.split2 > 0u && global_row >= u.split2) {
            out2[u.off2 + global_row - u.split2] = reduce_buf[row_base];
        } else if (global_row >= u.split1) {
            out1[u.off1 + global_row - u.split1] = reduce_buf[row_base];
        } else {
            out0[u.off0 + global_row] = reduce_buf[row_base];
        }
    }
}
