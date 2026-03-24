// Q8_0 matvec/matmul — 8-bit quantization
// 32-element blocks, 34 bytes each: 2-byte FP16 scale + 32 int8 values

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

var<workgroup> shared_data: array<f32, 256>;

fn fp16_to_f32(bits: u32) -> f32 {
    let sign = (bits >> 15u) & 1u;
    let exp = (bits >> 10u) & 0x1Fu;
    let mant = bits & 0x3FFu;
    if (exp == 0u && mant == 0u) {
        return select(0.0, -0.0, sign == 1u);
    }
    let f_exp = f32(i32(exp) - 15 + 127);
    let f_bits = (sign << 31u) | (u32(f_exp) << 23u) | (mant << 13u);
    return bitcast<f32>(f_bits);
}

fn workgroup_reduce(lid: u32, val: f32) -> f32 {
    shared_data[lid] = val;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid < s) {
            shared_data[lid] += shared_data[lid + s];
        }
        workgroupBarrier();
    }
    return shared_data[0];
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
    // Sign-extend from 8-bit to 32-bit
    return i32(val << 24u) >> 24;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x;
    let token = wid.y;
    let tid = lid.x;

    if (row >= uniforms.rows) {
        return;
    }

    let cols = uniforms.cols;
    let blocks_per_row = cols / 32u;
    // Each block is 34 bytes
    let row_byte_offset = row * blocks_per_row * 34u;
    let x_offset = token * cols;

    var sum = 0.0f;

    var block_idx = tid;
    while (block_idx < blocks_per_row) {
        let block_byte = row_byte_offset + block_idx * 34u;

        // Read FP16 scale from first 2 bytes
        let scale = read_fp16(block_byte);

        // Quantized int8 data starts at block_byte + 2
        let qs_byte_start = block_byte + 2u;
        let elem_offset = block_idx * 32u;

        // Process 32 int8 values, 4 at a time via u32 reads
        for (var w = 0u; w < 8u; w++) {
            let word_byte_addr = qs_byte_start + w * 4u;
            let word_idx = word_byte_addr / 4u;
            let word_off = word_byte_addr % 4u;

            // Read 4 bytes (may span two u32s if not aligned)
            var word: u32;
            if (word_off == 0u) {
                word = weights[word_idx];
            } else {
                word = (weights[word_idx] >> (word_off * 8u)) | (weights[word_idx + 1u] << ((4u - word_off) * 8u));
            }

            let i0 = sign_extend_i8(word & 0xFFu);
            let i1 = sign_extend_i8((word >> 8u) & 0xFFu);
            let i2 = sign_extend_i8((word >> 16u) & 0xFFu);
            let i3 = sign_extend_i8((word >> 24u) & 0xFFu);

            let base = elem_offset + w * 4u;
            sum += scale * f32(i0) * x[x_offset + base];
            sum += scale * f32(i1) * x[x_offset + base + 1u];
            sum += scale * f32(i2) * x[x_offset + base + 2u];
            sum += scale * f32(i3) * x[x_offset + base + 3u];
        }

        block_idx += 256u;
    }

    let result = workgroup_reduce(tid, sum);

    if (tid == 0u) {
        out[token * uniforms.rows + row] = result;
    }
}
