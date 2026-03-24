// Q4_1 matvec/matmul — 4-bit quantization with min offset
// 32-element blocks, 20 bytes each: 2-byte FP16 scale + 2-byte FP16 min + 16 bytes packed nibbles

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

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = select(wid.x, wid.x + wid.y * uniforms.extra, uniforms.extra > 0u);
    let token = select(wid.y, 0u, uniforms.extra > 0u);
    let tid = lid.x;

    if (row >= uniforms.rows) {
        return;
    }

    let cols = uniforms.cols;
    let blocks_per_row = cols / 32u;
    // Each block is 20 bytes
    let row_byte_offset = row * blocks_per_row * 20u;
    let x_offset = token * cols;

    var sum = 0.0f;

    var block_idx = tid;
    while (block_idx < blocks_per_row) {
        let block_byte = row_byte_offset + block_idx * 20u;

        // Read FP16 scale (bytes 0-1) and FP16 min (bytes 2-3)
        let d = read_fp16(block_byte);
        let m = read_fp16(block_byte + 2u);

        // Quantized data starts at block_byte + 4
        let qs_byte_start = block_byte + 4u;
        let elem_offset = block_idx * 32u;

        // 16 bytes of packed nibbles → 32 elements
        // Layout: low nibble of qs[i] → element i (0..15)
        //         high nibble of qs[i] → element i+16 (16..31)
        for (var i = 0u; i < 16u; i++) {
            let byte_val = read_byte(qs_byte_start + i);

            let lo = d * f32(byte_val & 0xFu) + m;
            let hi = d * f32((byte_val >> 4u) & 0xFu) + m;

            sum += lo * x[x_offset + elem_offset + i];
            sum += hi * x[x_offset + elem_offset + i + 16u];
        }

        block_idx += 256u;
    }

    let result = workgroup_reduce(tid, sum);

    if (tid == 0u) {
        out[token * uniforms.rows + row] = result;
    }
}
