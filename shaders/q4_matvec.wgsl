// Q4_0 matvec/matmul — 4-bit quantization
// 32-element blocks, 18 bytes each: 2-byte FP16 scale + 16 bytes packed nibbles

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

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = gid.x;
    let token = gid.y;
    let tid = lid.x;

    if (row >= uniforms.rows) {
        return;
    }

    let cols = uniforms.cols;
    let blocks_per_row = cols / 32u;
    // Each block is 18 bytes. Row byte offset:
    let row_byte_offset = row * blocks_per_row * 18u;
    let x_offset = token * cols;

    var sum = 0.0f;

    var block_idx = tid;
    while (block_idx < blocks_per_row) {
        // Block byte offset within weight data
        let block_byte = row_byte_offset + block_idx * 18u;

        // Read FP16 scale from first 2 bytes
        let scale_u32_idx = block_byte / 4u;
        let scale_byte_off = block_byte % 4u;
        var scale_bits: u32;
        if (scale_byte_off <= 2u) {
            scale_bits = (weights[scale_u32_idx] >> (scale_byte_off * 8u)) & 0xFFFFu;
        } else {
            // Scale spans two u32s
            scale_bits = (weights[scale_u32_idx] >> 24u) | ((weights[scale_u32_idx + 1u] & 0xFFu) << 8u);
        }
        let scale = fp16_to_f32(scale_bits);

        // Quantized data starts at block_byte + 2
        let qs_byte_start = block_byte + 2u;
        let elem_offset = block_idx * 32u;

        // 16 bytes of packed nibbles → 32 elements
        for (var i = 0u; i < 16u; i++) {
            let byte_addr = qs_byte_start + i;
            let word_idx = byte_addr / 4u;
            let byte_off = byte_addr % 4u;
            let byte_val = (weights[word_idx] >> (byte_off * 8u)) & 0xFFu;

            let lo = f32(i32(byte_val & 0xFu) - 8);
            let hi = f32(i32((byte_val >> 4u) & 0xFu) - 8);

            sum += scale * lo * x[x_offset + elem_offset + i * 2u];
            sum += scale * hi * x[x_offset + elem_offset + i * 2u + 1u];
        }

        block_idx += 256u;
    }

    let result = workgroup_reduce(tid, sum);

    if (tid == 0u) {
        out[token * uniforms.rows + row] = result;
    }
}
