// TQ2_0 matvec/matmul — 2-bit ternary quantization (256-element blocks)
// 66 bytes per block: 2-byte FP16 scale + 64 bytes of 2-bit ternary values
// Two halves of 128 elements each (32 bytes per half)
// Decode: each byte has 4 values: (byte & 3) - 1, ((byte >> 2) & 3) - 1, etc.

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
    let row = wid.x;
    let token = wid.y;
    let tid = lid.x;

    if (row >= uniforms.rows) {
        return;
    }

    let cols = uniforms.cols;
    let blocks_per_row = cols / 256u;
    // Each block is 66 bytes
    let row_byte_offset = row * blocks_per_row * 66u;
    let x_offset = token * cols;

    var sum = 0.0f;

    var block_idx = tid;
    while (block_idx < blocks_per_row) {
        let block_byte = row_byte_offset + block_idx * 66u;

        // TQ2 layout: qs[64] then d(FP16) — scale is at END of block
        let qs_byte_start = block_byte;
        let scale = read_fp16(block_byte + 64u);
        let elem_offset = block_idx * 256u;

        // Process 64 bytes → 256 elements (4 values per byte)
        for (var i = 0u; i < 64u; i++) {
            let byte_val = read_byte(qs_byte_start + i);
            let base = elem_offset + i * 4u;

            // Decode 4 ternary values from byte
            let v0 = f32(i32(byte_val & 3u) - 1);
            let v1 = f32(i32((byte_val >> 2u) & 3u) - 1);
            let v2 = f32(i32((byte_val >> 4u) & 3u) - 1);
            let v3 = f32(i32((byte_val >> 6u) & 3u) - 1);

            sum += scale * v0 * x[x_offset + base];
            sum += scale * v1 * x[x_offset + base + 1u];
            sum += scale * v2 * x[x_offset + base + 2u];
            sum += scale * v3 * x[x_offset + base + 3u];
        }

        block_idx += 256u;
    }

    let result = workgroup_reduce(tid, sum);

    if (tid == 0u) {
        out[token * uniforms.rows + row] = result;
    }
}
