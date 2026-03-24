// TQ1_0 matvec — base-3 ternary packing
// 256-element blocks, 54 bytes each: qs[48] + qh[4] + d(FP16)
// qs[0..31]:  5 trits per byte x 32 bytes = 160 values
// qs[32..47]: 5 trits per byte x 16 bytes = 80 values
// qh[0..3]:   4 trits per byte x 4 bytes  = 16 values
// Total: 160 + 80 + 16 = 256 values

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

fn read_u8(base: u32, offset: u32) -> u32 {
    let addr = base + offset;
    let word = weights[addr >> 2u];
    return (word >> ((addr & 3u) * 8u)) & 0xFFu;
}

fn read_i8(base: u32, offset: u32) -> i32 {
    let v = read_u8(base, offset);
    return select(i32(v), i32(v) - 256, v >= 128u);
}

fn read_u16(base: u32, offset: u32) -> u32 {
    let addr = base + offset;
    let word = weights[addr >> 2u];
    let shift = (addr & 2u) * 8u;
    return (word >> shift) & 0xFFFFu;
}

fn read_u32_raw(base: u32, offset: u32) -> u32 {
    return weights[(base + offset) >> 2u];
}

// Decode one trit from a byte: returns -1, 0, or 1
// C code: uint8_t q = qs[m] * pow3[n]; (truncates to 8 bits)
//         int16_t xi = ((uint16_t)q * 3) >> 8;
fn decode_trit(byte_val: u32, pow3_val: u32) -> i32 {
    let q = (byte_val * pow3_val) & 0xFFu;  // truncate to uint8
    let xi = (q * 3u) >> 8u;
    return i32(xi) - 1;
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
    let block_bytes = 54u;
    let row_byte_offset = row * blocks_per_row * block_bytes;
    let x_offset = token * cols;

    // Powers of 3 for trit extraction
    let pow3 = array<u32, 5>(1u, 3u, 9u, 27u, 81u);

    var sum = 0.0f;

    var block_idx = tid;
    while (block_idx < blocks_per_row) {
        let block_byte = row_byte_offset + block_idx * block_bytes;
        let elem_base = block_idx * 256u;

        // Read FP16 scale at offset 52 (after qs[48] + qh[4])
        let d = fp16_to_f32(read_u16(block_byte, 52u));

        var block_sum = 0.0f;

        // Section 1: qs[0..31] -> 160 values (5 trits x 32 bytes)
        for (var n = 0u; n < 5u; n++) {
            let p3 = pow3[n];
            for (var m = 0u; m < 32u; m++) {
                let byte_val = read_u8(block_byte, m);
                let w = decode_trit(byte_val, p3);
                block_sum += f32(w) * x[x_offset + elem_base + n * 32u + m];
            }
        }

        // Section 2: qs[32..47] -> 80 values (5 trits x 16 bytes)
        for (var n = 0u; n < 5u; n++) {
            let p3 = pow3[n];
            for (var m = 0u; m < 16u; m++) {
                let byte_val = read_u8(block_byte, 32u + m);
                let w = decode_trit(byte_val, p3);
                block_sum += f32(w) * x[x_offset + elem_base + 160u + n * 16u + m];
            }
        }

        // Section 3: qh[0..3] -> 16 values (4 trits x 4 bytes)
        for (var n = 0u; n < 4u; n++) {
            let p3 = pow3[n];
            for (var m = 0u; m < 4u; m++) {
                let byte_val = read_u8(block_byte, 48u + m);
                let w = decode_trit(byte_val, p3);
                block_sum += f32(w) * x[x_offset + elem_base + 240u + n * 4u + m];
            }
        }

        sum += block_sum * d;
        block_idx += 256u;
    }

    let result = workgroup_reduce(tid, sum);

    if (tid == 0u) {
        out[token * uniforms.rows + row] = result;
    }
}
