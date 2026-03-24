// Q3_K matvec — 3-bit k-quant, 256 elements per block, 110 bytes/block
// Layout: hmask[32] (bytes 0-31), qs[64] (bytes 32-95), scales[12] (bytes 96-107), d FP16 (bytes 108-109)

struct Uniforms { rows: u32, cols: u32, n_tokens: u32, extra: u32 }

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> u: Uniforms;

fn fp16_to_f32(bits: u32) -> f32 {
    let sign = (bits >> 15u) & 1u;
    let exp = (bits >> 10u) & 0x1Fu;
    let mant = bits & 0x3FFu;
    if (exp == 0u && mant == 0u) { return select(0.0, -0.0, sign == 1u); }
    let f_exp = f32(i32(exp) - 15 + 127);
    let f_bits = (sign << 31u) | (u32(f_exp) << 23u) | (mant << 13u);
    return bitcast<f32>(f_bits);
}

var<workgroup> shared_data: array<f32, 256>;

fn workgroup_reduce(lid: u32, val: f32) -> f32 {
    shared_data[lid] = val;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid < s) { shared_data[lid] += shared_data[lid + s]; }
        workgroupBarrier();
    }
    return shared_data[0];
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

fn read_u32(offset: u32) -> u32 {
    // Assumes 4-byte aligned offset
    return weights[offset >> 2u];
}

const BLOCK_BYTES: u32 = 110u;
const QK_K: u32 = 256u;

// Unpack 12 bytes of packed scales into 16 6-bit values.
// Mirrors bn_q3k_unpack_scales from quant_internal.h:
//   aux[2] = ((aux[0] >> 4) & 0x0f0f0f0f) | (((tmp >> 4) & 0x03030303) << 4)
//   aux[3] = ((aux[1] >> 4) & 0x0f0f0f0f) | (((tmp >> 6) & 0x03030303) << 4)
//   aux[0] = (aux[0] & 0x0f0f0f0f)         | (((tmp >> 0) & 0x03030303) << 4)
//   aux[1] = (aux[1] & 0x0f0f0f0f)         | (((tmp >> 2) & 0x03030303) << 4)
// Result is 16 bytes (aux[0..3]), each byte is a 6-bit scale.
fn unpack_q3k_scale(scales_base: u32, idx: u32) -> u32 {
    // Read 12 bytes as 3 u32s (byte-aligned reads)
    let b0 = read_u8(scales_base + 0u);
    let b1 = read_u8(scales_base + 1u);
    let b2 = read_u8(scales_base + 2u);
    let b3 = read_u8(scales_base + 3u);
    let aux0 = b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);

    let b4 = read_u8(scales_base + 4u);
    let b5 = read_u8(scales_base + 5u);
    let b6 = read_u8(scales_base + 6u);
    let b7 = read_u8(scales_base + 7u);
    let aux1 = b4 | (b5 << 8u) | (b6 << 16u) | (b7 << 24u);

    let b8 = read_u8(scales_base + 8u);
    let b9 = read_u8(scales_base + 9u);
    let b10 = read_u8(scales_base + 10u);
    let b11 = read_u8(scales_base + 11u);
    let tmp = b8 | (b9 << 8u) | (b10 << 16u) | (b11 << 24u);

    var r: array<u32, 4>;
    r[2] = ((aux0 >> 4u) & 0x0F0F0F0Fu) | (((tmp >> 4u) & 0x03030303u) << 4u);
    r[3] = ((aux1 >> 4u) & 0x0F0F0F0Fu) | (((tmp >> 6u) & 0x03030303u) << 4u);
    r[0] = (aux0 & 0x0F0F0F0Fu)          | (((tmp >> 0u) & 0x03030303u) << 4u);
    r[1] = (aux1 & 0x0F0F0F0Fu)          | (((tmp >> 2u) & 0x03030303u) << 4u);

    // Extract byte at position idx from the 16-byte result
    let word_idx = idx / 4u;
    let byte_idx = idx % 4u;
    return (r[word_idx] >> (byte_idx * 8u)) & 0x3Fu;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = select(wid.x, wid.x + wid.y * u.extra, u.extra > 0u);
    let token = select(wid.y, 0u, u.extra > 0u);
    let tid = lid.x;

    if (row >= u.rows) { return; }

    let cols = u.cols;
    let n_blocks = cols / QK_K;
    let row_byte = row * n_blocks * BLOCK_BYTES;
    let x_off = token * cols;

    var sum = 0.0f;

    var bi = tid;
    while (bi < n_blocks) {
        let base = row_byte + bi * BLOCK_BYTES;

        let hmask_base  = base;         // bytes 0-31
        let qs_base     = base + 32u;   // bytes 32-95
        let scales_base = base + 96u;   // bytes 96-107
        let d = fp16_to_f32(read_u16(base + 108u));

        let elem_off = bi * QK_K;

        var is_idx = 0u;
        var m = 1u; // hmask bit position
        var out_idx = 0u;

        for (var n = 0u; n < QK_K; n += 128u) {
            var shift = 0u;
            let q_start = qs_base + n / 4u;
            for (var j = 0u; j < 4u; j++) {
                // First sub-block of 16
                let sc0 = unpack_q3k_scale(scales_base, is_idx);
                is_idx++;
                let dl0 = d * f32(i32(sc0) - 32);
                for (var l = 0u; l < 16u; l++) {
                    let low2 = (read_u8(q_start + l) >> shift) & 3u;
                    let hbit = read_u8(hmask_base + l) & m;
                    var q3 = i32(low2);
                    if (hbit == 0u) { q3 -= 4; }
                    sum += dl0 * f32(q3) * x[x_off + elem_off + out_idx];
                    out_idx++;
                }

                // Second sub-block of 16
                let sc1 = unpack_q3k_scale(scales_base, is_idx);
                is_idx++;
                let dl1 = d * f32(i32(sc1) - 32);
                for (var l = 0u; l < 16u; l++) {
                    let low2 = (read_u8(q_start + l + 16u) >> shift) & 3u;
                    let hbit = read_u8(hmask_base + l + 16u) & m;
                    var q3 = i32(low2);
                    if (hbit == 0u) { q3 -= 4; }
                    sum += dl1 * f32(q3) * x[x_off + elem_off + out_idx];
                    out_idx++;
                }

                shift += 2u;
                m <<= 1u;
            }
        }

        bi += 256u;
    }

    let result = workgroup_reduce(tid, sum);
    if (tid == 0u) {
        out[token * u.rows + row] = result;
    }
}
