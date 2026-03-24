// Q5_K matvec — 5-bit k-quant, 256 elements per block, 176 bytes/block
// Layout: d FP16 (bytes 0-1), dmin FP16 (bytes 2-3), scales[12] (bytes 4-15),
//         qh[32] (bytes 16-47), qs[128] (bytes 48-175)

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

const BLOCK_BYTES: u32 = 176u;
const QK_K: u32 = 256u;

// Same scale/min unpacking as Q4_K (shared 12-byte packed format)
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
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = gid.x;
    let token = gid.y;
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

        let d    = fp16_to_f32(read_u16(base));
        let dmin = fp16_to_f32(read_u16(base + 2u));
        let scales_base = base + 4u;
        let qh_base = base + 16u;
        let qs_base = base + 48u;
        let elem_off = bi * QK_K;

        // Process 256 elements in 4 groups of 64
        for (var j = 0u; j < QK_K; j += 64u) {
            let sub = j / 32u;
            let group = j / 64u;     // 0..3
            let bit_lo = group * 2u;  // bits 0,2,4,6
            let bit_hi = group * 2u + 1u; // bits 1,3,5,7

            let q_off = qs_base + j / 2u;

            // Low nibbles + high bit: 32 elements
            let sm0 = get_scale_min(sub, scales_base);
            let ds0 = d * f32(sm0.x);
            let dm0 = dmin * f32(sm0.y);
            for (var l = 0u; l < 32u; l++) {
                let qbyte = read_u8(q_off + l);
                let hbyte = read_u8(qh_base + l);
                let q5 = (qbyte & 0xFu) | (((hbyte >> bit_lo) & 1u) << 4u);
                sum += (ds0 * f32(q5) - dm0) * x[x_off + elem_off + j + l];
            }

            // High nibbles + high bit: 32 elements
            let sm1 = get_scale_min(sub + 1u, scales_base);
            let ds1 = d * f32(sm1.x);
            let dm1 = dmin * f32(sm1.y);
            for (var l = 0u; l < 32u; l++) {
                let qbyte = read_u8(q_off + l);
                let hbyte = read_u8(qh_base + l);
                let q5 = (qbyte >> 4u) | (((hbyte >> bit_hi) & 1u) << 4u);
                sum += (ds1 * f32(q5) - dm1) * x[x_off + elem_off + j + l + 32u];
            }
        }

        bi += 256u;
    }

    let result = workgroup_reduce(tid, sum);
    if (tid == 0u) {
        out[token * u.rows + row] = result;
    }
}
