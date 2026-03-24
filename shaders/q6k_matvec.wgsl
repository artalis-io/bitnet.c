// Q6_K matvec — 6-bit k-quant, 256 elements per block, 210 bytes/block
// Layout: ql[128] (bytes 0-127), qh[64] (bytes 128-191), scales[16] (bytes 192-207), d FP16 (bytes 208-209)

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

fn read_i8(offset: u32) -> i32 {
    let v = read_u8(offset);
    return select(i32(v), i32(v) - 256, v >= 128u);
}

fn read_u16(offset: u32) -> u32 {
    let word = weights[offset >> 2u];
    let shift = (offset & 2u) * 8u;
    return (word >> shift) & 0xFFFFu;
}

const BLOCK_BYTES: u32 = 210u;
const QK_K: u32 = 256u;

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

        let ql_base = base;           // bytes 0-127
        let qh_base = base + 128u;    // bytes 128-191
        let sc_base = base + 192u;    // bytes 192-207
        let d = fp16_to_f32(read_u16(base + 208u));
        let elem_off = bi * QK_K;

        // Process 256 elements in 2 groups of 128
        // Each 128-element group uses 64 ql bytes, 32 qh bytes, 8 scales
        for (var n = 0u; n < QK_K; n += 128u) {
            let half = n / 128u;
            let ql_off = ql_base + half * 64u;
            let qh_off = qh_base + half * 32u;
            let sc_off = sc_base + half * 8u;

            // Within each 128-element group, 32 iterations produce 4 elements each
            for (var l = 0u; l < 32u; l++) {
                let is_idx = l / 16u; // 0 or 1 within each sub-group

                let ql0 = read_u8(ql_off + l);
                let ql1 = read_u8(ql_off + l + 32u);
                let qh_val = read_u8(qh_off + l);

                // 4 elements from each (l, l+32) pair with different bit extractions
                let q1 = i32((ql0 & 0xFu)       | (((qh_val >> 0u) & 3u) << 4u)) - 32;
                let q2 = i32((ql1 & 0xFu)       | (((qh_val >> 2u) & 3u) << 4u)) - 32;
                let q3 = i32((ql0 >> 4u)         | (((qh_val >> 4u) & 3u) << 4u)) - 32;
                let q4 = i32((ql1 >> 4u)         | (((qh_val >> 6u) & 3u) << 4u)) - 32;

                let s0 = f32(read_i8(sc_off + is_idx));
                let s1 = f32(read_i8(sc_off + is_idx + 2u));
                let s2 = f32(read_i8(sc_off + is_idx + 4u));
                let s3 = f32(read_i8(sc_off + is_idx + 6u));

                let xb = x_off + elem_off + n;
                sum += d * s0 * f32(q1) * x[xb + l];
                sum += d * s1 * f32(q2) * x[xb + l + 32u];
                sum += d * s2 * f32(q3) * x[xb + l + 64u];
                sum += d * s3 * f32(q4) * x[xb + l + 96u];
            }
        }

        bi += 256u;
    }

    let result = workgroup_reduce(tid, sum);
    if (tid == 0u) {
        out[token * u.rows + row] = result;
    }
}
