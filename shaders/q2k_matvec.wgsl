// Q2_K matvec — 2-bit k-quant, 256 elements per block, 84 bytes/block
// Layout: scales[16] (bytes 0-15), qs[64] (bytes 16-79), d FP16 (bytes 80-81), dmin FP16 (bytes 82-83)

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

const BLOCK_BYTES: u32 = 84u;
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

        // scales at bytes 0-15, qs at bytes 16-79, d at 80, dmin at 82
        let d    = fp16_to_f32(read_u16(base + 80u));
        let dmin = fp16_to_f32(read_u16(base + 82u));
        let qs_base = base + 16u;
        let elem_off = bi * QK_K;

        var is_idx = 0u;
        var out_idx = 0u;
        // Two halves of 128 elements each
        for (var n = 0u; n < QK_K; n += 128u) {
            var shift = 0u;
            let q_start = qs_base + n / 4u; // 32 bytes of qs per 128 elements
            for (var j = 0u; j < 4u; j++) {
                // First sub-block of 16
                let sc0 = read_u8(base + is_idx);
                is_idx++;
                var dl = d * f32(sc0 & 0xFu);
                var ml = dmin * f32(sc0 >> 4u);
                for (var l = 0u; l < 16u; l++) {
                    let qbyte = read_u8(q_start + l);
                    let qval = (qbyte >> shift) & 3u;
                    sum += (dl * f32(qval) - ml) * x[x_off + elem_off + out_idx];
                    out_idx++;
                }

                // Second sub-block of 16
                let sc1 = read_u8(base + is_idx);
                is_idx++;
                dl = d * f32(sc1 & 0xFu);
                ml = dmin * f32(sc1 >> 4u);
                for (var l = 0u; l < 16u; l++) {
                    let qbyte = read_u8(q_start + l + 16u);
                    let qval = (qbyte >> shift) & 3u;
                    sum += (dl * f32(qval) - ml) * x[x_off + elem_off + out_idx];
                    out_idx++;
                }

                shift += 2u;
            }
        }

        bi += 256u;
    }

    let result = workgroup_reduce(tid, sum);
    if (tid == 0u) {
        out[token * u.rows + row] = result;
    }
}
