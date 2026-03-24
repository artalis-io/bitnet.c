// IQ4_XS matvec — 4-bit non-linear with sub-block scales
// 256-element super-blocks, 136 bytes each:
//   d: FP16 (bytes 0-1)
//   scales_h: u16 (bytes 2-3) — high 2 bits of 8 6-bit scales
//   scales_l[4]: bytes (4-7) — low 4 bits of 8 scales (nibble-packed)
//   qs[128]: 4-bit codebook indices (bytes 8-135)
// 8 sub-blocks of 32 elements each, same IQ4NL codebook

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

fn read_u16(base: u32, offset: u32) -> u32 {
    let addr = base + offset;
    let word = weights[addr >> 2u];
    let shift = (addr & 2u) * 8u;
    return (word >> shift) & 0xFFFFu;
}

const IQ4NL_VALS: array<i32, 16> = array<i32, 16>(
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
);

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
    let block_bytes = 136u;
    let row_byte_offset = row * blocks_per_row * block_bytes;
    let x_offset = token * cols;

    var sum = 0.0f;

    var block_idx = tid;
    while (block_idx < blocks_per_row) {
        let block_byte = row_byte_offset + block_idx * block_bytes;
        let elem_base = block_idx * 256u;

        // FP16 super-block scale at offset 0
        let d = fp16_to_f32(read_u16(block_byte, 0u));
        // High bits of scales at offset 2
        let scales_h = read_u16(block_byte, 2u);

        // 8 sub-blocks of 32 elements each
        for (var j = 0u; j < 8u; j++) {
            // Extract 6-bit scale: 4 low bits from scales_l, 2 high bits from scales_h
            let lo = (read_u8(block_byte, 4u + j / 2u) >> ((j % 2u) * 4u)) & 0xFu;
            let hi = (scales_h >> (j * 2u)) & 3u;
            let scale_val = lo | (hi << 4u);
            let dl = d * f32(i32(scale_val) - 32);

            // 16 bytes of packed 4-bit codebook indices -> 32 elements
            let qs_offset = 8u + j * 16u;
            var sub_sum = 0.0f;
            for (var i = 0u; i < 16u; i++) {
                let byte_val = read_u8(block_byte, qs_offset + i);
                let lo_idx = byte_val & 0xFu;
                let hi_idx = (byte_val >> 4u) & 0xFu;

                sub_sum += f32(IQ4NL_VALS[lo_idx]) * x[x_offset + elem_base + j * 32u + i];
                sub_sum += f32(IQ4NL_VALS[hi_idx]) * x[x_offset + elem_base + j * 32u + i + 16u];
            }
            sum += sub_sum * dl;
        }

        block_idx += 256u;
    }

    let result = workgroup_reduce(tid, sum);

    if (tid == 0u) {
        out[token * uniforms.rows + row] = result;
    }
}
