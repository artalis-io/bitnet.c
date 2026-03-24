// IQ2_XXS matvec — 2-bit codebook quantization
// 256-element blocks, 66 bytes each:
//   d: FP16 (bytes 0-1)
//   qs[64]: packed grid indices + signs/scales (bytes 2-65), read as 16 uint32
// Layout: qs is processed as uint32[16]. For each pair of 32-element sub-blocks (ib32 += 2):
//   q32[0..3] = 4 uint32 values covering 2 sub-blocks
//   q32[1] >> 28 = scale for first sub-block, q32[3] >> 28 = scale for second
//   Each q32[l]: bits[0:7] = grid index 1, bits[8:15] = grid index 2,
//                bits[16:22] = sign index, bits[24:27] = (unused or scale)
//   grid1 = iq2xxs_grid[q32[l] & 0xFF] -> 8 weight values (as uint64 = 2 x u32)
//   grid2 = iq2xxs_grid[(q32[l] >> 8) & 0xFF] -> 8 weight values
//   signs applied to grid1 only (8 bits from ksigns_iq2xs)

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

fn read_u32_raw(base: u32, offset: u32) -> u32 {
    // Unaligned u32 read: handles any byte alignment
    let addr = base + offset;
    let align = addr & 3u;
    if (align == 0u) {
        return weights[addr >> 2u];
    }
    let lo_word = weights[addr >> 2u];
    let hi_word = weights[(addr >> 2u) + 1u];
    let shift = align * 8u;
    return (lo_word >> shift) | (hi_word << (32u - shift));
}

// Sign lookup: 7-bit index -> 8-bit sign mask
const KSIGNS_IQ2XS: array<u32, 128> = array<u32, 128>(
      0u, 129u, 130u,   3u, 132u,   5u,   6u, 135u, 136u,   9u,  10u, 139u,  12u, 141u, 142u,  15u,
    144u,  17u,  18u, 147u,  20u, 149u, 150u,  23u,  24u, 153u, 154u,  27u, 156u,  29u,  30u, 159u,
    160u,  33u,  34u, 163u,  36u, 165u, 166u,  39u,  40u, 169u, 170u,  43u, 172u,  45u,  46u, 175u,
     48u, 177u, 178u,  51u, 180u,  53u,  54u, 183u, 184u,  57u,  58u, 187u,  60u, 189u, 190u,  63u,
    192u,  65u,  66u, 195u,  68u, 197u, 198u,  71u,  72u, 201u, 202u,  75u, 204u,  77u,  78u, 207u,
     80u, 209u, 210u,  83u, 212u,  85u,  86u, 215u, 216u,  89u,  90u, 219u,  92u, 221u, 222u,  95u,
     96u, 225u, 226u,  99u, 228u, 101u, 102u, 231u, 232u, 105u, 106u, 235u, 108u, 237u, 238u, 111u,
    240u, 113u, 114u, 243u, 116u, 245u, 246u, 119u, 120u, 249u, 250u, 123u, 252u, 125u, 126u, 255u
);

// IQ2XXS grid: 256 entries of uint64 stored as 512 uint32 (lo, hi pairs)
// Each entry = 8 weight byte values. Access: grid[idx*2] = lo 4 bytes, grid[idx*2+1] = hi 4 bytes
const IQ2XXS_GRID: array<u32, 512> = array<u32, 512>(
    0x08080808u, 0x08080808u, 0x0808082bu, 0x08080808u, 0x08081919u, 0x08080808u, 0x08082b08u, 0x08080808u,
    0x08082b2bu, 0x08080808u, 0x08190819u, 0x08080808u, 0x08191908u, 0x08080808u, 0x082b0808u, 0x08080808u,
    0x082b082bu, 0x08080808u, 0x082b2b08u, 0x08080808u, 0x082b2b2bu, 0x08080808u, 0x19080819u, 0x08080808u,
    0x19081908u, 0x08080808u, 0x19190808u, 0x08080808u, 0x19192b08u, 0x08080808u, 0x192b0819u, 0x08080808u,
    0x192b1908u, 0x08080808u, 0x2b080808u, 0x08080808u, 0x2b08082bu, 0x08080808u, 0x2b082b2bu, 0x08080808u,
    0x2b2b082bu, 0x08080808u, 0x08080819u, 0x08080819u, 0x08081908u, 0x08080819u, 0x08190808u, 0x08080819u,
    0x08191919u, 0x08080819u, 0x19080808u, 0x08080819u, 0x2b081908u, 0x08080819u, 0x2b192b08u, 0x08080819u,
    0x08080808u, 0x0808082bu, 0x0808082bu, 0x0808082bu, 0x082b082bu, 0x0808082bu, 0x2b08082bu, 0x0808082bu,
    0x08080819u, 0x08081908u, 0x08081908u, 0x08081908u, 0x08190808u, 0x08081908u, 0x082b0819u, 0x08081908u,
    0x082b1908u, 0x08081908u, 0x19080808u, 0x08081908u, 0x1908082bu, 0x08081908u, 0x19082b08u, 0x08081908u,
    0x192b0808u, 0x08081908u, 0x2b080819u, 0x08081908u, 0x2b081908u, 0x08081908u, 0x2b190808u, 0x08081908u,
    0x2b2b1908u, 0x08081908u, 0x08080808u, 0x08081919u, 0x0808082bu, 0x08081919u, 0x08082b08u, 0x08081919u,
    0x082b0808u, 0x08081919u, 0x1908192bu, 0x08081919u, 0x192b2b19u, 0x08081919u, 0x2b080808u, 0x08081919u,
    0x2b190819u, 0x08081919u, 0x08082b19u, 0x0808192bu, 0x08190808u, 0x0808192bu, 0x19080808u, 0x0808192bu,
    0x2b081908u, 0x0808192bu, 0x2b2b1908u, 0x0808192bu, 0x08080808u, 0x08082b08u, 0x08081919u, 0x08082b08u,
    0x08082b08u, 0x08082b08u, 0x08191908u, 0x08082b08u, 0x082b2b08u, 0x08082b08u, 0x19080819u, 0x08082b08u,
    0x19081908u, 0x08082b08u, 0x19190808u, 0x08082b08u, 0x1919082bu, 0x08082b08u, 0x2b082b08u, 0x08082b08u,
    0x08081908u, 0x08082b19u, 0x19080808u, 0x08082b19u, 0x0808082bu, 0x08082b2bu, 0x08191908u, 0x08082b2bu,
    0x08080819u, 0x08190808u, 0x08081908u, 0x08190808u, 0x08190808u, 0x08190808u, 0x082b0819u, 0x08190808u,
    0x19080808u, 0x08190808u, 0x192b0808u, 0x08190808u, 0x2b081908u, 0x08190808u, 0x2b190808u, 0x08190808u,
    0x2b191919u, 0x08190808u, 0x08080808u, 0x08190819u, 0x08082b08u, 0x08190819u, 0x082b0808u, 0x08190819u,
    0x19190808u, 0x08190819u, 0x19192b2bu, 0x08190819u, 0x2b080808u, 0x08190819u, 0x082b1908u, 0x0819082bu,
    0x19081919u, 0x0819082bu, 0x08080808u, 0x08191908u, 0x08082b08u, 0x08191908u, 0x082b0808u, 0x08191908u,
    0x082b1919u, 0x08191908u, 0x19082b19u, 0x08191908u, 0x2b080808u, 0x08191908u, 0x08192b08u, 0x08191919u,
    0x192b082bu, 0x08191919u, 0x08080808u, 0x0819192bu, 0x0819192bu, 0x0819192bu, 0x08080819u, 0x08192b08u,
    0x08081908u, 0x08192b08u, 0x08190808u, 0x08192b08u, 0x19080808u, 0x08192b08u, 0x2b080819u, 0x08192b08u,
    0x08080808u, 0x08192b19u, 0x08081919u, 0x08192b19u, 0x2b2b0808u, 0x08192b19u, 0x19190819u, 0x08192b2bu,
    0x08080808u, 0x082b0808u, 0x0808082bu, 0x082b0808u, 0x08082b2bu, 0x082b0808u, 0x19081908u, 0x082b0808u,
    0x192b0819u, 0x082b0808u, 0x2b080808u, 0x082b0808u, 0x2b08082bu, 0x082b0808u, 0x082b2b19u, 0x082b0819u,
    0x19082b08u, 0x082b0819u, 0x08080808u, 0x082b082bu, 0x0808082bu, 0x082b082bu, 0x08080819u, 0x082b1908u,
    0x08081908u, 0x082b1908u, 0x08190808u, 0x082b1908u, 0x19080808u, 0x082b1908u, 0x1919192bu, 0x082b1908u,
    0x08080808u, 0x082b1919u, 0x19080819u, 0x082b1919u, 0x192b1908u, 0x082b1919u, 0x2b190808u, 0x082b192bu,
    0x08082b08u, 0x082b2b08u, 0x082b0808u, 0x082b2b08u, 0x2b191908u, 0x082b2b08u, 0x19081908u, 0x082b2b2bu,
    0x08080819u, 0x19080808u, 0x08081908u, 0x19080808u, 0x08190808u, 0x19080808u, 0x08192b08u, 0x19080808u,
    0x082b0819u, 0x19080808u, 0x082b1908u, 0x19080808u, 0x19080808u, 0x19080808u, 0x19082b08u, 0x19080808u,
    0x1919192bu, 0x19080808u, 0x192b0808u, 0x19080808u, 0x2b080819u, 0x19080808u, 0x2b081908u, 0x19080808u,
    0x2b190808u, 0x19080808u, 0x08080808u, 0x19080819u, 0x082b0808u, 0x19080819u, 0x192b0819u, 0x19080819u,
    0x2b080808u, 0x19080819u, 0x2b081919u, 0x19080819u, 0x08080819u, 0x1908082bu, 0x08190808u, 0x1908082bu,
    0x19082b08u, 0x1908082bu, 0x1919192bu, 0x1908082bu, 0x192b2b08u, 0x1908082bu, 0x08080808u, 0x19081908u,
    0x08082b08u, 0x19081908u, 0x082b0808u, 0x19081908u, 0x2b080808u, 0x19081908u, 0x2b192b19u, 0x19081908u,
    0x0819082bu, 0x19081919u, 0x082b1908u, 0x19081919u, 0x08080808u, 0x1908192bu, 0x08080819u, 0x19082b08u,
    0x08081908u, 0x19082b08u, 0x08190808u, 0x19082b08u, 0x19080808u, 0x19082b08u, 0x19081919u, 0x19082b08u,
    0x08080808u, 0x19082b19u, 0x19192b08u, 0x19082b19u, 0x192b0819u, 0x19082b19u, 0x2b08082bu, 0x19082b19u,
    0x19081919u, 0x19082b2bu, 0x2b190808u, 0x19082b2bu, 0x08080808u, 0x19190808u, 0x08082b08u, 0x19190808u,
    0x08190819u, 0x19190808u, 0x08192b19u, 0x19190808u, 0x082b0808u, 0x19190808u, 0x2b080808u, 0x19190808u,
    0x2b082b08u, 0x19190808u, 0x08081908u, 0x19190819u, 0x1908082bu, 0x19190819u, 0x2b2b1908u, 0x19190819u,
    0x2b190819u, 0x1919082bu, 0x2b190808u, 0x19191908u, 0x2b19082bu, 0x19191908u, 0x08082b2bu, 0x19191919u,
    0x08080819u, 0x1919192bu, 0x19191908u, 0x1919192bu, 0x08080808u, 0x19192b08u, 0x08190819u, 0x19192b08u,
    0x08192b19u, 0x19192b08u, 0x192b1908u, 0x19192b08u, 0x19080808u, 0x19192b19u, 0x08082b08u, 0x19192b2bu,
    0x08081908u, 0x192b0808u, 0x08190808u, 0x192b0808u, 0x19080808u, 0x192b0808u, 0x192b2b08u, 0x192b0808u,
    0x08080808u, 0x192b0819u, 0x19191919u, 0x192b0819u, 0x08192b08u, 0x192b082bu, 0x192b0808u, 0x192b082bu,
    0x08080808u, 0x192b1908u, 0x08081919u, 0x192b1908u, 0x08190808u, 0x192b1919u, 0x0819082bu, 0x192b1919u,
    0x2b081908u, 0x192b1919u, 0x1908082bu, 0x192b2b08u, 0x08080808u, 0x2b080808u, 0x0808082bu, 0x2b080808u,
    0x08082b2bu, 0x2b080808u, 0x19080819u, 0x2b080808u, 0x2b08082bu, 0x2b080808u, 0x08081908u, 0x2b080819u,
    0x08192b08u, 0x2b080819u, 0x19080808u, 0x2b080819u, 0x08190819u, 0x2b08082bu, 0x08080819u, 0x2b081908u,
    0x08081908u, 0x2b081908u, 0x08190808u, 0x2b081908u, 0x08191919u, 0x2b081908u, 0x19080808u, 0x2b081908u,
    0x192b0808u, 0x2b081908u, 0x08080808u, 0x2b081919u, 0x1908192bu, 0x2b081919u, 0x2b191908u, 0x2b081919u,
    0x08082b19u, 0x2b08192bu, 0x19080808u, 0x2b08192bu, 0x192b0808u, 0x2b08192bu, 0x0808082bu, 0x2b082b08u,
    0x08081908u, 0x2b082b19u, 0x08190819u, 0x2b082b2bu, 0x08081908u, 0x2b190808u, 0x08190808u, 0x2b190808u,
    0x082b1908u, 0x2b190808u, 0x19080808u, 0x2b190808u, 0x2b2b0819u, 0x2b190808u, 0x0819192bu, 0x2b190819u,
    0x2b080808u, 0x2b190819u, 0x19081919u, 0x2b19082bu, 0x08080808u, 0x2b191908u, 0x082b082bu, 0x2b191908u,
    0x19081908u, 0x2b191908u, 0x19190819u, 0x2b191919u, 0x2b080819u, 0x2b192b08u, 0x082b0808u, 0x2b192b19u,
    0x0808082bu, 0x2b2b0808u, 0x19190808u, 0x2b2b0808u, 0x2b081919u, 0x2b2b0808u, 0x08082b19u, 0x2b2b0819u,
    0x08080808u, 0x2b2b082bu, 0x08192b08u, 0x2b2b1908u, 0x19190808u, 0x2b2b2b08u, 0x08081908u, 0x2b2b2b19u
);

// Extract byte j from a u32 grid value
fn grid_byte(g: u32, j: u32) -> f32 {
    return f32((g >> (j * 8u)) & 0xFFu);
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
    let blocks_per_row = cols / 256u;
    let block_bytes = 66u;
    let row_byte_offset = row * blocks_per_row * block_bytes;
    let x_offset = token * cols;

    var sum = 0.0f;

    var block_idx = tid;
    while (block_idx < blocks_per_row) {
        let block_byte = row_byte_offset + block_idx * block_bytes;
        let elem_base = block_idx * 256u;

        // FP16 scale at offset 0
        let d = fp16_to_f32(read_u16(block_byte, 0u));

        // qs data starts at offset 2 (64 bytes = 16 uint32)
        let qs_base = block_byte + 2u;

        // Process in pairs of 32-element sub-blocks (each pair = 64 elements)
        var xi = elem_base;
        for (var ib32 = 0u; ib32 < 8u; ib32 += 2u) {
            // Read 4 uint32 values for this pair of sub-blocks
            let q32_off = ib32 * 8u; // 4 uint32 * 4 bytes per sub-block pair, 2 sub-blocks
            let q32_0 = read_u32_raw(qs_base, q32_off);
            let q32_1 = read_u32_raw(qs_base, q32_off + 4u);
            let q32_2 = read_u32_raw(qs_base, q32_off + 8u);
            let q32_3 = read_u32_raw(qs_base, q32_off + 12u);

            let db1 = d * (0.5 + f32(q32_1 >> 28u));
            let db2 = d * (0.5 + f32(q32_3 >> 28u));

            // 4 iterations, each processing 16 elements
            var q32_arr = array<u32, 4>(q32_0, q32_1, q32_2, q32_3);
            for (var l = 0u; l < 4u; l++) {
                let q32_l = q32_arr[l];
                let grid_idx1 = q32_l & 0xFFu;
                let grid_idx2 = (q32_l >> 8u) & 0xFFu;
                let sign_idx = (q32_l >> 16u) & 0x7Fu;
                let signs = KSIGNS_IQ2XS[sign_idx];
                let db = select(db2, db1, l < 2u);

                // grid1: 8 weight values (uint64 = 2 x u32)
                let g1_lo = IQ2XXS_GRID[grid_idx1 * 2u];
                let g1_hi = IQ2XXS_GRID[grid_idx1 * 2u + 1u];

                // grid2: 8 weight values
                let g2_lo = IQ2XXS_GRID[grid_idx2 * 2u];
                let g2_hi = IQ2XXS_GRID[grid_idx2 * 2u + 1u];

                // First 8 elements from grid1 with sign application
                for (var j = 0u; j < 4u; j++) {
                    let w = grid_byte(g1_lo, j);
                    let s = f32(1 - 2 * i32((signs >> j) & 1u));
                    sum += db * w * s * x[x_offset + xi];
                    xi++;
                }
                for (var j = 0u; j < 4u; j++) {
                    let w = grid_byte(g1_hi, j);
                    let s = f32(1 - 2 * i32((signs >> (j + 4u)) & 1u));
                    sum += db * w * s * x[x_offset + xi];
                    xi++;
                }

                // Next 8 elements from grid2 without sign application
                for (var j = 0u; j < 4u; j++) {
                    sum += db * grid_byte(g2_lo, j) * x[x_offset + xi];
                    xi++;
                }
                for (var j = 0u; j < 4u; j++) {
                    sum += db * grid_byte(g2_hi, j) * x[x_offset + xi];
                    xi++;
                }
            }
        }

        block_idx += 256u;
    }

    let result = workgroup_reduce(tid, sum);

    if (tid == 0u) {
        out[token * uniforms.rows + row] = result;
    }
}
