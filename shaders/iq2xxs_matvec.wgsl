// IQ2_XXS TILED matvec — 2-bit codebook quantization
// 256-element blocks, 66 bytes each
//
// Tiled: TILE_ROWS=32, 8 threads per row, async (no per-block barriers).
// Dispatch: (ceil(rows / TILE_ROWS), n_tokens, 1)

const TILE_ROWS: u32 = 32u;
const WG_SIZE: u32 = 256u;
const THREADS_PER_ROW: u32 = 8u;
const ELEMS_PER_THREAD: u32 = 256u / THREADS_PER_ROW;
const BLOCK_SIZE: u32 = 256u;

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

var<workgroup> reduce_buf: array<f32, 256>;

fn fp16_to_f32(bits: u32) -> f32 {
    let sign = (bits >> 15u) & 1u;
    let exp = (bits >> 10u) & 0x1Fu;
    let mant = bits & 0x3FFu;
    if (exp == 0u && mant == 0u) {
        return select(0.0, -0.0, sign == 1u);
    }
    let f_bits = (sign << 31u) | ((exp + 112u) << 23u) | (mant << 13u);
    return bitcast<f32>(f_bits);
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

fn grid_byte(g: u32, j: u32) -> f32 {
    return f32((g >> (j * 8u)) & 0xFFu);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let tile_start = select(wid.x * TILE_ROWS, (wid.x + wid.y * uniforms.extra) * TILE_ROWS, uniforms.extra > 0u);
    let token = select(wid.y, 0u, uniforms.extra > 0u);
    let tid = lid.x;

    let local_row = tid / THREADS_PER_ROW;
    let local_elem = tid % THREADS_PER_ROW;
    let global_row = tile_start + local_row;

    let cols = uniforms.cols;
    let blocks_per_row = cols / BLOCK_SIZE;
    let block_bytes = 66u;
    let x_base = token * cols;
    let row_byte_offset = global_row * blocks_per_row * block_bytes;

    var acc: f32 = 0.0;

    if (global_row < uniforms.rows) {
        for (var b = 0u; b < blocks_per_row; b++) {
            let block_byte = row_byte_offset + b * block_bytes;
            let d = fp16_to_f32(read_u16(block_byte, 0u));
            let qs_base = block_byte + 2u;
            let elem_base = b * BLOCK_SIZE;

            let my_start = local_elem * ELEMS_PER_THREAD;

            for (var i = 0u; i < ELEMS_PER_THREAD; i++) {
                let elem = my_start + i;
                // 8 sub-blocks processed in pairs. Each pair = 64 elements.
                // ib32 pair = elem / 64, giving 4 pairs (ib32 = 0,2,4,6)
                let pair = elem / 64u;
                let in_pair = elem % 64u;
                let ib32_base = pair * 2u;
                // Within pair: 4 groups of 16 elements
                let l = in_pair / 16u;    // 0..3
                let j = in_pair % 16u;    // 0..15

                let q32_off = ib32_base * 8u;
                let q32_0 = read_u32_raw(qs_base, q32_off);
                let q32_1 = read_u32_raw(qs_base, q32_off + 4u);
                let q32_2 = read_u32_raw(qs_base, q32_off + 8u);
                let q32_3 = read_u32_raw(qs_base, q32_off + 12u);

                let db1 = d * (0.5 + f32(q32_1 >> 28u));
                let db2 = d * (0.5 + f32(q32_3 >> 28u));
                let db = select(db2, db1, l < 2u);

                var q32_l: u32;
                switch l {
                    case 0u: { q32_l = q32_0; }
                    case 1u: { q32_l = q32_1; }
                    case 2u: { q32_l = q32_2; }
                    default: { q32_l = q32_3; }
                }

                let grid_idx1 = q32_l & 0xFFu;
                let grid_idx2 = (q32_l >> 8u) & 0xFFu;
                let sign_idx = (q32_l >> 16u) & 0x7Fu;
                let signs = KSIGNS_IQ2XS[sign_idx];

                // 16 elements: first 8 from grid1 (with signs), next 8 from grid2 (no signs)
                if (j < 8u) {
                    // grid1 with sign
                    var g: u32;
                    if (j < 4u) {
                        g = IQ2XXS_GRID[grid_idx1 * 2u];
                    } else {
                        g = IQ2XXS_GRID[grid_idx1 * 2u + 1u];
                    }
                    let jj = j % 4u;
                    let w = grid_byte(g, jj);
                    let s = f32(1 - 2 * i32((signs >> j) & 1u));
                    acc += db * w * s * x[x_base + elem_base + elem];
                } else {
                    // grid2 without sign
                    let j2 = j - 8u;
                    var g: u32;
                    if (j2 < 4u) {
                        g = IQ2XXS_GRID[grid_idx2 * 2u];
                    } else {
                        g = IQ2XXS_GRID[grid_idx2 * 2u + 1u];
                    }
                    let jj = j2 % 4u;
                    acc += db * grid_byte(g, jj) * x[x_base + elem_base + elem];
                }
            }
        }
    }

    reduce_buf[tid] = acc;
    workgroupBarrier();

    let row_base = local_row * THREADS_PER_ROW;
    if (local_elem < 4u) { reduce_buf[row_base + local_elem] += reduce_buf[row_base + local_elem + 4u]; }
    workgroupBarrier();
    if (local_elem < 2u) { reduce_buf[row_base + local_elem] += reduce_buf[row_base + local_elem + 2u]; }
    workgroupBarrier();
    if (local_elem < 1u) { reduce_buf[row_base + local_elem] += reduce_buf[row_base + local_elem + 1u]; }
    workgroupBarrier();

    if (local_elem == 0u && global_row < uniforms.rows) {
        out[token * uniforms.rows + global_row] = reduce_buf[row_base];
    }
}
