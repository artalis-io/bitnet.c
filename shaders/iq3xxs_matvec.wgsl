// IQ3_XXS TILED matvec — 3-bit codebook quantization
// 256-element blocks, 98 bytes each
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

const IQ3XXS_GRID: array<u32, 256> = array<u32, 256>(
    0x04040404u, 0x04040414u, 0x04040424u, 0x04040c0cu, 0x04040c1cu, 0x04040c3eu, 0x04041404u, 0x04041414u,
    0x04041c0cu, 0x04042414u, 0x04043e1cu, 0x04043e2cu, 0x040c040cu, 0x040c041cu, 0x040c0c04u, 0x040c0c14u,
    0x040c140cu, 0x040c142cu, 0x040c1c04u, 0x040c1c14u, 0x040c240cu, 0x040c2c24u, 0x040c3e04u, 0x04140404u,
    0x04140414u, 0x04140424u, 0x04140c0cu, 0x04141404u, 0x04141414u, 0x04141c0cu, 0x04141c1cu, 0x04141c3eu,
    0x04142c0cu, 0x04142c3eu, 0x04143e2cu, 0x041c040cu, 0x041c043eu, 0x041c0c04u, 0x041c0c14u, 0x041c142cu,
    0x041c3e04u, 0x04240c1cu, 0x04241c3eu, 0x04242424u, 0x04242c3eu, 0x04243e1cu, 0x04243e2cu, 0x042c040cu,
    0x042c043eu, 0x042c1c14u, 0x042c2c14u, 0x04341c2cu, 0x04343424u, 0x043e0c04u, 0x043e0c24u, 0x043e0c34u,
    0x043e241cu, 0x043e340cu, 0x0c04040cu, 0x0c04041cu, 0x0c040c04u, 0x0c040c14u, 0x0c04140cu, 0x0c04141cu,
    0x0c041c04u, 0x0c041c14u, 0x0c041c24u, 0x0c04243eu, 0x0c042c04u, 0x0c0c0404u, 0x0c0c0414u, 0x0c0c0c0cu,
    0x0c0c1404u, 0x0c0c1414u, 0x0c14040cu, 0x0c14041cu, 0x0c140c04u, 0x0c140c14u, 0x0c14140cu, 0x0c141c04u,
    0x0c143e14u, 0x0c1c0404u, 0x0c1c0414u, 0x0c1c1404u, 0x0c1c1c0cu, 0x0c1c2434u, 0x0c1c3434u, 0x0c24040cu,
    0x0c24042cu, 0x0c242c04u, 0x0c2c1404u, 0x0c2c1424u, 0x0c2c2434u, 0x0c2c3e0cu, 0x0c34042cu, 0x0c3e1414u,
    0x0c3e2404u, 0x14040404u, 0x14040414u, 0x14040c0cu, 0x14040c1cu, 0x14041404u, 0x14041414u, 0x14041434u,
    0x14041c0cu, 0x14042414u, 0x140c040cu, 0x140c041cu, 0x140c042cu, 0x140c0c04u, 0x140c0c14u, 0x140c140cu,
    0x140c1c04u, 0x140c341cu, 0x140c343eu, 0x140c3e04u, 0x14140404u, 0x14140414u, 0x14140c0cu, 0x14140c3eu,
    0x14141404u, 0x14141414u, 0x14141c3eu, 0x14142404u, 0x14142c2cu, 0x141c040cu, 0x141c0c04u, 0x141c0c24u,
    0x141c3e04u, 0x141c3e24u, 0x14241c2cu, 0x14242c1cu, 0x142c041cu, 0x142c143eu, 0x142c240cu, 0x142c3e24u,
    0x143e040cu, 0x143e041cu, 0x143e0c34u, 0x143e242cu, 0x1c04040cu, 0x1c040c04u, 0x1c040c14u, 0x1c04140cu,
    0x1c04141cu, 0x1c042c04u, 0x1c04342cu, 0x1c043e14u, 0x1c0c0404u, 0x1c0c0414u, 0x1c0c1404u, 0x1c0c1c0cu,
    0x1c0c2424u, 0x1c0c2434u, 0x1c14040cu, 0x1c14041cu, 0x1c140c04u, 0x1c14142cu, 0x1c142c14u, 0x1c143e14u,
    0x1c1c0c0cu, 0x1c1c1c1cu, 0x1c241c04u, 0x1c24243eu, 0x1c243e14u, 0x1c2c0404u, 0x1c2c0434u, 0x1c2c1414u,
    0x1c2c2c2cu, 0x1c340c24u, 0x1c341c34u, 0x1c34341cu, 0x1c3e1c1cu, 0x1c3e3404u, 0x24040424u, 0x24040c3eu,
    0x24041c2cu, 0x24041c3eu, 0x24042c1cu, 0x24042c3eu, 0x240c3e24u, 0x24141404u, 0x24141c3eu, 0x24142404u,
    0x24143404u, 0x24143434u, 0x241c043eu, 0x241c242cu, 0x24240424u, 0x24242c0cu, 0x24243424u, 0x242c142cu,
    0x242c241cu, 0x242c3e04u, 0x243e042cu, 0x243e0c04u, 0x243e0c14u, 0x243e1c04u, 0x2c040c14u, 0x2c04240cu,
    0x2c043e04u, 0x2c0c0404u, 0x2c0c0434u, 0x2c0c1434u, 0x2c0c2c2cu, 0x2c140c24u, 0x2c141c14u, 0x2c143e14u,
    0x2c1c0414u, 0x2c1c2c1cu, 0x2c240c04u, 0x2c24141cu, 0x2c24143eu, 0x2c243e14u, 0x2c2c0414u, 0x2c2c1c0cu,
    0x2c342c04u, 0x2c3e1424u, 0x2c3e2414u, 0x34041424u, 0x34042424u, 0x34042434u, 0x34043424u, 0x340c140cu,
    0x340c340cu, 0x34140c3eu, 0x34143424u, 0x341c1c04u, 0x341c1c34u, 0x34242424u, 0x342c042cu, 0x342c2c14u,
    0x34341c1cu, 0x343e041cu, 0x343e140cu, 0x3e04041cu, 0x3e04042cu, 0x3e04043eu, 0x3e040c04u, 0x3e041c14u,
    0x3e042c14u, 0x3e0c1434u, 0x3e0c2404u, 0x3e140c14u, 0x3e14242cu, 0x3e142c14u, 0x3e1c0404u, 0x3e1c0c2cu,
    0x3e1c1c1cu, 0x3e1c3404u, 0x3e24140cu, 0x3e24240cu, 0x3e2c0404u, 0x3e2c0414u, 0x3e2c1424u, 0x3e341c04u
);

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
    let block_bytes = 98u;
    let x_base = token * cols;
    let row_byte_offset = global_row * blocks_per_row * block_bytes;

    var acc: f32 = 0.0;

    if (global_row < uniforms.rows) {
        for (var b = 0u; b < blocks_per_row; b++) {
            let block_byte = row_byte_offset + b * block_bytes;
            let d = fp16_to_f32(read_u16(block_byte, 0u));
            let qs_base = block_byte + 2u;
            let sas_base = block_byte + 66u;
            let elem_base = b * BLOCK_SIZE;

            // Each thread handles 32 elements (256 / 8)
            let my_start = local_elem * ELEMS_PER_THREAD;

            for (var i = 0u; i < ELEMS_PER_THREAD; i++) {
                let elem = my_start + i;
                // 8 sub-blocks of 32 elements, each sub-block has 4 groups of 8
                let ib32 = elem / 32u;
                let in_sub = elem % 32u;
                let l = in_sub / 8u;      // group 0..3
                let j = in_sub % 8u;      // element within group 0..7

                let sas_addr = sas_base + ib32 * 4u;
                let aux32 = read_u32_raw(0u, sas_addr);
                let db = d * (0.5 + f32(aux32 >> 28u)) * 0.5;

                let signs = KSIGNS_IQ2XS[(aux32 >> (7u * l)) & 0x7Fu];
                let qs_off = ib32 * 8u;
                let grid_idx0 = read_u8(qs_base, qs_off + 2u * l);
                let grid_idx1 = read_u8(qs_base, qs_off + 2u * l + 1u);

                var w: f32;
                if (j < 4u) {
                    let grid0 = IQ3XXS_GRID[grid_idx0];
                    w = f32((grid0 >> (j * 8u)) & 0xFFu);
                    if ((signs & (1u << j)) != 0u) { w = -w; }
                } else {
                    let grid1 = IQ3XXS_GRID[grid_idx1];
                    w = f32((grid1 >> ((j - 4u) * 8u)) & 0xFFu);
                    if ((signs & (1u << j)) != 0u) { w = -w; }
                }

                acc += db * w * x[x_base + elem_base + elem];
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
