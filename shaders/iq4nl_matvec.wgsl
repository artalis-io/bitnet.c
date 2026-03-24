// IQ4_NL TILED matvec — 4-bit non-linear quantization with codebook lookup
// 32-element blocks, 18 bytes each: d(FP16) + qs[16] packed 4-bit codebook indices
//
// Tiled: TILE_ROWS=32, 8 threads per row, async (no per-block barriers).
// Dispatch: (ceil(rows / TILE_ROWS), n_tokens, 1)

const TILE_ROWS: u32 = 32u;
const WG_SIZE: u32 = 256u;
const THREADS_PER_ROW: u32 = 8u;
const ELEMS_PER_THREAD: u32 = 32u / THREADS_PER_ROW;

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

fn read_byte(addr: u32) -> u32 {
    return (weights[addr >> 2u] >> ((addr & 3u) * 8u)) & 0xFFu;
}

fn read_u16_at(addr: u32) -> u32 {
    let lo = read_byte(addr);
    let hi = read_byte(addr + 1u);
    return lo | (hi << 8u);
}

const IQ4NL_VALS: array<i32, 16> = array<i32, 16>(
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
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
    let blocks_per_row = cols / 32u;
    let x_base = token * cols;

    let row_byte_base = global_row * blocks_per_row * 18u;

    var acc: f32 = 0.0;

    if (global_row < uniforms.rows) {
        for (var b = 0u; b < blocks_per_row; b++) {
            let block_byte = row_byte_base + b * 18u;
            let d = fp16_to_f32(read_u16_at(block_byte));
            let elem_base = b * 32u;

            let my_start = local_elem * ELEMS_PER_THREAD;
            var block_sum: f32 = 0.0;
            for (var i = 0u; i < ELEMS_PER_THREAD; i++) {
                let elem = my_start + i;
                var val: i32;
                if (elem < 16u) {
                    let byte_val = read_byte(block_byte + 2u + elem);
                    val = IQ4NL_VALS[byte_val & 0xFu];
                } else {
                    let byte_val = read_byte(block_byte + 2u + elem - 16u);
                    val = IQ4NL_VALS[(byte_val >> 4u) & 0xFu];
                }
                block_sum += f32(val) * x[x_base + elem_base + elem];
            }
            acc += block_sum * d;
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
