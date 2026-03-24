// TQ1_0 TILED matvec — base-3 ternary packing
// 256-element blocks, 54 bytes each: qs[48] + qh[4] + d(FP16)
// qs[0..31]:  5 trits per byte x 32 bytes = 160 values
// qs[32..47]: 5 trits per byte x 16 bytes = 80 values
// qh[0..3]:   4 trits per byte x 4 bytes  = 16 values
// Total: 160 + 80 + 16 = 256 values
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
    let f_exp = f32(i32(exp) - 15 + 127);
    let f_bits = (sign << 31u) | (u32(f_exp) << 23u) | (mant << 13u);
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

fn decode_trit(byte_val: u32, pow3_val: u32) -> i32 {
    let q = (byte_val * pow3_val) & 0xFFu;
    let xi = (q * 3u) >> 8u;
    return i32(xi) - 1;
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
    let block_bytes = 54u;
    let x_base = token * cols;

    let row_byte_offset = global_row * blocks_per_row * block_bytes;

    let pow3 = array<u32, 5>(1u, 3u, 9u, 27u, 81u);

    var acc: f32 = 0.0;

    if (global_row < uniforms.rows) {
        for (var b = 0u; b < blocks_per_row; b++) {
            let block_byte = row_byte_offset + b * block_bytes;
            let d = fp16_to_f32(read_u16(block_byte, 52u));
            let elem_base = b * BLOCK_SIZE;

            // Each thread handles 32 elements (256 / 8)
            // Elements are mapped across 3 sections:
            //   Section 1: elements 0..159 (qs[0..31], 5 trits each)
            //   Section 2: elements 160..239 (qs[32..47], 5 trits each)
            //   Section 3: elements 240..255 (qh[0..3], 4 trits each)
            let my_start = local_elem * ELEMS_PER_THREAD;
            var block_sum: f32 = 0.0;

            for (var i = 0u; i < ELEMS_PER_THREAD; i++) {
                let elem = my_start + i;
                var w: i32 = 0;

                if (elem < 160u) {
                    // Section 1: qs[0..31], 5 trits per byte
                    let n = elem / 32u;  // trit index 0..4
                    let m = elem % 32u;  // byte index 0..31
                    let byte_val = read_u8(block_byte, m);
                    w = decode_trit(byte_val, pow3[n]);
                } else if (elem < 240u) {
                    // Section 2: qs[32..47], 5 trits per byte
                    let idx = elem - 160u;
                    let n = idx / 16u;   // trit index 0..4
                    let m = idx % 16u;   // byte index 0..15
                    let byte_val = read_u8(block_byte, 32u + m);
                    w = decode_trit(byte_val, pow3[n]);
                } else {
                    // Section 3: qh[0..3], 4 trits per byte
                    let idx = elem - 240u;
                    let n = idx / 4u;    // trit index 0..3
                    let m = idx % 4u;    // byte index 0..3
                    let byte_val = read_u8(block_byte, 48u + m);
                    w = decode_trit(byte_val, pow3[n]);
                }

                block_sum += f32(w) * x[x_base + elem_base + elem];
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
