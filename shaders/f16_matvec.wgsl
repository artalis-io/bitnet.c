// F16 matvec: out[row] = sum(fp16_weights[row, :] * x[:])
// Weight data: packed as u32 (two FP16 values per u32)
// No block structure — direct element-wise dot product.

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Uniforms { rows: u32, cols: u32, n_tokens: u32, extra: u32 }
@group(0) @binding(3) var<uniform> u: Uniforms;

var<workgroup> shared_data: array<f32, 256>;

fn fp16_to_f32(bits: u32) -> f32 {
    let sign = (bits >> 15u) & 1u;
    let exp = (bits >> 10u) & 0x1Fu;
    let mant = bits & 0x3FFu;
    if (exp == 0u && mant == 0u) { return select(0.0, -0.0, sign == 1u); }
    let f_bits = (sign << 31u) | ((exp + 112u) << 23u) | (mant << 13u);
    return bitcast<f32>(f_bits);
}

fn workgroup_reduce(lid: u32, val: f32) -> f32 {
    shared_data[lid] = val;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid < s) { shared_data[lid] += shared_data[lid + s]; }
        workgroupBarrier();
    }
    return shared_data[0];
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = gid.x;
    let token = gid.y;
    if (row >= u.rows) { return; }

    let cols = u.cols;
    let x_base = token * cols;
    // F16: 2 bytes per element, packed as u32 (2 values per word)
    let words_per_row = cols / 2u;
    let w_base = row * words_per_row;

    var partial: f32 = 0.0;
    // Each thread processes elements stride-256
    for (var i = lid.x; i < words_per_row; i += 256u) {
        let word = weights[w_base + i];
        let w0 = fp16_to_f32(word & 0xFFFFu);
        let w1 = fp16_to_f32(word >> 16u);
        let col = i * 2u;
        partial += w0 * x[x_base + col] + w1 * x[x_base + col + 1u];
    }

    let result = workgroup_reduce(lid.x, partial);
    if (lid.x == 0u) {
        out[token * u.rows + row] = result;
    }
}
