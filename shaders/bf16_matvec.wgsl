// BF16 matvec/matmul — bfloat16 (no block structure)
// 2 bytes per element, decode: f32 = bitcast<f32>(u32(bf16_bits) << 16u)

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

fn bf16_to_f32(bits: u32) -> f32 {
    return bitcast<f32>(bits << 16u);
}

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
    // BF16: 2 bytes per element, so each u32 holds 2 elements
    // Row offset in u32s: row * cols / 2
    let row_u32_offset = row * (cols / 2u);
    let x_offset = token * cols;

    var sum = 0.0f;

    // Each thread processes elements starting at tid, stepping by 256
    // Process 2 elements per u32 read
    var pair_idx = tid;
    let pairs_per_row = cols / 2u;
    while (pair_idx < pairs_per_row) {
        let word = weights[row_u32_offset + pair_idx];

        let lo_bits = word & 0xFFFFu;
        let hi_bits = (word >> 16u) & 0xFFFFu;

        let v0 = bf16_to_f32(lo_bits);
        let v1 = bf16_to_f32(hi_bits);

        let elem_idx = pair_idx * 2u;
        sum += v0 * x[x_offset + elem_idx];
        sum += v1 * x[x_offset + elem_idx + 1u];

        pair_idx += 256u;
    }

    let result = workgroup_reduce(tid, sum);

    if (tid == 0u) {
        out[token * uniforms.rows + row] = result;
    }
}
