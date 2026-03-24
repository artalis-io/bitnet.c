// F32 matvec: out[row] = sum(weights[row, :] * x[:])
// Weight data: direct f32 array. No block structure, no scaling.

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Uniforms { rows: u32, cols: u32, n_tokens: u32, extra: u32 }
@group(0) @binding(3) var<uniform> u: Uniforms;

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

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x;
    let token = wid.y;
    if (row >= u.rows) { return; }

    let cols = u.cols;
    let x_base = token * cols;
    let w_base = row * cols;

    var partial: f32 = 0.0;
    for (var i = lid.x; i < cols; i += 256u) {
        let w = bitcast<f32>(weights[w_base + i]);
        partial += w * x[x_base + i];
    }

    let result = workgroup_reduce(lid.x, partial);
    if (lid.x == 0u) {
        out[token * u.rows + row] = result;
    }
}
