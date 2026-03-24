// I2_S matvec/matmul — 2-bit interleaved ternary quantization
// 128-element chunks, 32 bytes per chunk
// Per-tensor scale stored as bitcast<f32>(uniforms.extra)

struct Uniforms {
    rows: u32,
    cols: u32,
    n_tokens: u32,
    extra: u32,  // per-tensor scale bits (bitcast to f32)
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
    // I2_S: per-tensor FP32 scale stored at end of weight data (offset = rows * cols / 4 bytes = rows * cols / 16 u32s)
    let scale_offset = uniforms.rows * cols / 16u;
    let scale = bitcast<f32>(weights[scale_offset]);


    // I2_S: 128 elements per chunk, 32 bytes per chunk
    let chunks_per_row = cols / 128u;
    // 32 bytes per chunk = 8 u32s per chunk
    let u32s_per_chunk = 8u;
    let row_offset_u32 = row * chunks_per_row * u32s_per_chunk;

    let x_offset = token * cols;

    var sum = 0.0f;

    // Each thread processes cols/256 elements worth of chunks
    // Since each chunk has 128 elements, we process chunks_per_row/2 chunks per thread
    // But more precisely: distribute chunks across 256 threads
    var chunk_idx = tid;
    while (chunk_idx < chunks_per_row) {
        let chunk_u32_offset = row_offset_u32 + chunk_idx * u32s_per_chunk;
        let elem_offset = chunk_idx * 128u;

        // Process 128 elements from 32 bytes (8 u32s)
        for (var w = 0u; w < 8u; w++) {
            let word = weights[chunk_u32_offset + w];
            // Each u32 has 4 bytes, each byte has 4 values = 16 values per u32
            for (var b = 0u; b < 4u; b++) {
                let byte_val = (word >> (b * 8u)) & 0xFFu;
                let base_idx = elem_offset + w * 16u + b * 4u;

                // Decode 4 values from byte: bits [6:7], [4:5], [2:3], [0:1]
                let v0 = (byte_val >> 0u) & 3u;
                let v1 = (byte_val >> 2u) & 3u;
                let v2 = (byte_val >> 4u) & 3u;
                let v3 = (byte_val >> 6u) & 3u;

                // Map: 0→-1, 1→0, 2→+1, 3→0
                let d0 = select(f32(i32(v0) - 1), 0.0, v0 == 3u);
                let d1 = select(f32(i32(v1) - 1), 0.0, v1 == 3u);
                let d2 = select(f32(i32(v2) - 1), 0.0, v2 == 3u);
                let d3 = select(f32(i32(v3) - 1), 0.0, v3 == 3u);

                sum += d0 * x[x_offset + base_idx + 0u];
                sum += d1 * x[x_offset + base_idx + 1u];
                sum += d2 * x[x_offset + base_idx + 2u];
                sum += d3 * x[x_offset + base_idx + 3u];
            }
        }

        chunk_idx += 256u;
    }

    let result = workgroup_reduce(tid, sum);

    if (tid == 0u) {
        out[token * uniforms.rows + row] = result * scale;
    }
}
