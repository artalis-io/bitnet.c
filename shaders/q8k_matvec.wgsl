// Q8_K matvec/matmul — 8-bit quantization with 256-element super-blocks
// 292 bytes per block: 4-byte F32 scale + 256 int8 values + 32 bytes bsums (unused here)
// Decode: value = f32_scale * int8(qs[i])

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

fn sign_extend_i8(val: u32) -> i32 {
    return i32(val << 24u) >> 24;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = select(wid.x, wid.x + wid.y * uniforms.extra, uniforms.extra > 0u);
    let token = select(wid.y, 0u, uniforms.extra > 0u);
    let tid = lid.x;

    if (row >= uniforms.rows) {
        return;
    }

    let cols = uniforms.cols;
    let blocks_per_row = cols / 256u;
    // Each block is 292 bytes
    let row_byte_offset = row * blocks_per_row * 292u;
    let x_offset = token * cols;

    var sum = 0.0f;

    var block_idx = tid;
    while (block_idx < blocks_per_row) {
        let block_byte = row_byte_offset + block_idx * 292u;

        // Read F32 scale from first 4 bytes (aligned to block start)
        let scale_u32_idx = block_byte / 4u;
        let scale_byte_off = block_byte % 4u;
        var scale: f32;
        if (scale_byte_off == 0u) {
            scale = bitcast<f32>(weights[scale_u32_idx]);
        } else {
            // Unaligned read: reconstruct u32 from two words
            let lo = weights[scale_u32_idx] >> (scale_byte_off * 8u);
            let hi = weights[scale_u32_idx + 1u] << ((4u - scale_byte_off) * 8u);
            scale = bitcast<f32>(lo | hi);
        }

        // Int8 data starts at block_byte + 4, 256 bytes
        let qs_byte_start = block_byte + 4u;
        let elem_offset = block_idx * 256u;

        // Process 256 int8 values, 4 at a time
        for (var w = 0u; w < 64u; w++) {
            let word_byte_addr = qs_byte_start + w * 4u;
            let word_idx = word_byte_addr / 4u;
            let word_off = word_byte_addr % 4u;

            var word: u32;
            if (word_off == 0u) {
                word = weights[word_idx];
            } else {
                word = (weights[word_idx] >> (word_off * 8u)) | (weights[word_idx + 1u] << ((4u - word_off) * 8u));
            }

            let i0 = sign_extend_i8(word & 0xFFu);
            let i1 = sign_extend_i8((word >> 8u) & 0xFFu);
            let i2 = sign_extend_i8((word >> 16u) & 0xFFu);
            let i3 = sign_extend_i8((word >> 24u) & 0xFFu);

            let base = elem_offset + w * 4u;
            sum += scale * f32(i0) * x[x_offset + base];
            sum += scale * f32(i1) * x[x_offset + base + 1u];
            sum += scale * f32(i2) * x[x_offset + base + 2u];
            sum += scale * f32(i3) * x[x_offset + base + 3u];
        }

        // Skip 32 bytes of bsums (not needed for direct decode)
        block_idx += 256u;
    }

    let result = workgroup_reduce(tid, sum);

    if (tid == 0u) {
        out[token * uniforms.rows + row] = result;
    }
}
