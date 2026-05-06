// SSM decay/update rate computation from stacked [alpha | beta] matvec output.

struct Uniforms {
    p0: u32, p1: u32, p2: u32, p3: u32,
    p4: u32, p5: u32, p6: u32, p7: u32,
}

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> alpha: array<f32>;
@group(0) @binding(2) var<storage, read_write> beta: array<f32>;
@group(0) @binding(3) var<storage, read> dt_bias: array<f32>;
@group(0) @binding(4) var<storage, read> a_log: array<f32>;
@group(0) @binding(5) var<uniform> u: Uniforms;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let h = lid.x;
    let n = u.p0;
    let beta_off = u.p1;
    if (h >= n) {
        return;
    }

    let dt = src[h] + dt_bias[h];
    let dt_sp = select(log(1.0 + exp(dt)), dt, dt > 20.0);
    alpha[h] = exp(dt_sp * a_log[h]);
    let b = src[beta_off + h];
    beta[h] = 1.0 / (1.0 + exp(-b));
}
