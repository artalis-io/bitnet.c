// SSM decay/update rate computation (per V-head):
//   alpha[h] = exp(softplus(alpha[h] + dt_bias[h]) * A_log[h])
//   beta[h]  = sigmoid(beta[h])
// Dispatch: (1, 1, 1) — small (num_v_heads elements, typically 32)

struct Uniforms {
    p0: u32, p1: u32, p2: u32, p3: u32,
    p4: u32, p5: u32, p6: u32, p7: u32,
}

@group(0) @binding(0) var<storage, read_write> alpha: array<f32>;
@group(0) @binding(1) var<storage, read_write> beta: array<f32>;
@group(0) @binding(2) var<storage, read> dt_bias: array<f32>;
@group(0) @binding(3) var<storage, read> a_log: array<f32>;
@group(0) @binding(4) var<uniform> u: Uniforms;

// p0 = num_v_heads

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let h = lid.x;
    let n = u.p0;

    if (h >= n) {
        return;
    }

    // Softplus: log(1 + exp(x)) for x <= 20, else x
    let dt = alpha[h] + dt_bias[h];
    var dt_sp: f32;
    if (dt > 20.0) {
        dt_sp = dt;
    } else {
        dt_sp = log(1.0 + exp(dt));
    }
    alpha[h] = exp(dt_sp * a_log[h]);

    // Sigmoid
    beta[h] = 1.0 / (1.0 + exp(-beta[h]));
}
