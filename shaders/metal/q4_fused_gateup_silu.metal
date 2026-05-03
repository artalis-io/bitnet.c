#include <metal_stdlib>
using namespace metal;

// Q4_0 NATIVE fused gate-up SiLU — reads GGUF format (18B/block)
// out[i] = silu(gate[i]·x) * up[i]·x
// Gate rows 0..gate_rows-1, up rows gate_rows..total_rows-1 in stacked buffer.
// Pre-scaling trick. 32 rows/tile, 8 threads/row.
// Dispatch: (ceil(gate_rows/32), 1, 1)

kernel void q4_fused_gateup_silu(device const char  *weights [[buffer(0)]],
                                  device const float *x       [[buffer(1)]],
                                  device float       *out     [[buffer(2)]],
                                  constant uint      *p       [[buffer(3)]],
                                  uint3 wid [[threadgroup_position_in_grid]],
                                  uint3 lid [[thread_position_in_threadgroup]]) {
    uint total_rows = p[0], cols = p[1], gate_rows = p[2];
    uint tile_start = wid.x * 32;
    uint row_lane = lid.x & 7;
    uint local_row = lid.x >> 3;
    uint global_row = tile_start + local_row;

    uint nb = cols >> 5;
    float gate_acc = 0.0f, up_acc = 0.0f;

    if (global_row < gate_rows) {
        device const char *gate_data = weights + (size_t)global_row * nb * 18;
        device const char *up_data   = weights + (size_t)(global_row + gate_rows) * nb * 18;

        for (uint b = row_lane; b < nb; b += 8) {
            device const char *gblk = gate_data + (size_t)b * 18;
            device const char *ublk = up_data   + (size_t)b * 18;
            float gd = float(*(device const half *)gblk);
            float ud = float(*(device const half *)ublk);

            device const ushort *gqs0 = (device const ushort *)(gblk + 2);
            device const ushort *gqs1 = (device const ushort *)(gblk + 10);
            device const ushort *uqs0 = (device const ushort *)(ublk + 2);
            device const ushort *uqs1 = (device const ushort *)(ublk + 10);
            uint eb = b * 32;

            float gsumy = 0.f, ga[8] = {0,0,0,0,0,0,0,0};
            float usumy = 0.f, ua[8] = {0,0,0,0,0,0,0,0};

            for (ushort i = 0; i < 8; i += 2) {
                float y0=x[eb+i], y1=x[eb+i+1], y16=x[eb+i+16], y17=x[eb+i+17];
                float y8=x[eb+8+i], y9=x[eb+8+i+1], y24=x[eb+24+i], y25=x[eb+24+i+1];
                float sy = y0+y1+y16+y17+y8+y9+y24+y25;
                gsumy += sy; usumy += sy;

                ushort gw0=gqs0[i/2], gw1=gqs1[i/2];
                ga[0] += y0        *float(gw0&0x000Fu); ga[1] += (y1/256)  *float(gw0&0x0F00u);
                ga[2] += (y16/16)  *float(gw0&0x00F0u); ga[3] += (y17/4096)*float(gw0&0xF000u);
                ga[4] += y8        *float(gw1&0x000Fu); ga[5] += (y9/256)  *float(gw1&0x0F00u);
                ga[6] += (y24/16)  *float(gw1&0x00F0u); ga[7] += (y25/4096)*float(gw1&0xF000u);

                ushort uw0=uqs0[i/2], uw1=uqs1[i/2];
                ua[0] += y0        *float(uw0&0x000Fu); ua[1] += (y1/256)  *float(uw0&0x0F00u);
                ua[2] += (y16/16)  *float(uw0&0x00F0u); ua[3] += (y17/4096)*float(uw0&0xF000u);
                ua[4] += y8        *float(uw1&0x000Fu); ua[5] += (y9/256)  *float(uw1&0x0F00u);
                ua[6] += (y24/16)  *float(uw1&0x00F0u); ua[7] += (y25/4096)*float(uw1&0xF000u);
            }
            gate_acc += gd * (gsumy*(-8.f) + ga[0]+ga[1]+ga[2]+ga[3]+ga[4]+ga[5]+ga[6]+ga[7]);
            up_acc   += ud * (usumy*(-8.f) + ua[0]+ua[1]+ua[2]+ua[3]+ua[4]+ua[5]+ua[6]+ua[7]);
        }
    }

    gate_acc += simd_shuffle_xor(gate_acc, 1);
    gate_acc += simd_shuffle_xor(gate_acc, 2);
    gate_acc += simd_shuffle_xor(gate_acc, 4);
    up_acc += simd_shuffle_xor(up_acc, 1);
    up_acc += simd_shuffle_xor(up_acc, 2);
    up_acc += simd_shuffle_xor(up_acc, 4);

    if (row_lane == 0 && global_row < gate_rows) {
        float g = gate_acc;
        out[global_row] = (g / (1.0f + exp(-g))) * up_acc;
    }
}
