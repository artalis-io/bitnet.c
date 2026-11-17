#ifndef BN_TEST_EXPERT_BANK_REPLAY_H
#define BN_TEST_EXPERT_BANK_REPLAY_H
/* Shared test-only staging controller; actual GGUF payloads, native compute. */
static size_t reserve_expert_banks(BnGPUBackend *g, void *r, int nt, int capacity, int reuse, int grouped) {
    size_t gb, db;
    assert(bn_gguf_tensor_size(BN_GGUF_TENSOR_Q5_K, 2560 * 640, &gb));
    assert(bn_gguf_tensor_size(BN_GGUF_TENSOR_Q8_0, 2560 * 640, &db));
    size_t payload = (2 * gb + db) * (size_t)capacity * 2;
    BnGPUMoEBankConfig c = {2, (unsigned)capacity, 2560, 640, payload, payload,0,0};
    assert(!g->moe_bank_reserve(g->ctx, r, &c));
    if (reuse) assert(!g->moe_bank_input_reserve(g->ctx, r, nt, (size_t)nt * 2560 * 22));
    if (grouped) {
        size_t metadata = (size_t)capacity * 2 * sizeof(BnGPUMoEBankItem);
        assert(!g->moe_bank_grouped_reserve(g->ctx, r, nt, metadata, (size_t)nt * 10 * 640 * 2 + metadata));
    }
    return payload;
}

static int banked_ffn(BnGPUBackend *g, void *r, BnGGUFFile *f, const int *indices,
    const unsigned char **weights, const size_t *weight_bytes, int nt, const int32_t *offsets,
    BnGPUResidentView x, BnGPUResidentView gate, BnGPUResidentView up, BnGPUResidentView out,
    BnGPUResidentView original, BnGPUResidentView slots, BnGPUResidentView status,
    int capacity, int profile, int reuse, int grouped, BnGPUMoEBankStats *stats) {
    size_t payload = reserve_expert_banks(g, r, nt, capacity, reuse, grouped);
    BnGPUMoEBankStats before;
    assert(!g->moe_bank_stats(r, &before));
    if (profile) {
        assert(!g->resident_synchronize(r));
#ifdef BN_CAPTURE_CUDA_PROFILER
        assert(cudaProfilerStart() == cudaSuccess);
#endif
    }
    uint64_t generation = 0;
    if (reuse) {
        BnGPUMoEBankInputPlan input = {nt, original, slots, status};
        assert(!g->moe_bank_prepare_input(g->ctx, r, &input, &generation) && generation);
    }
    int expert = 0, groups = 0;
    unsigned bank = 0;
    while (expert < 512) {
        while (expert < 512 && offsets[expert] == offsets[expert + 1]) expert++;
        if (expert == 512) break;
        BnGPUMoEBankLease lease;
        int result;
        do {
            result = g->moe_bank_acquire(r, bank % 2, (int)f->tensors[indices[0]].type,
                (int)f->tensors[indices[2]].type, &lease);
            if (result == BN_GPU_MOE_BANK_BUSY) thrd_yield();
        } while (result == BN_GPU_MOE_BANK_BUSY);
        assert(!result && lease.gate_bytes == weight_bytes[0] && lease.gate_bytes == weight_bytes[1] &&
            lease.down_bytes == weight_bytes[2] && lease.capacity == (unsigned)capacity);
        BnGPUMoEBankItem items[64];
        int count = 0;
        while (expert < 512 && count < capacity) {
            int first = offsets[expert], ng = offsets[expert + 1] - first;
            if (ng) {
                assert(ng > 0 && ng <= nt);
                items[count] = (BnGPUMoEBankItem){expert, first, ng};
                unsigned char *raw = (unsigned char *)lease.data + (size_t)count * lease.expert_bytes;
                for (int role = 0; role < 3; role++) memcpy(raw + (size_t)role * lease.gate_bytes,
                    weights[role] + (size_t)expert * weight_bytes[role], weight_bytes[role]);
                count++;
            }
            expert++;
        }
        BnGPUMoEBankResidentPlan p = {0};
        p.lease = lease; p.total_tokens = nt; p.n_experts = count; p.experts = items;
        p.x = x; p.gate = gate; p.up = up; p.out = out;
        p.prepared_input = generation;
        p.grouped = grouped;
        assert(!g->moe_bank_submit(g->ctx, r, &p));
        groups += count; bank++;
    }
    if (profile) {
        assert(!g->resident_synchronize(r));
#ifdef BN_CAPTURE_CUDA_PROFILER
        assert(cudaProfilerStop() == cudaSuccess);
#endif
    }
    assert(!g->moe_bank_stats(r, stats));
    assert(stats->input_preparations - before.input_preparations == (size_t)reuse);
    assert(stats->grouped_banks - before.grouped_banks == (grouped ? bank : 0));
    assert(stats->metadata_upload_bytes - before.metadata_upload_bytes == (grouped ? (size_t)groups * sizeof(BnGPUMoEBankItem) : 0));
    assert(stats->reserved_host_bytes == payload && stats->reserved_device_bytes == payload &&
        stats->submitted_banks - before.submitted_banks == bank && stats->submitted_experts - before.submitted_experts == (size_t)groups &&
        stats->upload_bytes - before.upload_bytes == (size_t)groups * (2 * weight_bytes[0] + weight_bytes[2]));
    return groups;
}

#endif
