#include "model_internal.h"
#include "moe.h"

int bn_model_load_moe_expert_map(BnGGUFFile *file,
                                 const BnMoEExpertTensorNames *names,
                                 int n_experts,
                                 int expert_hidden,
                                 BnMoEExpertMap *map) {
    return bn_moe_load_expert_map(file, names, n_experts, expert_hidden, map);
}

void bn_model_moe_io_shutdown(BnModel *model) {
    if (model && model->io)
        bn_moe_prefetch_destroy(&model->io->moe_io);
}
