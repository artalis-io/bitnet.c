#include "moe_internal.h"

int bn_moe_checked_mul_size(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > SIZE_MAX / a) return -1;
    *out = a * b;
    return 0;
}

static int moe_load_expert_map_proj(BnGGUFFile *f, const char *name,
                                    int n_experts, int *type_out,
                                    int *rows_out, int *cols_out,
                                    size_t *base_offset_out,
                                    size_t *expert_bytes_out) {
    int ti = bn_gguf_find_tensor(f, name);
    if (ti < 0) return -1;

    BnGGUFTensorInfo *info = &f->tensors[ti];
    if (info->n_dims < 3) {
        SH_LOG_ERROR("Expert tensor must be 3D", "name", name);
        return -1;
    }

    if (info->dims[0] > INT_MAX || info->dims[1] > INT_MAX || info->dims[2] > INT_MAX) {
        SH_LOG_ERROR("Expert tensor dimensions exceed INT_MAX", "name", name);
        return -1;
    }
    int cols = (int)info->dims[0];
    int rows = (int)info->dims[1];
    int n_exp = (int)info->dims[2];
    if (n_exp != n_experts) {
        SH_LOG_ERROR("Expert count mismatch in tensor", "name", name);
        return -1;
    }

    *type_out = (int)info->type;
    *rows_out = rows;
    *cols_out = cols;
    *base_offset_out = f->data_offset + info->offset;

    size_t expert_elements = 0;
    if (bn_moe_checked_mul_size((size_t)rows, (size_t)cols, &expert_elements) != 0 ||
        !bn_gguf_tensor_size(info->type, (uint64_t)expert_elements, expert_bytes_out)) {
        SH_LOG_ERROR("Unsupported expert tensor type", "name", name);
        return -1;
    }

    return 0;
}

static int moe_load_expert_map_gate_up_fused(BnGGUFFile *f, const char *name,
                                             int n_experts, int expert_hidden,
                                             BnMoEExpertMap *em) {
    int ti = bn_gguf_find_tensor(f, name);
    if (ti < 0) return -1;
    BnGGUFTensorInfo *info = &f->tensors[ti];
    if (info->n_dims < 3 || info->dims[0] > INT_MAX ||
        info->dims[1] > INT_MAX || info->dims[2] > INT_MAX)
        return -1;

    int cols = (int)info->dims[0];
    int rows = (int)info->dims[1];
    int n_exp = (int)info->dims[2];
    if (n_exp != n_experts || rows != expert_hidden * 2) {
        SH_LOG_ERROR("Fused gate/up expert tensor shape mismatch", "name", name);
        return -1;
    }

    size_t one_proj_elems = 0, fused_elems = 0;
    size_t one_proj_bytes = 0, fused_bytes = 0;
    if (bn_moe_checked_mul_size((size_t)expert_hidden, (size_t)cols, &one_proj_elems) != 0 ||
        bn_moe_checked_mul_size((size_t)rows, (size_t)cols, &fused_elems) != 0 ||
        !bn_gguf_tensor_size(info->type, (uint64_t)one_proj_elems, &one_proj_bytes) ||
        !bn_gguf_tensor_size(info->type, (uint64_t)fused_elems, &fused_bytes))
        return -1;

    em->gate_type = (int)info->type;
    em->up_type = (int)info->type;
    em->gate_rows = expert_hidden;
    em->up_rows = expert_hidden;
    em->gate_cols = cols;
    em->up_cols = cols;
    em->expert_gate_bytes = one_proj_bytes;
    em->expert_up_bytes = one_proj_bytes;
    em->gate_stride = fused_bytes;
    em->up_stride = fused_bytes;
    em->gate_offset = f->data_offset + info->offset;
    em->up_offset = em->gate_offset + one_proj_bytes;
    return 0;
}

int bn_moe_load_expert_map(BnGGUFFile *f,
                           const BnMoEExpertTensorNames *names,
                           int n_experts,
                           int expert_hidden,
                           BnMoEExpertMap *em) {
    if (!f || !names || !em || n_experts <= 0) return -1;
    memset(em, 0, sizeof(*em));

    if (names->gate && bn_gguf_find_tensor(f, names->gate) >= 0) {
        if (!names->up ||
            moe_load_expert_map_proj(f, names->gate, n_experts,
                &em->gate_type, &em->gate_rows, &em->gate_cols,
                &em->gate_offset, &em->expert_gate_bytes) != 0 ||
            moe_load_expert_map_proj(f, names->up, n_experts,
                &em->up_type, &em->up_rows, &em->up_cols,
                &em->up_offset, &em->expert_up_bytes) != 0)
            return -1;
        em->gate_stride = em->expert_gate_bytes;
        em->up_stride = em->expert_up_bytes;
    } else {
        if (!names->gate_up ||
            moe_load_expert_map_gate_up_fused(f, names->gate_up, n_experts,
                                              expert_hidden, em) != 0)
            return -1;
    }

    if (!names->down ||
        moe_load_expert_map_proj(f, names->down, n_experts,
            &em->down_type, &em->down_rows, &em->down_cols,
            &em->down_offset, &em->expert_down_bytes) != 0)
        return -1;
    em->down_stride = em->expert_down_bytes;
    return 0;
}
