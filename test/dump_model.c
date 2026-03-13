/*
 * dump_model.c -- Diagnostic tool to dump all GGUF metadata and tensors.
 *
 * Compile:
 *   cc -O2 -Wall -Wextra -std=c11 -Iinclude -o dump_model \
 *       test/dump_model.c src/gguf.c src/platform.c -lm
 *
 * Usage:
 *   ./dump_model <model.gguf>
 */

#include "platform.h"
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

static const char *gguf_value_type_name(uint32_t t) {
    switch (t) {
        case BN_GGUF_TYPE_UINT8:   return "UINT8";
        case BN_GGUF_TYPE_INT8:    return "INT8";
        case BN_GGUF_TYPE_UINT16:  return "UINT16";
        case BN_GGUF_TYPE_INT16:   return "INT16";
        case BN_GGUF_TYPE_UINT32:  return "UINT32";
        case BN_GGUF_TYPE_INT32:   return "INT32";
        case BN_GGUF_TYPE_FLOAT32: return "FLOAT32";
        case BN_GGUF_TYPE_BOOL:    return "BOOL";
        case BN_GGUF_TYPE_STRING:  return "STRING";
        case BN_GGUF_TYPE_ARRAY:   return "ARRAY";
        case BN_GGUF_TYPE_UINT64:  return "UINT64";
        case BN_GGUF_TYPE_INT64:   return "INT64";
        case BN_GGUF_TYPE_FLOAT64: return "FLOAT64";
        default:                return "UNKNOWN";
    }
}

static const char *gguf_tensor_type_name(uint32_t t) {
    switch (t) {
        case 0:  return "F32";
        case 1:  return "F16";
        case 2:  return "Q4_0";
        case 3:  return "Q4_1";
        case 6:  return "Q5_0";
        case 7:  return "Q5_1";
        case 8:  return "Q8_0";
        case 9:  return "Q8_1";
        case 10: return "Q2_K";
        case 11: return "Q3_K";
        case 12: return "Q4_K";
        case 13: return "Q5_K";
        case 14: return "Q6_K";
        case 15: return "Q8_K";
        case 16: return "IQ2_XXS";
        case 17: return "IQ2_XS";
        case 18: return "IQ3_XXS";
        case 19: return "IQ1_S";
        case 20: return "IQ4_NL";
        case 21: return "IQ3_S";
        case 22: return "IQ2_S";
        case 23: return "IQ4_XS";
        case 24: return "I8";
        case 25: return "I16";
        case 26: return "I32";
        case 27: return "I64";
        case 28: return "F64";
        case 29: return "IQ1_M";
        case 30: return "BF16";
        case 34: return "TQ1_0";
        case 35: return "TQ2_0";
        default: {
            static char buf[32];
            snprintf(buf, sizeof(buf), "UNKNOWN(%u)", t);
            return buf;
        }
    }
}

/* FP16 -> FP32 (IEEE-754 half) */
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            /* subnormal */
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000 | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

/* Print a KV value (scalars and first few elements of arrays) */
static void print_kv_value(const BnGGUFKeyValue *kv) {
    switch (kv->type) {
        case BN_GGUF_TYPE_UINT8:   printf("%u",   kv->value.u8);  break;
        case BN_GGUF_TYPE_INT8:    printf("%d",   kv->value.i8);  break;
        case BN_GGUF_TYPE_UINT16:  printf("%u",   kv->value.u16); break;
        case BN_GGUF_TYPE_INT16:   printf("%d",   kv->value.i16); break;
        case BN_GGUF_TYPE_UINT32:  printf("%u",   kv->value.u32); break;
        case BN_GGUF_TYPE_INT32:   printf("%d",   kv->value.i32); break;
        case BN_GGUF_TYPE_FLOAT32: printf("%.6g", kv->value.f32); break;
        case BN_GGUF_TYPE_BOOL:    printf("%s",   kv->value.b ? "true" : "false"); break;
        case BN_GGUF_TYPE_STRING:  printf("\"%s\"", kv->value.str.str ? kv->value.str.str : "(null)"); break;
        case BN_GGUF_TYPE_UINT64:  printf("%llu", (unsigned long long)kv->value.u64); break;
        case BN_GGUF_TYPE_INT64:   printf("%lld", (long long)kv->value.i64); break;
        case BN_GGUF_TYPE_FLOAT64: printf("%.10g", kv->value.f64); break;
        case BN_GGUF_TYPE_ARRAY: {
            const BnGGUFArray *a = &kv->value.arr;
            printf("[%s x %llu]", gguf_value_type_name(a->elem_type),
                   (unsigned long long)a->n);
            /* Print first few elements for insight */
            uint64_t show = a->n < 8 ? a->n : 8;
            if (a->elem_type == BN_GGUF_TYPE_STRING && a->strings) {
                printf(" = {");
                for (uint64_t i = 0; i < show; i++) {
                    printf("%s\"%s\"", i ? ", " : "",
                           a->strings[i].str ? a->strings[i].str : "(null)");
                }
                if (a->n > show) printf(", ... +%llu more", (unsigned long long)(a->n - show));
                printf("}");
            } else if (a->data) {
                printf(" = {");
                for (uint64_t i = 0; i < show; i++) {
                    if (i) printf(", ");
                    switch (a->elem_type) {
                        case BN_GGUF_TYPE_UINT32: printf("%u",   ((uint32_t *)a->data)[i]); break;
                        case BN_GGUF_TYPE_INT32:  printf("%d",   ((int32_t *)a->data)[i]);  break;
                        case BN_GGUF_TYPE_FLOAT32:printf("%.6g", ((float *)a->data)[i]);    break;
                        case BN_GGUF_TYPE_UINT8:  printf("%u",   ((uint8_t *)a->data)[i]);  break;
                        case BN_GGUF_TYPE_INT8:   printf("%d",   ((int8_t *)a->data)[i]);   break;
                        case BN_GGUF_TYPE_UINT16: printf("%u",   ((uint16_t *)a->data)[i]); break;
                        case BN_GGUF_TYPE_INT16:  printf("%d",   ((int16_t *)a->data)[i]);  break;
                        case BN_GGUF_TYPE_UINT64: printf("%llu", (unsigned long long)((uint64_t *)a->data)[i]); break;
                        case BN_GGUF_TYPE_INT64:  printf("%lld", (long long)((int64_t *)a->data)[i]); break;
                        case BN_GGUF_TYPE_FLOAT64:printf("%.10g",((double *)a->data)[i]);   break;
                        default: printf("?"); break;
                    }
                }
                if (a->n > show) printf(", ... +%llu more", (unsigned long long)(a->n - show));
                printf("}");
            }
            break;
        }
        default: printf("(unknown type %u)", kv->type); break;
    }
}

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    const char *path = argv[1];
    printf("=== GGUF Model Dump ===\n");
    printf("File: %s\n\n", path);

    /* Load file */
    BnMappedFile mf = bn_platform_load_file(path);
    if (!mf.data) {
        fprintf(stderr, "ERROR: failed to load '%s'\n", path);
        return 1;
    }
    printf("File size: %zu bytes (%.1f MB)\n", mf.size, mf.size / (1024.0 * 1024.0));

    /* Parse GGUF */
    BnGGUFFile *f = bn_gguf_open(mf.data, mf.size);
    if (!f) {
        fprintf(stderr, "ERROR: failed to parse GGUF\n");
        bn_platform_unload_file(&mf);
        return 1;
    }

    printf("GGUF version: %u\n", f->version);
    printf("Alignment:    %zu\n", f->alignment);
    printf("Data offset:  %zu (0x%zx)\n", f->data_offset, f->data_offset);
    printf("KV pairs:     %llu\n", (unsigned long long)f->n_kv);
    printf("Tensors:      %llu\n\n", (unsigned long long)f->n_tensors);

    /* ============================================================== */
    /* 1. Print ALL key-value pairs                                    */
    /* ============================================================== */
    printf("========================================\n");
    printf("KEY-VALUE PAIRS (%llu total)\n", (unsigned long long)f->n_kv);
    printf("========================================\n");
    for (uint64_t i = 0; i < f->n_kv; i++) {
        BnGGUFKeyValue *kv = &f->kvs[i];
        printf("  [%3llu] %-50s  %-8s = ", (unsigned long long)i,
               kv->key ? kv->key : "(null)", gguf_value_type_name(kv->type));
        print_kv_value(kv);
        printf("\n");
    }
    printf("\n");

    /* ============================================================== */
    /* 2. Print ALL tensor info                                        */
    /* ============================================================== */
    printf("========================================\n");
    printf("TENSORS (%llu total)\n", (unsigned long long)f->n_tensors);
    printf("========================================\n");

    /* Track tensor type counts */
    #define MAX_TYPE_ID 64
    int type_counts[MAX_TYPE_ID];
    memset(type_counts, 0, sizeof(type_counts));

    int has_ffn_gate = 0;
    int has_attn_sub_norm = 0;
    int has_output_weight = 0;
    int emb_type = -1;
    int scale_count = 0;

    for (uint64_t i = 0; i < f->n_tensors; i++) {
        BnGGUFTensorInfo *t = &f->tensors[i];
        const char *name = t->name ? t->name : "(null)";
        const char *ttype = gguf_tensor_type_name(t->type);

        /* Accumulate type count */
        if (t->type < MAX_TYPE_ID) type_counts[t->type]++;

        /* Compute element count */
        uint64_t n_elements = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) n_elements *= t->dims[d];

        /* Print tensor info */
        printf("  [%4llu] %-50s  type=%-8s  dims=[", (unsigned long long)i, name, ttype);
        for (uint32_t d = 0; d < t->n_dims; d++) {
            printf("%s%llu", d ? " x " : "", (unsigned long long)t->dims[d]);
        }
        printf("]  n_elem=%llu  offset=%llu\n",
               (unsigned long long)n_elements, (unsigned long long)t->offset);

        /* 4. For scale tensors, print the actual float value */
        if (strstr(name, ".scale")) {
            scale_count++;
            void *data = bn_gguf_tensor_data(f, (int)i);
            if (data) {
                if (t->type == BN_GGUF_TENSOR_F32) {
                    float val = *(float *)data;
                    printf("         ^^ SCALE VALUE (F32): %.8g\n", val);
                } else if (t->type == BN_GGUF_TENSOR_F16) {
                    uint16_t raw = *(uint16_t *)data;
                    float val = f16_to_f32(raw);
                    printf("         ^^ SCALE VALUE (F16): %.8g (raw=0x%04x)\n", val, raw);
                } else {
                    printf("         ^^ SCALE tensor has unexpected type %s\n", ttype);
                }
            } else {
                printf("         ^^ SCALE tensor data pointer is NULL!\n");
            }
        }

        /* Track specific tensors */
        if (strstr(name, "ffn_gate"))       has_ffn_gate = 1;
        if (strstr(name, "attn_sub_norm"))  has_attn_sub_norm = 1;
        if (strcmp(name, "output.weight") == 0) has_output_weight = 1;
        if (strcmp(name, "token_embd.weight") == 0) emb_type = (int)t->type;
    }
    printf("\n");

    /* ============================================================== */
    /* 5. Summary / Diagnostic checks                                  */
    /* ============================================================== */
    printf("========================================\n");
    printf("DIAGNOSTIC SUMMARY\n");
    printf("========================================\n");

    /* Architecture */
    const char *arch = bn_gguf_get_str(f, "general.architecture");
    printf("  Architecture:        %s\n", arch ? arch : "(not set)");

    const char *name_str = bn_gguf_get_str(f, "general.name");
    printf("  Model name:          %s\n", name_str ? name_str : "(not set)");

    /* Tensor type histogram */
    printf("\n  Tensor type counts:\n");
    for (int t = 0; t < MAX_TYPE_ID; t++) {
        if (type_counts[t] > 0) {
            printf("    %-12s : %d\n", gguf_tensor_type_name((uint32_t)t), type_counts[t]);
        }
    }

    printf("\n  ffn_gate tensors:    %s\n", has_ffn_gate ? "YES" : "NO");
    printf("  attn_sub_norm:       %s\n", has_attn_sub_norm ? "YES" : "NO");
    printf("  output.weight:       %s\n", has_output_weight ? "YES (separate output head)" : "NO (likely tied embeddings)");
    printf("  scale tensors:       %d\n", scale_count);

    printf("  token_embd type:     %s\n", emb_type >= 0 ? gguf_tensor_type_name((uint32_t)emb_type) : "(not found!)");

    /* Architecture-specific config readout */
    if (arch) {
        char key[128];
        printf("\n  Config values (prefix='%s'):\n", arch);

        snprintf(key, sizeof(key), "%s.embedding_length", arch);
        int idx = bn_gguf_find_key(f, key);
        if (idx >= 0) printf("    %-45s = %u\n", key, f->kvs[idx].value.u32);
        else          printf("    %-45s = (MISSING)\n", key);

        snprintf(key, sizeof(key), "%s.feed_forward_length", arch);
        idx = bn_gguf_find_key(f, key);
        if (idx >= 0) printf("    %-45s = %u\n", key, f->kvs[idx].value.u32);
        else          printf("    %-45s = (MISSING)\n", key);

        snprintf(key, sizeof(key), "%s.block_count", arch);
        idx = bn_gguf_find_key(f, key);
        if (idx >= 0) printf("    %-45s = %u\n", key, f->kvs[idx].value.u32);
        else          printf("    %-45s = (MISSING)\n", key);

        snprintf(key, sizeof(key), "%s.attention.head_count", arch);
        idx = bn_gguf_find_key(f, key);
        if (idx >= 0) printf("    %-45s = %u\n", key, f->kvs[idx].value.u32);
        else          printf("    %-45s = (MISSING)\n", key);

        snprintf(key, sizeof(key), "%s.attention.head_count_kv", arch);
        idx = bn_gguf_find_key(f, key);
        if (idx >= 0) printf("    %-45s = %u\n", key, f->kvs[idx].value.u32);
        else          printf("    %-45s = (MISSING)\n", key);

        snprintf(key, sizeof(key), "%s.context_length", arch);
        idx = bn_gguf_find_key(f, key);
        if (idx >= 0) printf("    %-45s = %u\n", key, f->kvs[idx].value.u32);
        else          printf("    %-45s = (MISSING)\n", key);

        snprintf(key, sizeof(key), "%s.rope.freq_base", arch);
        idx = bn_gguf_find_key(f, key);
        if (idx >= 0) printf("    %-45s = %.1f\n", key, f->kvs[idx].value.f32);
        else          printf("    %-45s = (MISSING)\n", key);

        snprintf(key, sizeof(key), "%s.attention.layer_norm_rms_epsilon", arch);
        idx = bn_gguf_find_key(f, key);
        if (idx >= 0) printf("    %-45s = %g\n", key, f->kvs[idx].value.f32);
        else          printf("    %-45s = (MISSING)\n", key);

        /* Extra keys that might be present in bitnet models */
        snprintf(key, sizeof(key), "%s.vocab_size", arch);
        idx = bn_gguf_find_key(f, key);
        if (idx >= 0) printf("    %-45s = %u\n", key, f->kvs[idx].value.u32);

        snprintf(key, sizeof(key), "%s.attention.key_length", arch);
        idx = bn_gguf_find_key(f, key);
        if (idx >= 0) printf("    %-45s = %u\n", key, f->kvs[idx].value.u32);

        snprintf(key, sizeof(key), "%s.attention.value_length", arch);
        idx = bn_gguf_find_key(f, key);
        if (idx >= 0) printf("    %-45s = %u\n", key, f->kvs[idx].value.u32);

        snprintf(key, sizeof(key), "%s.expert_count", arch);
        idx = bn_gguf_find_key(f, key);
        if (idx >= 0) printf("    %-45s = %u\n", key, f->kvs[idx].value.u32);

        snprintf(key, sizeof(key), "%s.expert_used_count", arch);
        idx = bn_gguf_find_key(f, key);
        if (idx >= 0) printf("    %-45s = %u\n", key, f->kvs[idx].value.u32);
    }

    /* Tokenizer info */
    printf("\n  Tokenizer:\n");
    const char *tok_model = bn_gguf_get_str(f, "tokenizer.ggml.model");
    printf("    model:             %s\n", tok_model ? tok_model : "(not set)");

    uint64_t vocab_n = bn_gguf_get_arr_n(f, "tokenizer.ggml.tokens");
    printf("    vocab size:        %llu\n", (unsigned long long)vocab_n);

    int bos_idx = bn_gguf_find_key(f, "tokenizer.ggml.bos_token_id");
    if (bos_idx >= 0) printf("    bos_token_id:      %u\n", f->kvs[bos_idx].value.u32);

    int eos_idx = bn_gguf_find_key(f, "tokenizer.ggml.eos_token_id");
    if (eos_idx >= 0) printf("    eos_token_id:      %u\n", f->kvs[eos_idx].value.u32);

    int pad_idx = bn_gguf_find_key(f, "tokenizer.ggml.padding_token_id");
    if (pad_idx >= 0) printf("    padding_token_id:  %u\n", f->kvs[pad_idx].value.u32);

    /* What model_load would compute */
    printf("\n  What model_load() would derive:\n");
    if (arch) {
        char key[128];
        snprintf(key, sizeof(key), "%s.embedding_length", arch);
        int dim = (int)bn_gguf_get_u32(f, key);
        snprintf(key, sizeof(key), "%s.attention.head_count", arch);
        int n_heads = (int)bn_gguf_get_u32(f, key);
        snprintf(key, sizeof(key), "%s.attention.head_count_kv", arch);
        int n_kv_heads = (int)bn_gguf_get_u32(f, key);
        if (n_kv_heads == 0) n_kv_heads = n_heads;

        int head_size = (n_heads > 0) ? dim / n_heads : 0;
        int kv_dim = head_size * n_kv_heads;
        int kv_mul = (n_kv_heads > 0) ? n_heads / n_kv_heads : 0;

        printf("    head_size:         %d\n", head_size);
        printf("    kv_dim:            %d\n", kv_dim);
        printf("    kv_mul:            %d\n", kv_mul);

        int model_has_gate = (bn_gguf_find_tensor(f, "blk.0.ffn_gate.weight") >= 0) ? 1 : 0;
        printf("    has_ffn_gate:      %d\n", model_has_gate);

        int is_bitnet = (strcmp(arch, "bitnet") == 0);
        printf("    act_type:          %d (%s)\n", is_bitnet ? 1 : 0,
               is_bitnet ? "ReLU^2" : "SiLU");
    }

    printf("\n=== Done ===\n");

    bn_gguf_free(f);
    bn_platform_unload_file(&mf);
    return 0;
}
