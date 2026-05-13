#include "gpu_graph_ir.h"
#include "../src/gpu_graph_lowering_internal.h"
#include <assert.h>
#include <stdio.h>

static void test_graph_values_and_aliases(void) {
    printf("test_graph_values_and_aliases... ");

    BnGPUValueGraph graph;
    bn_gpu_value_graph_init(&graph);

    int x = bn_gpu_value_graph_add_value(
        &graph, BN_GPU_IR_VALUE_MODEL_INPUT, 0, 1, 128,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_EXTERNAL, "x");
    int xb = bn_gpu_value_graph_add_value(
        &graph, BN_GPU_IR_VALUE_TRANSIENT, 0, 1, 128,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_WRITABLE, "xb");
    int xb_inplace = bn_gpu_value_graph_add_alias(&graph, xb, "xb.inplace");

    assert(x == 0);
    assert(xb == 1);
    assert(xb_inplace == 2);
    assert(graph.n_values == 3);
    assert(graph.values[xb_inplace].alias_of == xb);
    assert(graph.values[xb_inplace].flags & BN_GPU_IR_VALUE_ALIAS);

    bn_gpu_value_graph_free(&graph);
    printf("PASSED\n");
}

static void test_graph_multi_output_op_and_fallback(void) {
    printf("test_graph_multi_output_op_and_fallback... ");

    BnGPUValueGraph graph;
    bn_gpu_value_graph_init(&graph);

    int x = bn_gpu_value_graph_add_value(
        &graph, BN_GPU_IR_VALUE_MODEL_INPUT, 0, 1, 128,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_EXTERNAL, "x");
    int gate = bn_gpu_value_graph_add_value(
        &graph, BN_GPU_IR_VALUE_TRANSIENT, 0, 1, 256,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_WRITABLE, "gate");
    int up = bn_gpu_value_graph_add_value(
        &graph, BN_GPU_IR_VALUE_TRANSIENT, 0, 1, 256,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_WRITABLE, "up");

    BnGPUIROp *op = bn_gpu_value_graph_add_op(
        &graph, BN_GPU_IR_OP_FFN, "fused-gate-up");
    assert(op != NULL);
    assert(bn_gpu_ir_op_add_input(op, x) == 0);
    assert(bn_gpu_ir_op_add_output(op, gate) == 0);
    assert(bn_gpu_ir_op_add_output(op, up) == 0);
    op->rows = 512;
    op->cols = 128;
    bn_gpu_ir_op_set_fallback(
        op, BN_GPU_IR_FALLBACK_UNSUPPORTED_QUANT, "qtype lacks split kernel");

    assert(graph.n_ops == 1);
    assert(op->kind == BN_GPU_IR_OP_FFN);
    assert(op->n_inputs == 1);
    assert(op->n_outputs == 2);
    assert(op->inputs[0] == x);
    assert(op->outputs[0] == gate);
    assert(op->outputs[1] == up);
    assert(op->fallback.reason == BN_GPU_IR_FALLBACK_UNSUPPORTED_QUANT);

    bn_gpu_value_graph_clear(&graph);
    assert(graph.n_values == 0);
    assert(graph.n_ops == 0);
    assert(graph.cap_values >= 3);
    assert(graph.cap_ops >= 1);

    bn_gpu_value_graph_free(&graph);
    printf("PASSED\n");
}

static void test_graph_simple_op_builders(void) {
    printf("test_graph_simple_op_builders... ");

    BnGPUValueGraph graph;
    bn_gpu_value_graph_init(&graph);

    int x = bn_gpu_value_graph_add_value(
        &graph, BN_GPU_IR_VALUE_MODEL_INPUT, 0, 1, 128,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_EXTERNAL, "x");
    int norm = bn_gpu_value_graph_add_value(
        &graph, BN_GPU_IR_VALUE_WEIGHT, 0, 1, 128,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_EXTERNAL, "norm");
    int residual = bn_gpu_value_graph_add_value(
        &graph, BN_GPU_IR_VALUE_TRANSIENT, 0, 1, 128,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_WRITABLE, "residual");
    int logits_w = bn_gpu_value_graph_add_value(
        &graph, BN_GPU_IR_VALUE_WEIGHT, 0, 32000, 128,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_EXTERNAL, "logits.w");

    int xb = bn_gpu_value_graph_add_rmsnorm(
        &graph, x, norm, 128, 0x3a83126f, "rmsnorm");
    int copy = bn_gpu_value_graph_add_copy_region(&graph, xb, 4, 8, 64,
                                                  "copy");
    int added = bn_gpu_value_graph_add_residual_add(
        &graph, copy, residual, "residual.add");
    int activated = bn_gpu_value_graph_add_activation(
        &graph, added, BN_GPU_IR_INVALID_VALUE, BN_GPU_IR_ACTIVATION_SILU, 1,
        "silu");
    int logits = bn_gpu_value_graph_add_logits(
        &graph, activated, logits_w, 32000, 128, "logits");

    assert(xb >= 0);
    assert(copy >= 0);
    assert(added >= 0);
    assert(activated >= 0);
    assert(logits >= 0);
    assert(graph.n_ops == 5);

    assert(graph.ops[0].kind == BN_GPU_IR_OP_RMSNORM);
    assert(graph.ops[0].inputs[0] == x);
    assert(graph.ops[0].inputs[1] == norm);
    assert(graph.ops[0].outputs[0] == xb);
    assert(graph.ops[0].rows == 128);
    assert(graph.ops[0].aux0 == (int)0x3a83126f);

    assert(graph.ops[1].kind == BN_GPU_IR_OP_COPY);
    assert(graph.ops[1].inputs[0] == xb);
    assert(graph.ops[1].outputs[0] == copy);
    assert(graph.ops[1].rows == 64);
    assert(graph.ops[1].aux0 == 4);
    assert(graph.ops[1].aux1 == 8);

    assert(graph.ops[2].kind == BN_GPU_IR_OP_RESIDUAL_ADD);
    assert(graph.ops[2].inputs[0] == copy);
    assert(graph.ops[2].inputs[1] == residual);
    assert(graph.values[added].alias_of == copy);
    assert(graph.values[added].flags & BN_GPU_IR_VALUE_ALIAS);

    assert(graph.ops[3].kind == BN_GPU_IR_OP_ACTIVATION);
    assert(graph.ops[3].inputs[0] == added);
    assert(graph.ops[3].outputs[0] == activated);
    assert(graph.ops[3].aux0 == BN_GPU_IR_ACTIVATION_SILU);
    assert(graph.ops[3].flags == 1);
    assert(graph.values[activated].alias_of == added);
    assert(graph.values[activated].flags & BN_GPU_IR_VALUE_ALIAS);

    assert(graph.ops[4].kind == BN_GPU_IR_OP_LOGITS);
    assert(graph.ops[4].inputs[0] == activated);
    assert(graph.ops[4].inputs[1] == logits_w);
    assert(graph.ops[4].outputs[0] == logits);
    assert(graph.values[logits].kind == BN_GPU_IR_VALUE_MODEL_OUTPUT);
    assert(graph.values[logits].cols == 32000);

    bn_gpu_value_graph_free(&graph);
    printf("PASSED\n");
}

static void test_graph_builder_invalid_inputs(void) {
    printf("test_graph_builder_invalid_inputs... ");

    BnGPUValueGraph graph;
    bn_gpu_value_graph_init(&graph);

    int x = bn_gpu_value_graph_add_value(
        &graph, BN_GPU_IR_VALUE_MODEL_INPUT, 0, 1, 128,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_EXTERNAL, "x");
    assert(x == 0);

    int bad = bn_gpu_value_graph_add_rmsnorm(
        &graph, x, 42, 128, 0, "bad.rmsnorm");
    assert(bad == BN_GPU_IR_INVALID_VALUE);
    assert(graph.n_ops == 0);

    bad = bn_gpu_value_graph_add_activation(
        &graph, x, 99, BN_GPU_IR_ACTIVATION_SIGMOID, 0, "bad.activation");
    assert(bad == BN_GPU_IR_INVALID_VALUE);
    assert(graph.n_ops == 0);

    bad = bn_gpu_value_graph_add_logits(
        &graph, x, 99, 32000, 128, "bad.logits");
    assert(bad == BN_GPU_IR_INVALID_VALUE);
    assert(graph.n_ops == 0);

    bad = bn_gpu_value_graph_add_matvec(
        &graph, x, 99, 64, 128, 1, 0, "bad.matvec");
    assert(bad == BN_GPU_IR_INVALID_VALUE);
    assert(graph.n_ops == 0);

    bad = bn_gpu_value_graph_add_fused_gateup(
        &graph, x, 99, 64, 64, 128, BN_GPU_IR_ACTIVATION_SILU,
        "bad.fused");
    assert(bad == BN_GPU_IR_INVALID_VALUE);
    assert(graph.n_ops == 0);

    bn_gpu_value_graph_free(&graph);
    printf("PASSED\n");
}

static void test_graph_lower_simple_ops_to_shader(void) {
    printf("test_graph_lower_simple_ops_to_shader... ");

    BnGPUValueGraph graph;
    bn_gpu_value_graph_init(&graph);

    int x = bn_gpu_value_graph_add_value(
        &graph, BN_GPU_IR_VALUE_MODEL_INPUT, 0, 1, 128,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_EXTERNAL, "x");
    int norm = bn_gpu_value_graph_add_value(
        &graph, BN_GPU_IR_VALUE_WEIGHT, 0, 1, 128,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_EXTERNAL, "norm");
    int residual = bn_gpu_value_graph_add_value(
        &graph, BN_GPU_IR_VALUE_TRANSIENT, 0, 1, 128,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_WRITABLE, "residual");
    int logits_w = bn_gpu_value_graph_add_value(
        &graph, BN_GPU_IR_VALUE_WEIGHT, 42, 32000, 128,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_EXTERNAL, "logits.w");
    int matvec_w = bn_gpu_value_graph_add_value(
        &graph, BN_GPU_IR_VALUE_WEIGHT, 43, 256, 128,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_EXTERNAL, "matvec.w");
    int fused_w = bn_gpu_value_graph_add_value(
        &graph, BN_GPU_IR_VALUE_WEIGHT, 44, 512, 128,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_EXTERNAL, "fused.w");

    int xb = bn_gpu_value_graph_add_rmsnorm(
        &graph, x, norm, 128, 7, "rmsnorm");
    int copy = bn_gpu_value_graph_add_copy_region(&graph, xb, 4, 8, 64,
                                                  "copy");
    int added = bn_gpu_value_graph_add_residual_add(
        &graph, copy, residual, "residual.add");
    int activated = bn_gpu_value_graph_add_activation(
        &graph, added, BN_GPU_IR_INVALID_VALUE, BN_GPU_IR_ACTIVATION_RELU2, 1,
        "relu2");
    graph.ops[graph.n_ops - 1].aux1 = 32;
    int matvec = bn_gpu_value_graph_add_matvec(
        &graph, activated, matvec_w, 256, 128, 1, 12, "matvec");
    int fused = bn_gpu_value_graph_add_fused_gateup(
        &graph, activated, fused_w, 256, 256, 128,
        BN_GPU_IR_ACTIVATION_SILU, "fused");
    int logits = bn_gpu_value_graph_add_logits(
        &graph, activated, logits_w, 32000, 128, "logits");

    assert(logits >= 0);
    BnGPUIRLoweringValue values[20] = {0};
    for (int i = 0; i < 20; i++)
        values[i].shader_slot = BN_GPU_IR_NO_SHADER_SLOT;
    values[x].shader_slot = BN_GPU_VALUE_X;
    values[norm].weight_buf = (void *)0x1000;
    values[residual].shader_slot = BN_GPU_VALUE_XB2;
    values[logits_w].weight_buf = (void *)0x2000;
    values[logits_w].tensor_type = 42;
    values[matvec_w].weight_buf = (void *)0x3000;
    values[matvec_w].tensor_type = 43;
    values[fused_w].weight_buf = (void *)0x4000;
    values[fused_w].tensor_type = 44;
    values[xb].shader_slot = BN_GPU_VALUE_XB;
    values[copy].shader_slot = BN_GPU_VALUE_HB;
    values[added].shader_slot = BN_GPU_VALUE_HB;
    values[activated].shader_slot = BN_GPU_VALUE_HB;
    values[matvec].shader_slot = BN_GPU_VALUE_XB2;
    values[fused].shader_slot = BN_GPU_VALUE_HB2;
    values[logits].shader_slot = BN_GPU_VALUE_LOGITS;
    BnGPUIRLoweringMap map = { values, 20 };

    BnGPUOp ops[8];
    int n_ops = -1;
    assert(bn_gpu_value_graph_lower_to_shader(&graph, &map, ops, 8,
                                              &n_ops) == 0);
    assert(n_ops == 7);

    assert(ops[0].op_kind == BN_GPU_OP_RMSNORM);
    assert(ops[0].op_code == BN_GPU_CODE_RMSNORM);
    assert(ops[0].W_buf == (void *)0x1000);
    assert(ops[0].buf_in == BN_GPU_VALUE_X);
    assert(ops[0].buf_out == BN_GPU_VALUE_XB);
    assert(ops[0].p[0] == 128);
    assert(ops[0].p[1] == 7);

    assert(ops[1].op_kind == BN_GPU_OP_COPY);
    assert(ops[1].op_code == BN_GPU_CODE_COPY);
    assert(ops[1].buf_in == BN_GPU_VALUE_XB);
    assert(ops[1].buf_out == BN_GPU_VALUE_HB);
    assert(ops[1].p[0] == 4);
    assert(ops[1].p[1] == 8);
    assert(ops[1].p[2] == 64);

    assert(ops[2].op_kind == BN_GPU_OP_RESIDUAL);
    assert(ops[2].op_code == BN_GPU_CODE_RESIDUAL_ADD);
    assert(ops[2].buf_in == BN_GPU_VALUE_HB);
    assert(ops[2].buf_out == -1);
    assert(ops[2].buf_aux == BN_GPU_VALUE_XB2);
    assert(ops[2].p[0] == 128);

    assert(ops[3].op_kind == BN_GPU_OP_ACTIVATION);
    assert(ops[3].op_code == BN_GPU_CODE_RELU2_ACT);
    assert(ops[3].buf_in == BN_GPU_VALUE_HB);
    assert(ops[3].buf_out == -1);
    assert(ops[3].buf_aux == -1);
    assert(ops[3].p[0] == 128);
    assert(ops[3].p[1] == 32);

    assert(ops[4].op_kind == BN_GPU_OP_MATVEC);
    assert(ops[4].op_code == BN_GPU_CODE_MATVEC);
    assert(ops[4].type == 43);
    assert(ops[4].W_buf == (void *)0x3000);
    assert(ops[4].buf_in == BN_GPU_VALUE_HB);
    assert(ops[4].buf_out == BN_GPU_VALUE_XB2);
    assert(ops[4].rows == 256);
    assert(ops[4].cols == 128);
    assert(ops[4].p[0] == 256);
    assert(ops[4].p[1] == 128);
    assert(ops[4].p[2] == 1);
    assert(ops[4].p[5] == 12);

    assert(ops[5].op_kind == BN_GPU_OP_FFN);
    assert(ops[5].op_code == BN_GPU_CODE_FUSED_GATEUP_SILU);
    assert(ops[5].type == 44);
    assert(ops[5].W_buf == (void *)0x4000);
    assert(ops[5].buf_in == BN_GPU_VALUE_HB);
    assert(ops[5].buf_out == BN_GPU_VALUE_HB2);
    assert(ops[5].rows == 256);
    assert(ops[5].cols == 128);
    assert(ops[5].p[0] == 512);
    assert(ops[5].p[1] == 128);
    assert(ops[5].p[2] == 256);

    assert(ops[6].op_kind == BN_GPU_OP_LOGITS);
    assert(ops[6].op_code == BN_GPU_CODE_MATVEC);
    assert(ops[6].type == 42);
    assert(ops[6].W_buf == (void *)0x2000);
    assert(ops[6].buf_in == BN_GPU_VALUE_HB);
    assert(ops[6].buf_out == BN_GPU_VALUE_LOGITS);
    assert(ops[6].rows == 32000);
    assert(ops[6].cols == 128);
    assert(ops[6].p[0] == 32000);
    assert(ops[6].p[1] == 128);
    assert(ops[6].p[2] == 1);

    bn_gpu_value_graph_free(&graph);
    printf("PASSED\n");
}

static void test_graph_lowering_failures(void) {
    printf("test_graph_lowering_failures... ");

    BnGPUValueGraph graph;
    bn_gpu_value_graph_init(&graph);

    int x = bn_gpu_value_graph_add_value(
        &graph, BN_GPU_IR_VALUE_MODEL_INPUT, 0, 1, 128,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_EXTERNAL, "x");
    int copy = bn_gpu_value_graph_add_copy(&graph, x, "copy");
    assert(copy >= 0);

    BnGPUIRLoweringValue values[2] = {
        { .shader_slot = BN_GPU_VALUE_X },
        { .shader_slot = BN_GPU_VALUE_XB },
    };
    BnGPUIRLoweringMap map = { values, 2 };
    BnGPUOp ops[1];
    int n_ops = 0;

    assert(bn_gpu_value_graph_lower_to_shader(&graph, &map, ops, 0,
                                              &n_ops) == -1);

    graph.ops[0].kind = BN_GPU_IR_OP_SSM;
    assert(bn_gpu_value_graph_lower_to_shader(&graph, &map, ops, 1,
                                              &n_ops) == -1);

    graph.ops[0].kind = BN_GPU_IR_OP_COPY;
    values[copy].shader_slot = BN_GPU_IR_NO_SHADER_SLOT;
    assert(bn_gpu_value_graph_lower_to_shader(&graph, &map, ops, 1,
                                              &n_ops) == -1);

    bn_gpu_value_graph_free(&graph);
    printf("PASSED\n");
}

static void test_graph_lower_split_matvec_to_shader(void) {
    printf("test_graph_lower_split_matvec_to_shader... ");

    BnGPUValueGraph graph;
    bn_gpu_value_graph_init(&graph);

    int x = bn_gpu_value_graph_add_value(
        &graph, BN_GPU_IR_VALUE_MODEL_INPUT, 0, 1, 128,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_EXTERNAL, "x");
    int w = bn_gpu_value_graph_add_value(
        &graph, BN_GPU_IR_VALUE_WEIGHT, BN_GGUF_TENSOR_Q5_K, 384, 128,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_EXTERNAL, "w");
    BnGPUIROp *op = bn_gpu_value_graph_add_matvec_split(
        &graph, x, w, 384, 128, 128, 256, 128, 128, 128, 64, 96,
        "q", "k", "v");
    assert(op);
    assert(op->n_outputs == 3);

    BnGPUIRLoweringValue values[5] = {0};
    for (int i = 0; i < 5; i++)
        values[i].shader_slot = BN_GPU_IR_NO_SHADER_SLOT;
    values[x].shader_slot = BN_GPU_VALUE_XB;
    values[w].weight_buf = (void *)0x5000;
    values[w].tensor_type = BN_GGUF_TENSOR_Q5_K;
    values[op->outputs[0]].shader_slot = BN_GPU_VALUE_Q;
    values[op->outputs[1]].shader_slot = BN_GPU_VALUE_KEY_CACHE;
    values[op->outputs[2]].shader_slot = BN_GPU_VALUE_VALUE_CACHE;
    BnGPUIRLoweringMap map = { values, 5 };

    BnGPUOp ops[1];
    int n_ops = -1;
    assert(bn_gpu_value_graph_lower_to_shader(&graph, &map, ops, 1,
                                              &n_ops) == 0);
    assert(n_ops == 1);
    assert(ops[0].op_kind == BN_GPU_OP_MATVEC);
    assert(ops[0].op_code == BN_GPU_CODE_Q5K_MATVEC_SPLIT);
    assert(ops[0].type == BN_GGUF_TENSOR_Q5_K);
    assert(ops[0].W_buf == (void *)0x5000);
    assert(ops[0].buf_in == BN_GPU_VALUE_XB);
    assert(ops[0].buf_out == BN_GPU_VALUE_Q);
    assert(ops[0].buf_aux == BN_GPU_VALUE_KEY_CACHE);
    assert(ops[0].rows == BN_GPU_VALUE_VALUE_CACHE);
    assert(ops[0].cols == 128);
    assert(ops[0].p[0] == 384);
    assert(ops[0].p[1] == 128);
    assert(ops[0].p[2] == 128);
    assert(ops[0].p[3] == 256);
    assert(ops[0].p[6] == 64);
    assert(ops[0].p[7] == 96);

    bn_gpu_value_graph_free(&graph);
    printf("PASSED\n");
}

static void test_graph_lower_attention_to_shader(void) {
    printf("test_graph_lower_attention_to_shader... ");

    BnGPUValueGraph graph;
    bn_gpu_value_graph_init(&graph);

    int q = bn_gpu_value_graph_add_value(
        &graph, BN_GPU_IR_VALUE_TRANSIENT, 0, 1, 128,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_WRITABLE, "q");
    int k = bn_gpu_value_graph_add_value(
        &graph, BN_GPU_IR_VALUE_KV_CACHE, 0, 1, 128,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_WRITABLE, "k");
    int rope = bn_gpu_value_graph_add_rope(
        &graph, q, k, 8, 16, 5, 16, 4, 32, "rope");
    int flash = bn_gpu_value_graph_add_flash_attention(
        &graph, q, 8, 16, 6, 2, 64, 1024, 128, 0x3f000000u,
        "flash");
    int scores = bn_gpu_value_graph_add_attention_scores(
        &graph, q, 8, 16, 6, 2, 64, 1024, 128, 0x3f000000u,
        "scores");
    int probs = bn_gpu_value_graph_add_softmax(
        &graph, scores, 8, 6, 1024, "softmax");
    int combine = bn_gpu_value_graph_add_attention_combine(
        &graph, probs, 8, 16, 6, 2, 64, 1024, 128, "combine");
    assert(rope >= 0);
    assert(flash >= 0);
    assert(combine >= 0);

    BnGPUIRLoweringValue values[8] = {0};
    for (int i = 0; i < 8; i++)
        values[i].shader_slot = BN_GPU_IR_NO_SHADER_SLOT;
    values[q].shader_slot = BN_GPU_VALUE_Q;
    values[k].shader_slot = BN_GPU_VALUE_KEY_CACHE;
    values[rope].shader_slot = BN_GPU_VALUE_Q;
    values[flash].shader_slot = BN_GPU_VALUE_XB;
    values[scores].shader_slot = BN_GPU_VALUE_ATT;
    values[probs].shader_slot = BN_GPU_VALUE_ATT;
    values[combine].shader_slot = BN_GPU_VALUE_XB;
    BnGPUIRLoweringMap map = { values, 8 };

    BnGPUOp ops[5];
    int n_ops = -1;
    assert(bn_gpu_value_graph_lower_to_shader(&graph, &map, ops, 5,
                                              &n_ops) == 0);
    assert(n_ops == 5);

    assert(ops[0].op_kind == BN_GPU_OP_ROPE);
    assert(ops[0].op_code == BN_GPU_CODE_ROPE_QK);
    assert(ops[0].buf_in == BN_GPU_VALUE_Q);
    assert(ops[0].buf_aux == BN_GPU_VALUE_KEY_CACHE);
    assert(ops[0].p[0] == 8);
    assert(ops[0].p[1] == 16);
    assert(ops[0].p[2] == 5);
    assert(ops[0].p[3] == 16);
    assert(ops[0].p[4] == 4);
    assert(ops[0].p[5] == 32);

    assert(ops[1].op_kind == BN_GPU_OP_ATTENTION);
    assert(ops[1].op_code == BN_GPU_CODE_FLASH_ATTN);
    assert(ops[1].buf_in == BN_GPU_VALUE_Q);
    assert(ops[1].buf_out == BN_GPU_VALUE_XB);
    assert(ops[1].p[0] == 8);
    assert(ops[1].p[1] == 16);
    assert(ops[1].p[7] == 0x3f000000u);

    assert(ops[2].op_kind == BN_GPU_OP_ATTENTION);
    assert(ops[2].op_code == BN_GPU_CODE_GQA_SCORES);
    assert(ops[2].buf_in == BN_GPU_VALUE_Q);
    assert(ops[2].p[2] == 6);
    assert(ops[2].p[6] == 128);

    assert(ops[3].op_kind == BN_GPU_OP_ATTENTION);
    assert(ops[3].op_code == BN_GPU_CODE_SOFTMAX);
    assert(ops[3].buf_in == -1);
    assert(ops[3].p[0] == 8);
    assert(ops[3].p[1] == 6);
    assert(ops[3].p[2] == 1024);

    assert(ops[4].op_kind == BN_GPU_OP_ATTENTION);
    assert(ops[4].op_code == BN_GPU_CODE_GQA_COMBINE);
    assert(ops[4].buf_in == -1);
    assert(ops[4].buf_out == BN_GPU_VALUE_XB);
    assert(ops[4].p[0] == 8);
    assert(ops[4].p[1] == 16);
    assert(ops[4].p[6] == 128);
    assert(ops[4].p[7] == 0);

    bn_gpu_value_graph_free(&graph);
    printf("PASSED\n");
}

static void test_graph_lower_ssm_to_shader(void) {
    printf("test_graph_lower_ssm_to_shader... ");

    BnGPUValueGraph graph;
    bn_gpu_value_graph_init(&graph);

    int qkv = bn_gpu_value_graph_add_value(
        &graph, BN_GPU_IR_VALUE_TRANSIENT, 0, 1, 256,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_WRITABLE, "qkv");
    int z = bn_gpu_value_graph_add_value(
        &graph, BN_GPU_IR_VALUE_TRANSIENT, 0, 1, 128,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_WRITABLE, "z");
    int conv_w = bn_gpu_value_graph_add_value(
        &graph, BN_GPU_IR_VALUE_WEIGHT, 0, 1, 0,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_EXTERNAL, "conv.w");
    int gate_w = bn_gpu_value_graph_add_value(
        &graph, BN_GPU_IR_VALUE_WEIGHT, 0, 1, 0,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_EXTERNAL, "gate.w");
    int out = bn_gpu_value_graph_add_value(
        &graph, BN_GPU_IR_VALUE_TRANSIENT, 0, 1, 128,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_WRITABLE, "out");

    uint32_t conv_params[8] = { 256, 4, 32, 768, 0, 0, 0, 0 };
    int conv = bn_gpu_value_graph_add_ssm(
        &graph, qkv, BN_GPU_IR_INVALID_VALUE, conv_w,
        BN_GPU_IR_INVALID_VALUE, BN_GPU_IR_SSM_CONV_SILU, 0, conv_params,
        "conv");
    uint32_t delta_params[8] = { 16, 32, 4, 0x3f000000u, 64, 128, 0, 256 };
    int delta = bn_gpu_value_graph_add_ssm(
        &graph, qkv, qkv, BN_GPU_IR_INVALID_VALUE, out,
        BN_GPU_IR_SSM_DELTA, 3, delta_params, "delta");
    uint32_t gate_params[8] = { 32, 7, 0, 0, 0, 0, 0, 0 };
    int gate = bn_gpu_value_graph_add_ssm(
        &graph, out, z, gate_w, BN_GPU_IR_INVALID_VALUE,
        BN_GPU_IR_SSM_GATE, 3, gate_params, "gate");
    assert(conv >= 0);
    assert(delta == out);
    assert(gate >= 0);

    BnGPUIRLoweringValue values[8] = {0};
    for (int i = 0; i < 8; i++)
        values[i].shader_slot = BN_GPU_IR_NO_SHADER_SLOT;
    values[qkv].shader_slot = BN_GPU_VALUE_SSM_QKV;
    values[z].shader_slot = BN_GPU_VALUE_SSM_Z;
    values[conv_w].weight_buf = (void *)0x6000;
    values[gate_w].weight_buf = (void *)0x7000;
    values[out].shader_slot = BN_GPU_VALUE_XB2;
    values[conv].shader_slot = BN_GPU_VALUE_SSM_QKV;
    values[gate].shader_slot = BN_GPU_VALUE_XB2;
    BnGPUIRLoweringMap map = { values, 8 };

    BnGPUOp ops[3];
    int n_ops = -1;
    assert(bn_gpu_value_graph_lower_to_shader(&graph, &map, ops, 3,
                                              &n_ops) == 0);
    assert(n_ops == 3);

    assert(ops[0].op_kind == BN_GPU_OP_SSM);
    assert(ops[0].op_code == BN_GPU_CODE_SSM_CONV_SILU);
    assert(ops[0].W_buf == (void *)0x6000);
    assert(ops[0].buf_in == BN_GPU_VALUE_SSM_QKV);
    assert(ops[0].buf_out == -1);
    assert(ops[0].buf_aux == -1);
    assert(ops[0].p[0] == 256);
    assert(ops[0].p[3] == 768);

    assert(ops[1].op_kind == BN_GPU_OP_SSM);
    assert(ops[1].op_code == BN_GPU_CODE_SSM_DELTA);
    assert(ops[1].W_buf == NULL);
    assert(ops[1].buf_in == BN_GPU_VALUE_SSM_QKV);
    assert(ops[1].buf_out == BN_GPU_VALUE_XB2);
    assert(ops[1].buf_aux == BN_GPU_VALUE_SSM_QKV);
    assert(ops[1].rows == 3);
    assert(ops[1].p[3] == 0x3f000000u);

    assert(ops[2].op_kind == BN_GPU_OP_SSM);
    assert(ops[2].op_code == BN_GPU_CODE_SSM_GATE);
    assert(ops[2].W_buf == (void *)0x7000);
    assert(ops[2].buf_in == BN_GPU_VALUE_XB2);
    assert(ops[2].buf_out == -1);
    assert(ops[2].buf_aux == BN_GPU_VALUE_SSM_Z);
    assert(ops[2].rows == 3);
    assert(ops[2].p[1] == 7);

    bn_gpu_value_graph_free(&graph);
    printf("PASSED\n");
}

static void test_graph_lower_utility_to_shader(void) {
    printf("test_graph_lower_utility_to_shader... ");

    BnGPUValueGraph graph;
    bn_gpu_value_graph_init(&graph);

    int x = bn_gpu_value_graph_add_value(
        &graph, BN_GPU_IR_VALUE_TRANSIENT, 0, 1, 128,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_WRITABLE, "x");
    int aux = bn_gpu_value_graph_add_value(
        &graph, BN_GPU_IR_VALUE_TRANSIENT, 0, 1, 128,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_WRITABLE, "aux");
    int norm = bn_gpu_value_graph_add_value(
        &graph, BN_GPU_IR_VALUE_WEIGHT, 0, 1, 128,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_EXTERNAL, "norm");
    int out = bn_gpu_value_graph_add_value(
        &graph, BN_GPU_IR_VALUE_TRANSIENT, 0, 1, 128,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_WRITABLE, "out");

    uint32_t weighted_params[8] = { 128, 0x3f000000u, 0, 0, 0, 0, 0, 0 };
    int weighted = bn_gpu_value_graph_add_utility(
        &graph, x, aux, BN_GPU_IR_INVALID_VALUE, BN_GPU_IR_INVALID_VALUE,
        BN_GPU_IR_UTILITY_WEIGHTED_ADD, 0, weighted_params, "weighted");
    uint32_t residual_params[8] = { 128, 7, 0, 0, 0, 0, 0, 0 };
    int residual = bn_gpu_value_graph_add_utility(
        &graph, x, aux, norm, out, BN_GPU_IR_UTILITY_RESIDUAL_RMSNORM, 0,
        residual_params, "residual_norm");
    uint32_t ph_params[8] = { 16, 7, 1, 32, 0, 0, 0, 0 };
    int ph = bn_gpu_value_graph_add_utility(
        &graph, x, BN_GPU_IR_INVALID_VALUE, norm, BN_GPU_IR_INVALID_VALUE,
        BN_GPU_IR_UTILITY_PER_HEAD_RMSNORM, 8, ph_params, "per_head");
    uint32_t deint_params[8] = { 128, 16, 0, 0, 0, 0, 0, 0 };
    int deint = bn_gpu_value_graph_add_utility(
        &graph, x, BN_GPU_IR_INVALID_VALUE, BN_GPU_IR_INVALID_VALUE, out,
        BN_GPU_IR_UTILITY_DEINTERLEAVE_Q, 0, deint_params, "deint");
    assert(weighted >= 0);
    assert(residual == out);
    assert(ph >= 0);
    assert(deint == out);

    BnGPUIRLoweringValue values[8] = {0};
    for (int i = 0; i < 8; i++)
        values[i].shader_slot = BN_GPU_IR_NO_SHADER_SLOT;
    values[x].shader_slot = BN_GPU_VALUE_X;
    values[aux].shader_slot = BN_GPU_VALUE_XB2;
    values[norm].weight_buf = (void *)0x8000;
    values[out].shader_slot = BN_GPU_VALUE_XB;
    values[weighted].shader_slot = BN_GPU_VALUE_X;
    values[ph].shader_slot = BN_GPU_VALUE_X;
    BnGPUIRLoweringMap map = { values, 8 };

    BnGPUOp ops[4];
    int n_ops = -1;
    assert(bn_gpu_value_graph_lower_to_shader(&graph, &map, ops, 4,
                                              &n_ops) == 0);
    assert(n_ops == 4);

    assert(ops[0].op_kind == BN_GPU_OP_RESIDUAL);
    assert(ops[0].op_code == BN_GPU_CODE_WEIGHTED_ADD);
    assert(ops[0].buf_in == BN_GPU_VALUE_X);
    assert(ops[0].buf_out == -1);
    assert(ops[0].buf_aux == BN_GPU_VALUE_XB2);
    assert(ops[0].p[1] == 0x3f000000u);

    assert(ops[1].op_kind == BN_GPU_OP_RMSNORM);
    assert(ops[1].op_code == BN_GPU_CODE_RESIDUAL_RMSNORM);
    assert(ops[1].W_buf == (void *)0x8000);
    assert(ops[1].buf_in == BN_GPU_VALUE_X);
    assert(ops[1].buf_out == BN_GPU_VALUE_XB);
    assert(ops[1].buf_aux == BN_GPU_VALUE_XB2);

    assert(ops[2].op_kind == BN_GPU_OP_RMSNORM);
    assert(ops[2].op_code == BN_GPU_CODE_PER_HEAD_RMSNORM);
    assert(ops[2].rows == 8);
    assert(ops[2].buf_out == -1);
    assert(ops[2].p[3] == 32);

    assert(ops[3].op_kind == BN_GPU_OP_COPY);
    assert(ops[3].op_code == BN_GPU_CODE_DEINTERLEAVE_Q);
    assert(ops[3].buf_in == BN_GPU_VALUE_X);
    assert(ops[3].buf_out == BN_GPU_VALUE_XB);
    assert(ops[3].p[0] == 128);

    bn_gpu_value_graph_free(&graph);
    printf("PASSED\n");
}

int main(void) {
    test_graph_values_and_aliases();
    test_graph_multi_output_op_and_fallback();
    test_graph_simple_op_builders();
    test_graph_builder_invalid_inputs();
    test_graph_lower_simple_ops_to_shader();
    test_graph_lowering_failures();
    test_graph_lower_split_matvec_to_shader();
    test_graph_lower_attention_to_shader();
    test_graph_lower_ssm_to_shader();
    test_graph_lower_utility_to_shader();
    printf("PASSED\n");
    return 0;
}
