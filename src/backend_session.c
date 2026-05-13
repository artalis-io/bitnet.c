#include "backend_session.h"
#include "gpu_graph_ir.h"
#include "gpu_shader_ir_internal.h"
#include <stdlib.h>

struct BnBackendSession {
    void *gpu_graph;
    BnGPUValueGraph gpu_value_graph;
    void *gpu_lowering_values;
    int cap_gpu_lowering_values;
};

BnBackendSession *bn_backend_session_create(void) {
    return (BnBackendSession *)calloc(1, sizeof(BnBackendSession));
}

void bn_backend_session_release_gpu_graph(BnBackendSession *backend) {
    if (!backend) return;
    if (backend->gpu_graph) {
        BnGPUGraph *g = (BnGPUGraph *)backend->gpu_graph;
        free(g->ops);
        free(g);
        backend->gpu_graph = NULL;
    }
}

void bn_backend_session_free(BnBackendSession *backend) {
    if (!backend) return;
    bn_backend_session_release_gpu_graph(backend);
    bn_gpu_value_graph_free(&backend->gpu_value_graph);
    free(backend->gpu_lowering_values);
    free(backend);
}

void *bn_backend_session_gpu_graph(const BnBackendSession *backend) {
    return backend ? backend->gpu_graph : NULL;
}

void *bn_backend_session_ensure_gpu_graph(BnBackendSession *backend, int cap_ops) {
    if (!backend || cap_ops <= 0) return NULL;
    BnGPUGraph *graph = (BnGPUGraph *)backend->gpu_graph;
    if (graph && graph->cap >= cap_ops) return graph;

    if (!graph) {
        graph = (BnGPUGraph *)calloc(1, sizeof(BnGPUGraph));
        if (!graph) return NULL;
    } else {
        free(graph->ops);
        graph->ops = NULL;
    }
    graph->ops = (BnGPUOp *)malloc((size_t)cap_ops * sizeof(BnGPUOp));
    if (!graph->ops) {
        free(graph);
        backend->gpu_graph = NULL;
        return NULL;
    }
    graph->cap = cap_ops;
    backend->gpu_graph = graph;
    return graph;
}

void *bn_backend_session_ensure_gpu_command_buffer(BnBackendSession *backend,
                                                   int cap_ops,
                                                   int *out_cap) {
    BnGPUGraph *graph =
        (BnGPUGraph *)bn_backend_session_ensure_gpu_graph(backend, cap_ops);
    if (!graph)
        return NULL;
    if (out_cap)
        *out_cap = graph->cap;
    return graph->ops;
}

BnGPUValueGraph *bn_backend_session_gpu_value_graph(BnBackendSession *backend) {
    return backend ? &backend->gpu_value_graph : NULL;
}

int bn_backend_session_ensure_gpu_lowering_values(BnBackendSession *backend,
                                                  int elem_size,
                                                  int cap_values,
                                                  void **out_values,
                                                  int *out_cap) {
    if (!backend || elem_size <= 0 || cap_values < 0 || !out_values)
        return -1;
    if (backend->cap_gpu_lowering_values < cap_values) {
        int new_cap = backend->cap_gpu_lowering_values
            ? backend->cap_gpu_lowering_values * 2
            : 16;
        while (new_cap < cap_values) new_cap *= 2;
        void *new_values = realloc(
            backend->gpu_lowering_values,
            (size_t)new_cap * (size_t)elem_size);
        if (!new_values)
            return -1;
        backend->gpu_lowering_values = new_values;
        backend->cap_gpu_lowering_values = new_cap;
    }
    *out_values = backend->gpu_lowering_values;
    if (out_cap)
        *out_cap = backend->cap_gpu_lowering_values;
    return 0;
}

void bn_backend_session_set_gpu_graph(BnBackendSession *backend, void *graph) {
    if (!backend) return;
    backend->gpu_graph = graph;
}
