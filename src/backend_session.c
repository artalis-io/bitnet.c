#include "backend_model.h"
#include "gpu_shader_ir.h"
#include <stdlib.h>

struct BnBackendSession {
    void *gpu_graph;
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

void bn_backend_session_set_gpu_graph(BnBackendSession *backend, void *graph) {
    if (!backend) return;
    backend->gpu_graph = graph;
}
