# Sail Positioning Note

## Rename Recommendation

If this project is meant to be more than a narrow BitNet implementation, consider
renaming it from `bitnet.c` to **Sail**.

Recommended expansion:

**SAIL — Small AI Inference Layer**

Why it fits:

- It belongs naturally beside **Hull** and **Keel**.
- It is broader than BitNet and can cover multiple local/open-weight model
  backends over time.
- It describes the architectural role: the local inference layer used by Hull
  applications and agents.
- It keeps `bitnet.c` available as the first backend/kernel rather than making
  the whole product name depend on one model architecture.

Suggested naming:

- Project: `Sail`
- Full name: `SAIL — Small AI Inference Layer`
- Repo: `artalis-io/sail`
- Domain: `libsail.dev` or `sail.artalis.io`
- Legacy: `bitnet.c` becomes the original CPU-first inference backend inside
  Sail.

## Platform Positioning

The Artalis platform can be framed as three low-level primitives:

- **Hull — Hardened Userspace Lockdown Layer**
  - Capability-secure runtime for local-first and agent-native applications.
  - Owns manifests, policies, audit, sandboxing, orchestration, Lua/JS, WASM,
    bundles, and application lifecycle.

- **Keel — Kernel Event Engine, Lightweight**
  - Default event loop and transport substrate for Hull.
  - Also usable directly by native C services.
  - Owns HTTP, WebSocket, SSE, async clients, timers, fd watchers, routing,
    middleware hooks, and side-car transport.

- **Sail — Small AI Inference Layer**
  - Local inference layer for open-weight models.
  - Runs as an embedded library or sandboxed side-car behind Hull and Keel.
  - Owns CPU-first inference, quantized kernels, prompt/session state, model
    loading, and local agent execution.

One-sentence architecture:

> Hull is the secure runtime boundary, Keel is the event and transport substrate,
> and Sail is the local inference layer for private agentic applications.

## Side-Car Role

Sail should be positioned as the native inference side-car for Hull rather than
as a general cloud inference server.

Primary use cases:

- Run local coding and workflow agents without SaaS model calls.
- Keep private data and prompts on-device or on-prem.
- Provide a single-binary or small-binary deployment path for constrained
  environments.
- Support CPU-first inference first, with optional GPU acceleration where useful.
- Expose a narrow local protocol through Keel so Hull can apply explicit
  capabilities, audit, and sandbox policy around agent execution.

This keeps the story coherent:

- Hull decides what an agent/app is allowed to do.
- Keel moves requests, streams, events, and side-car traffic.
- Sail runs local model inference under that controlled boundary.
