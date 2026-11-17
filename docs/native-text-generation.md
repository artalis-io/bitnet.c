# Native raw-text generation

`bn_generate_native_text` connects the bounded JSON encoder, prepared native
backend session, greedy token selection and streaming UTF-8 decoder. Runtime
inputs come from the encoder and subsequent model logits. Reference files are
used only by the benchmark's verification callbacks.

The explicit API starts with a prepared backend-only session at position zero.
It encodes the entire counted UTF-8 prompt once, prefills caller-selected batch
sizes, and forwards each selected ID before delivering its output. BUSY retries
retain the same staged history and admission policy. Greedy ties choose the
lowest ID. CPU fallback, automatic BOS/EOS insertion, EOS stopping and chat
formatting are outside this raw API. The existing generation APIs are unchanged.

Callers supply immutable encoder/decoder handles for the same checkpoint,
independent encoding scratch, an ID buffer and a counted output buffer. The
output buffer must hold at least six bytes to flush any retained UTF-8 prefix;
a larger model token can require more. The orchestration adds no owned heap
allocation. Backend expert placement and callbacks retain their own budgets.
Keep borrowed JSON alive, use disjoint workspace and serialize backend/session
access. No tokenizer dependency is added to `BnModel`.

Token callbacks receive one complete UTF-8 chunk per committed generated token,
possibly empty, with an explicit byte count. NUL bytes are supported. A final
flush uses ID -1, including after callback-requested early stopping. A failed
forward emits no token callback. Capacity checks precede staging that generated
ID. An error can follow committed prefill batches or earlier tokens; reset and
reprepare the session before retrying. This is a synchronous API, not a serving
scheduler or an asynchronous cancellation interface. Committed-token delivery
waits for a decode forward before the first token callback and also forwards
the last reported token. Prediction-first delivery and avoiding that final
forward remain performance work; this control is not a latency benchmark.

The independent reference uses Rust tokenizers 0.23.3 to encode a multilingual
4,948-byte prompt into 1,026 IDs, without automatic special-token insertion.
The preserved source graph runs fresh 513-token prefill, resumed 513-token
prefill, then five self-fed greedy steps. Streaming chunks follow Python
incremental UTF-8 replacement and complete output is checked against Rust full
decode. The source selects IDs `[25744, 14, 223, 21, 16]`, producing `789, 3.`.

A second capture corrected an inherited provenance description of the old
repeated-ID prompt. All 2,538 binary payloads, totaling 9,679,171,056 bytes,
remained identical. No numerical expected values were changed.

The native control compares all prompt IDs, each forward's input IDs, every
published last-row F32 logit, all output chunks and final text. Two executions
cover both special-token policies. Each execution performs seven forwards and
five self-fed steps, checks 3,619,840 logit bytes, and ends at position 1,031.
The text benchmark requires last-row-only head projection (`all-logits=0`).

For this prompt, the shared encoder owns 8,885,376 bytes and decoder owns
4,266,280 bytes. Request workspace contains 455,296 bytes of encoding scratch,
4,104 bytes of prompt IDs and a 65,536-byte output buffer. These figures exclude
borrowed JSON/parser storage and the separately planned model, expert caches,
backend request, KV and activation buffers. Verification buffers belong to the
benchmark and are not part of the generation API.

Synthetic control tests use independently captured Rust prefix decoding values
and a deliberately controlled forward stub. They cover 168 prefix/early-stop,
batch and repeated-BUSY combinations across two callers sharing immutable
tokenizer models, plus malformed inputs, capacity/overlap rejection, stale
results, stage/forward failures, finite logits, tie selection and counted NUL
prompts/output. The standard test target uses the shared implementation header
and portable module dependencies; sanitizer probes compile the actual generate
module. They deny linked allocations during API calls. These stubs test
orchestration, not model arithmetic; real-weight comparisons exercise the same
public API separately.

Qualification results, source/reference hashes and artifact locations are in
`native-text-generation.json`. The explicit model profile still adapts Diffbot
INT4 BF16 row lookup to Exllama F16 Engram math; this is not proof of original
Diffbot full-runtime equivalence. Ordinary CLI/chat admission, native CPU/ARM
whole-model qualification, asynchronous serving and matched vLLM performance
parity remain open.
