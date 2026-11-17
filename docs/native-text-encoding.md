# Native text encoding

A native JSON byte-level BPE encoder now produces counted token IDs for the
qualified tokenizer profile. Added-token matching runs before the three ordered
number, CJK and letter/punctuation splits. The encoder rejects unsupported
normalization and token transformations. It adds no BOS, EOS or chat template.
Counted UTF-8 inputs can include NUL; malformed UTF-8 and insufficient capacity
reject without publishing partial outputs.

The encoder owns immutable vocabulary indices, ranked merges and an added-token
trie, borrowing strings from a JSON document that must outlive use. Callers
start with zeroed handles and
supply independent bounded scratch. BPE uses a priority heap with versioned
stale-event rejection and leftmost rank ties, avoiding quadratic rescans.
Construction sorts in place, with exactly two metadata allocations. Encoding
and diagnostic pre-tokenization made **zero linked heap calls** even with
allocation denied during the calls.

Independent Rust tokenizers 0.23.3 references verified:

- **134,033 checkpoint text cases / 330,625 exact token IDs**,
  including Rust-decoded text for every one of the 129,280 vocabulary IDs.
- **3,473 synthetic cases / 347,177 exact IDs**,
  covering ranked merge ties, added tokens, spaces, numbers, CJK, marks and NUL.
- **292,561 boundary cases / 1,359,035 exact IDs**:
  all combinations through length four of 23 character-class representatives.
- Every valid Unicode scalar in four contexts: **4,448,256 scalar placements**
  and **7,618,356 exact pre-tokenizer boundaries**. Character categories
  come from separately captured source Regex behavior.
- Long inputs up to **212,992 bytes**, source string/array merge serialization,
  malformed setup/input, insufficient budgets/capacity and direct alias checks.

The actual encoder owns **8,885,376 bytes**; scratch for the
largest captured input owns **19,595,344 bytes**. These bounds
exclude borrowed JSON/parser storage, allocator overhead and static code/data.
Caller output and borrowed document storage must be disjoint. Handles cannot
be copied or freed while in use.

All 18 qualification gates passed. ASAN/UBSAN covers all new C paths and full
reference replay; two workers sharing an encoder passed normal, ASAN and
ThreadSanitizer source comparisons with separate scratch/output. Review found
a DEL/combining-mark boundary error; it was fixed and covered by 133
additional source-captured ASCII-prefix cases. Both encoder
allocation failures clean up without publishing state. All source corpora and
generated headers replay identically; conflicting reference payloads are never
overwritten. The protected clean product build, standard suite, full CUDA suite
and diagnostics passed with zero warnings; four user reference binaries were
preserved. Fresh strict Qwen compatibility and the existing source-indexed
DeepSeek forward control also passed.

This is an encoding component, not native chat/CLI admission. Streaming UTF-8
decode, template/special-token policy, real-text full-model qualification,
broader model/backend/context coverage, production serving and matched vLLM
parity remain open. The full goal remains active and incomplete.

Evidence: [native-text-encoding.json](native-text-encoding.json).
