# Native streaming text decoding

`BnTokenizerDecoder` provides an immutable JSON ByteLevel vocabulary and special-token
policy. If a token contains any character outside the ByteLevel alphabet,
decoding uses that token’s whole raw UTF-8 spelling, matching Rust. `BnTokenizerDecodeStream` keeps request-local UTF-8 state. This avoids the
corruption caused by concatenating per-token lossy strings when UTF-8 characters
span tokens. The existing per-token vocabulary and GGUF tokenizer APIs retain
their behavior.

Start with zeroed handles. The decoder borrows JSON strings: keep the document
immutable and alive until the decoder is freed, and keep the decoder alive until
all streams have finished. Streams need independent state and output buffers;
quiescent value copies support checkpoints. `skip_special` discards bytes of
special added tokens without breaking a pending character. Added-token metadata
must explicitly contain Boolean `special` flags.

Each push returns counted, complete UTF-8 chunks, including embedded NUL.
The chunk and pending-byte contract follows Python incremental UTF-8 replacement.
Every captured sequence's combined output and final flush is independently
checked against Rust tokenizers 0.23.3 full ByteLevel decoding. This does not claim
Rust `DecodeStream` chunk timing: that API can retain a literal replacement
character indefinitely. The bounded state also preserves Python's deferred
`ED A0..BF` surrogate prefix until another byte or flush.

`finish` flushes incomplete input and makes a stream terminal until reset.
Invalid IDs and insufficient capacity preserve state, output and the written
count. Push validates and sizes a local state copy before writing output.
Caller memory must be valid; output and mutable state must respect the public
header's disjointness and borrowed-document requirements.

The actual 129,280-token model owns 4,266,280 bytes in exactly two allocations,
excluding borrowed JSON/parser storage and allocator overhead. A stream occupies
24 bytes on the qualified host and retains at most three unfinished bytes.
Push, reset, init and stream free perform no heap operations. The synthetic
261-token decoder owns 8,653 bytes.

Independent references cover 419,754 actual and 161,716 synthetic sequences:
581,470 cases, 1,282,874 IDs, 1,864,344 token/flush steps and 4,191,788 output bytes.
A separate source capture adds 1,548 admitted mixed-alphabet cases; six further
cases correctly reject embedded NUL in JSON strings. ByteLevel-mapped NUL output
remains supported.
These include every vocabulary ID, every byte pair, three/four-byte boundaries,
invalid UTF-8, special tokens, unmapped Unicode, NUL and long streams. Candidate
checks exercise individual steps, grouped input, empty pushes, checkpoint copies,
capacity retries and allocation failures. The source producer refuses differing
existing reference payloads.

Qualification results and immutable artifact hashes are recorded in the adjacent
JSON metadata. This component prepares native text generation; it does not yet
admit text-to-inference, chat templates or native CLI generation, and makes no
vLLM performance claim.
