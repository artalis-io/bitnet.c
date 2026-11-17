# Native CUDA raw-text CLI

The executable accepts a native model directory through an explicit raw-text
mode. It loads configuration, safetensors metadata, tokenizer and converted
resident weights, prepares bounded expert placement and request buffers, then
calls `bn_generate_native_text`. Generation does not read reference files.
Existing GGUF commands retain their behavior.

```sh
./bitnet --native-text MODEL_DIR --profile exl3-int4-f16 \
  --prompt-file prompt.txt -n 32 \
  --host-budget-mb 4096 --device-budget-mb 32768
```

Use `-p 'text'` instead of a file, or `--prompt-file -` for counted stdin.
Prompt files can contain NUL and are limited to 16 MiB of valid UTF-8.
Stdout contains only counted generated UTF-8 bytes, without an added newline.
Diagnostics and optional `--token-ids` output go to stderr.

The required profile selects EXL3 experts and INT4 BF16 row lookup adapted to
F16 Engram math. This is the previously qualified arithmetic adapter, not proof
of original Diffbot full-runtime equivalence. The mode uses greedy selection
and adds no BOS, chat template or automatic EOS stopping. Other native families,
native CPU/Metal execution and chat policy are not admitted by this mode.

Defaults are 512-token prefill batches, a 4,096-token context cap, 640 MiB each
of RAM and decode expert-cache payload, 48 temporary expert slots and prefill
upload window 32. `--admit-decode` enables persistent decode-cache admission;
`--pinned-staging` selects the bounded pinned expert upload path. `--keep-special`
preserves special-token text. Unknown and duplicate options reject before load.
The selected temporary quota must cover the possible cold working set so a
single request cannot retry forever waiting for impossible capacity.

Host and device payload caps are optional and checked in stages. Tokenizer and
request-buffer bounds are checked before their allocations; resident-weight
bounds are checked before GPU upload. Existing request planners require real
resident weights and backend capabilities. After weight preparation, the CLI
checks the full request bound before allocating expert stores and activation/KV
resources. These are conservative payload bounds, excluding JSON/index,
allocator, driver/library overhead and OS file cache; they are not RSS limits.
Zero aggregate caps leave admission to the explicit component plans and quotas.

```sh
./bitnet --native-text MODEL_DIR --profile exl3-int4-f16 \
  --prompt-file prompt.txt -n 32 --prepare-only
```

`--prepare-only` **loads resident CUDA weights and prepares the request** before
printing payload-plan JSON, then frees it without inference. It is not a
GPU-free dry run. Separating backend-independent sizing from resident-weight
validation remains future work.

All failure paths release the request before model/backend storage and the
borrowed safetensors index. Invalid input/context, insufficient payload caps,
unsupported profiles, allocation/device errors and stdout failures return a
nonzero exit. A stream may already contain committed output when a later error
occurs. Usage errors return 2; runtime errors return 1.

For diagnostics, `--trace-dir DIR` writes `events.jsonl` and per-forward
`forward-N.i32le` inputs and `forward-N.f32le` last-row logits. The directory must
already exist. Files use exclusive creation and never overwrite an existing
path. Tracing is optional and affects timings. The normal path writes no such
files.

Independent source references cover both default `512 + 512 + 2` and explicit
`513 + 513` prefill schedules for a multilingual 4,948-byte / 1,026-token prompt,
followed by five self-fed source tokens. CLI verification compares every actual
input ID and last-row F32 logit, generated IDs, callback ordering, chunk lengths
and raw stdout. The default schedule uses eight forwards and compares
4,136,960 logit bytes; the earlier schedule uses seven and 3,619,840 bytes. Both
produce `789, 3.` and finish at position 1,031.

The real CLI integration suite covers file/stdin/inline inputs, special-token
policy, pinned staging and decode admission, exact rounded-MiB cap admission,
weight/request/host budget rejection, context and UTF-8 errors, empty files,
trace protection, stdout and device errors, and incompatible/duplicate options.
Model-free tests cover parsing, checked arithmetic, bounded counted reads and
little-endian serialization. Adjacent JSON records final results and hashes.

This remains synchronous committed-token delivery: the first token callback
waits for its decode forward, and the last reported token is forwarded too.
Prediction-first delivery, avoiding that last forward, asynchronous serving,
longer-context whole-model qualification and matched vLLM parity remain open.
No throughput or latency parity is claimed from these correctness controls.
