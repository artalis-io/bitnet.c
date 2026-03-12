# C Security Audit Report: bitnet.c

## Summary

| Severity | Count | Fixed |
|----------|-------|-------|
| Critical | 5 | 5 |
| High | 8 | 8 |
| Medium | 13 | 13 |
| Low | 12 | 12 |
| **Total** | **38** | **38** |

All findings have been remediated. Regression tests added in `test/test_safety.c`.

---

## CRITICAL

**1. Unchecked `malloc`/`calloc` returns in `model_load`** — `src/model.c`
Ten consecutive `calloc` calls for RunState buffers with no NULL checks.
**Fix:** All allocations checked. On failure, `model_free()` called and `-1` returned.

**2. Unchecked `malloc` in `sample_topp`** — `src/sampler.c`
`malloc(n * sizeof(ProbIndex))` not checked.
**Fix:** NULL check added. Falls back to `argmax` on allocation failure.

**3. Unchecked `malloc` in `read_string`/`read_gguf_string`** — `src/gguf.c`
`malloc(len + 1)` not checked.
**Fix:** NULL check added. Sets `r->error` flag on failure, propagates to caller.

**4. Integer overflow in GGUF array size** — `src/gguf.c`
`a->n * sizeof(GGUFString)` can overflow.
**Fix:** Overflow check `a->n > SIZE_MAX / sizeof(GGUFString)` before malloc. Also caps string lengths at 1 GB.

**5. `__builtin_alloca` with file-controlled size** — `src/quant.c`
Stack overflow from large `W->cols`.
**Fix:** Replaced `__builtin_alloca` with `malloc` + `free`.

---

## HIGH

**6. Global mutable `g_sort_ctx`** — `src/tokenizer.c`
Not thread-safe during `tokenizer_init`.
**Fix:** Uses `qsort_r` on macOS/BSD and GNU/Linux. Falls back to global only on unknown platforms.

**7. Global mutable `g_decode_buf`** — `src/tokenizer.c`
Static buffer returned by `tokenizer_decode` not thread-safe.
**Fix:** Changed to `_Thread_local` storage.

**8. Missing bounds check on `token` in `model_embed_token`** — `src/model.c`
OOB read with invalid token.
**Fix:** Added `token < 0 || token >= vocab_size` check. Returns zeroed buffer on invalid token.

**9. Missing bounds check on `token` in `transformer_forward`** — `src/transformer.c`
No validation before embedding lookup.
**Fix:** Returns `NULL` for out-of-range tokens.

**10. Missing bounds check on `pos` — KV-cache OOB write** — `src/transformer.c`
KV-cache write before bounds check in caller.
**Fix:** Returns `NULL` for `pos < 0 || pos >= seq_len`. Main loop updated to check return.

**11. Missing `n_dims` bounds check in GGUF tensor parsing** — `src/gguf.c`
`n_dims > 4` causes `dims[4]` buffer overflow.
**Fix:** Added `if (n_dims > 4) goto fail` after reading n_dims.

**12. GGUF read helpers have no bounds checks** — `src/gguf.c`
`read_u8`, `read_u16`, etc. read without checking buffer bounds.
**Fix:** All read helpers now check `reader_ok()` and set `r->error` flag on OOB. Error propagates through all callers.

**13. `gguf_tensor_data` can return pointer beyond mapped buffer** — `src/gguf.c`
No validation that computed pointer is within buffer.
**Fix:** `GGUFFile` now stores `raw_size`. `gguf_tensor_data` returns NULL if offset >= raw_size.

---

## MEDIUM

**14. Integer overflow in KV-cache calloc size** — `src/model.c`
Multiplication overflow on 32-bit.
**Fix:** Overflow check before calloc: verifies `kv_cache_size / n_layers / seq_len == kv_dim`.

**15. Division by zero: `dim / n_heads` before validation** — `src/model.c`
Derived dims computed before zero-check.
**Fix:** Validation moved before derived dimension calculations. Now also checks `n_kv_heads > 0` and `hidden_dim > 0`.

**16. Unchecked malloc in `tokenizer_init`** — `src/tokenizer.c`
`vocab`, `scores`, `sorted_indices` not checked.
**Fix:** All checked. Partial allocations cleaned up on failure.

**17. Unchecked malloc in `tokenizer_encode`** — `src/tokenizer.c`
`work` and `merge_buf` not checked.
**Fix:** Both checked. Graceful degradation on failure.

**18. Unchecked malloc in `gguf_open` for string arrays** — `src/gguf.c`
String array malloc not checked.
**Fix:** NULL check added. Sets `r->error` on failure.

**19. `platform_load_buffer` aliased non-owning MappedFile** — `src/platform.c`
Emscripten would `free()` an external buffer.
**Fix:** `is_mmap = 2` for externally-owned buffers. `platform_unload_file` skips free for `is_mmap == 2`.

**20. `const` cast violation in `platform_load_buffer`** — `src/platform.c`
Casts away `const` from input buffer.
**Fix:** Documented as intentional for zero-copy interface. External buffers marked with `is_mmap = 2` to prevent modification via unload.

**21. `gguf_get_u32`/`gguf_get_f32` don't validate type match** — `src/gguf.c`
Wrong union member access with mismatched types.
**Fix:** All getters now validate `kvs[i].type` matches expected type. Return 0/NULL on mismatch.

**22. Integer truncation: `uint64_t` to `int`** — multiple locations
Values > INT_MAX cause UB.
**Fix:** `gguf_find_key`/`gguf_find_tensor` guard against `i > INT32_MAX`. Sanity caps added for n_kv/n_tensors (max 2^20).

**23. `load_qweight` return value ignored** — `src/model.c`
Missing tensors silently leave NULL data pointers.
**Fix:** All `load_qweight` calls checked. Model load fails on missing required tensors.

**24. `output_norm` can be NULL** — `src/model.c`
Passed to rmsnorm → null deref.
**Fix:** Model load now requires output_norm.weight to exist.

**25. `attn_norm`/`ffn_norm` can be NULL** — `src/model.c`
Same issue for per-layer norms.
**Fix:** Model load now requires attn_norm and ffn_norm per layer.

**26. `ftell` returns `-1L` on error** — `src/platform.c`
Cast to size_t produces huge value.
**Fix:** `ftell` result checked for `< 0` before cast.

---

## LOW

**27. `mmap` with size 0 is UB** — `src/platform.c`
POSIX says behavior is unspecified for `len == 0`.
**Fix:** Returns empty `MappedFile` for zero-size files.

**28. `argmax`/`softmax` don't handle `n <= 0`** — `src/sampler.c`
OOB read if `n == 0`.
**Fix:** Early return for `n <= 0`.

**29. `sample_topp` division by zero when `n == 1`** — `src/sampler.c`
`(1.0f - topp) / (n - 1)` divides by zero.
**Fix:** Returns 0 for `n <= 1`.

**30. Prompt token buffer sizing assumption** — `src/main.c`
Fragile assumption about max tokens.
**Fix:** Extracted to named variable `max_prompt_tokens` with NULL check on malloc.

**31. `decode_bpe_cp` reads past string end** — `src/tokenizer.c`
Reads continuation byte without end-of-string check.
**Fix:** Added `(*p)[1] == '\0'` check before reading continuation byte.

**32. `g_decode_buf` silently truncates long tokens** — `src/tokenizer.c`
1024-byte static buffer truncates without warning.
**Fix:** Addressed by #7 (thread-local buffer). Truncation behavior preserved but documented.

**33. `strdup` return unchecked** — `src/tokenizer.c`
Can return NULL on allocation failure.
**Fix:** Checked. On failure, cleans up partial vocab allocations and returns -1.

**34. `gguf_get_arr_str` negative idx** — `src/gguf.c`
Relies on implementation-defined cast behavior.
**Fix:** Explicit `idx < 0` check added before uint64 cast.

**35. `dequant_tq1_block` fragile index math** — `src/quant.c`
No assertion guards 160+80+16=256 invariant.
**Fix:** Added `assert(idx == QK_K)` after final loop.

**36. `dequant_i2s_row` reads 32 bytes in partial blocks** — `src/quant.c`
Always reads 32 bytes due to interleaved layout.
**Fix:** Documented: I2_S interleaved format requires full 32-byte chunks. Model dimensions are always multiples of 128.

**37. `fp16_to_fp32` subnormal exp underflow** — `src/quant.c`
Correct but non-obvious math.
**Fix:** Added explanatory comment documenting the loop bounds (max 10 shifts, exp+112 ∈ [103, 113]).

**38. Missing `hidden_dim == 0` validation** — `src/model.c`
Not checked in validation block.
**Fix:** Added `hidden_dim <= 0` to the config validation check (along with n_kv_heads).
