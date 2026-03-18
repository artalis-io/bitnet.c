---
name: c-audit
description: Audit C code for memory safety, SIMD correctness, thread safety, and GGUF parsing security. Use when reviewing or hardening bitnet.c modules.
user-invocable: true
---

# C Code Audit Skill

Perform comprehensive security, safety, and quality audits on bitnet.c C code.

**Target:** $ARGUMENTS (default: all `src/` and `include/` files)

## Usage

```
/c-audit                       # Audit all source files
/c-audit src/transformer.c     # Audit a specific file
/c-audit --fix                 # Audit and apply fixes
```

## Audit Categories

### 1. Memory Safety (Critical)

| Issue | Pattern to Find | Severity |
|-------|-----------------|----------|
| Buffer overflow | `strcpy`, `strcat`, `sprintf`, `gets`, unbounded loops | Critical |
| Unbounded string ops | `strlen`, `strcmp` on untrusted input | Critical |
| Unsafe integer parsing | `atoi`, `atol`, `atof` (no error detection) | High |
| Integer overflow | `malloc(a * b)` without overflow check | Critical |
| Use-after-free | Pointer used after `free()` | Critical |
| Double-free | `free()` called twice on same pointer | Critical |
| Null dereference | Pointer used without NULL check | High |
| Uninitialized memory | Variables used before assignment | High |
| Missing null terminator | String buffer not explicitly terminated | High |
| Memory leak | `malloc`/`calloc` without corresponding `free` | Medium |
| Unsafe VLA | VLA sized by model config without `BN_MAX_VLA_ELEMS` guard | High |
| Stack overflow | Large stack arrays > 32KB | Medium |

**Safe Replacements:**
```c
// Copying
strcpy(dst, src)           -> snprintf(dst, sizeof(dst), "%s", src);

// Formatting
sprintf(buf, fmt, ...)     -> snprintf(buf, sizeof(buf), fmt, ...);

// Memory allocation (overflow-safe)
malloc(count * size)       -> calloc(count, size);

// Integer parsing
atoi(str)                  -> strtol(str, &end, 10) with validation

// VLA guards (all VLAs sized by model config MUST be guarded)
float arr[dim];            -> if (dim > BN_MAX_VLA_ELEMS) return -1;
                              float arr[dim];
```

**Arena vs Direct Allocation:**
```c
// RunState buffers use SHArena (freed as a block in model_free)
float *buf = sh_arena_alloc(arena, n * sizeof(float));

// Temporary allocations during init may use malloc/free
// Verify: every malloc has a matching free on ALL paths (including error paths)
```

### 2. GGUF Parsing Safety (Critical)

Model files are untrusted input. GGUF parsing must be defensive.

| Issue | What to Check | Severity |
|-------|---------------|----------|
| Tensor data OOB | Tensor offset + size exceeds file size | Critical |
| Integer overflow in tensor size | `dims[0] * dims[1] * type_size` overflow | Critical |
| Excessive allocation | `n_tensors` or `n_kv` causing huge allocation | High |
| String bounds | GGUF string length exceeds remaining buffer | Critical |
| Type confusion | Tensor type validated before casting data pointer | High |
| Dimension validation | `n_dims` checked (1-4), dimensions non-negative | High |
| KV lookup | `gguf_find_*` return values checked for not-found | Medium |

### 3. Input Validation

| Issue | What to Check |
|-------|---------------|
| Token bounds | Token validated against `vocab_size` before embedding lookup |
| Position bounds | Position validated against `seq_len` |
| Array indices | All array indices validated before access |
| Pointer validity | NULL checks before dereference |
| Size parameters | Non-negative, within reasonable bounds |
| Model config | Dimensions checked before VLA allocation |

### 4. SIMD Safety (High)

| Issue | What to Check | Severity |
|-------|---------------|----------|
| Alignment | NEON `vld1q_*` doesn't require alignment, but `__attribute__((aligned))` arrays should be verified | Medium |
| Remainder handling | Loops assume dimension is multiple of vector width (4/8/16) — verify model guarantees this | High |
| Type punning | `vreinterpret*` casts between signed/unsigned verified correct | Medium |
| Out-of-bounds SIMD load | Last vector load may read past array if size not aligned | High |
| Backend consistency | NEON, AVX2, WASM, scalar backends produce equivalent results | High |
| Prefetch safety | `__builtin_prefetch` on valid addresses only | Low |

**SIMD remainder pattern:**
```c
// GOOD: dimension guaranteed multiple of 4 by model config
for (int i = 0; i < dim; i += 4)
    vst1q_f32(out + i, vmulq_f32(vld1q_f32(x + i), scale));

// BAD: no remainder handling for arbitrary sizes
// If size may not be multiple of vector width, add scalar tail:
for (; i < size; i++) out[i] = x[i] * ss;
```

### 5. Thread Safety (High)

| Issue | What to Check | Severity |
|-------|---------------|----------|
| Data races | Range functions only write to their assigned range `[start, end)` | Critical |
| Shared state | Context structs are read-only during dispatch (except output arrays) | Critical |
| VLA in threads | VLAs in range functions are thread-local (stack) — verify no shared scratch | High |
| Pool NULL | `bn_tp_dispatch(NULL, ...)` falls back to serial correctly | Medium |
| Task count | `task.n` matches the actual iteration space | High |

**Thread safety pattern:**
```c
// GOOD: each thread writes only its range
void my_range(void *ctx, int start, int end) {
    MyCtx *c = ctx;
    for (int i = start; i < end; i++)
        c->out[i] = compute(c, i);  // out[i] is disjoint per thread
}

// BAD: shared accumulator without atomics
void bad_range(void *ctx, int start, int end) {
    MyCtx *c = ctx;
    for (int i = start; i < end; i++)
        c->total += c->data[i];  // RACE CONDITION
}
```

### 6. Integer Overflow

Overflow in size computations can cause undersized allocations.

```c
// BAD: overflow on 32-bit
int total = num_layers * seq_len * kv_dim * sizeof(float);

// GOOD: use size_t throughout
size_t total = (size_t)num_layers * seq_len * kv_dim * sizeof(float);

// GOOD: check before multiply
if (count > 0 && (size_t)count > SIZE_MAX / elem_size) {
    return -1;  // overflow
}
```

**Key areas in bitnet.c:**
- KV cache allocation (`n_attn_layers * seq_len * kv_dim`)
- SSM state allocation (`n_ssm * num_v_heads * head_k_dim * head_v_dim`)
- Arena allocation sizes
- Embedding table copies (`vocab_size * dim`)

### 7. Resource Management

| Issue | What to Check |
|-------|---------------|
| Model lifecycle | `bn_model_load` paired with `bn_model_free` |
| File mapping | `bn_mapped_file_open` paired with `bn_mapped_file_close` |
| Thread pool | `bn_tp_create` paired with `bn_tp_free` |
| Arena | `sh_arena_create` paired with `sh_arena_free` |
| GGUF parsing | `BnGGUFFile` resources freed after use |
| Error paths | Resources freed on all exit paths in `model_load` |

### 8. Test Coverage

Check test files (`test/test_*.c`) for:
- [ ] Basic functionality tests for each module
- [ ] Edge cases (empty input, max values, NULL)
- [ ] Error path tests (malformed GGUF, OOB tokens)
- [ ] Bounds checking tests
- [ ] All public API functions have at least one test
- [ ] GGUF parser tested with malicious/truncated files
- [ ] Quant kernels tested for NEON vs scalar equivalence
- [ ] Safety regression tests (`test_safety.c`)

### 9. Dead Code Detection

| Pattern | Issue | Fix |
|---------|-------|-----|
| `if (0) { ... }` | Dead branch | Remove |
| `return; code_after;` | Unreachable code | Remove |
| `#if 0 ... #endif` | Disabled code | Remove or document |
| `#ifdef DEBUG` blocks | Debug-only code | Keep minimal, remove file dumps |
| Unused `#define` | Dead macro | Remove |
| Unused static function | Dead function | Remove |
| Stale `(void)var;` casts | Suppressed warnings for removed code | Remove |

### 10. Build Hardening

**Development build (`make debug`):**
```makefile
-DDEBUG -g -O0
```

**Sanitizer build (`make asan`):**
```makefile
-fsanitize=address,undefined -g -O0 -fno-omit-frame-pointer
```

**Production build (`make`):**
```makefile
-O3 -Wall -Wextra -Wshadow -std=c11
```

**Audit Checks:**
- [ ] `-Wall -Wextra -Wshadow` in production CFLAGS
- [ ] Debug build available (`make debug`)
- [ ] ASan + UBSan build available (`make asan`)
- [ ] All tests pass under sanitizers
- [ ] Zero compiler warnings in production build
- [ ] AVX2 cross-compile check passes (`make avx2-check`)

## Audit Procedure

When `/c-audit` is invoked:

1. **Locate Files**
   - If `$ARGUMENTS` specifies files, audit those
   - Otherwise scan: `src/*.c`, `src/quant/*.c`, `src/transformer/*.c`, `include/*.h`, `test/test_*.c`, `Makefile`

2. **Scan for Critical Issues**
   - Search for unsafe functions: `strcpy`, `sprintf`, `gets`, `strcat`, `atoi`, `atol`
   - Search for unchecked allocations: `malloc`/`calloc` without NULL check
   - Search for integer overflow in size calculations (missing `size_t` casts)
   - Search for VLAs without `BN_MAX_VLA_ELEMS` guard
   - Search for SIMD loops without remainder handling on non-guaranteed sizes
   - Search for missing bounds checks on token/position/array indices

3. **Review GGUF Parsing**
   - Tensor data access validated against file bounds
   - String lengths validated
   - Dimension counts validated
   - Integer overflow in tensor size calculations

4. **Check Thread Safety**
   - Range functions write only to `[start, end)` of output
   - Context structs are read-only during dispatch
   - No shared mutable state between threads

5. **Check SIMD Consistency**
   - All NEON kernels have scalar fallback
   - Loop bounds match dimension guarantees
   - Prefetch addresses are valid

6. **Check Resource Management**
   - Every `_init()`/`_create()`/`_load()` has matching `_free()`/`_close()`
   - Error paths in `model_load` free partial state
   - Arena-allocated buffers don't need individual frees

7. **Detect Dead Code**
   - Unused static functions
   - Unreachable code after returns
   - Stale `#ifdef` blocks
   - Unused variables

8. **Check Build Hardening**
   - Compile with `-Wall -Wextra -Wshadow` — verify zero warnings
   - Verify sanitizer build target exists
   - Run `make avx2-check` for cross-compile

9. **Generate Report**
   Format as markdown table with findings, severity, file:line, and suggested fix.

## Report Format

```markdown
## C Audit Report: bitnet.c

**Date:** YYYY-MM-DD
**Files Scanned:** N
**Issues Found:** N (Critical: N, High: N, Medium: N, Low: N)

### Critical Issues

| # | File:Line | Issue | Current Code | Suggested Fix |
|---|-----------|-------|--------------|---------------|
| C1 | src/foo.c:42 | Buffer overflow | `strcpy(buf, src)` | `snprintf(buf, sizeof(buf), "%s", src)` |

### High Issues
...

### Medium Issues
...

### Low Issues
...

### Recommendations
1. ...
```

## Fix Mode (--fix)

When `--fix` is specified:

1. Generate the audit report first
2. For each fixable issue, apply the transformation
3. Rebuild (`make clean && make`)
4. Re-run tests (`make test`)
5. Report any test failures or new warnings

**Auto-fixable Issues:**
- `strcpy` -> `snprintf` with buffer size
- `sprintf` -> `snprintf` with buffer size
- `atoi` -> `strtol` with validation
- Missing NULL checks on allocations (add early return)
- Integer overflow in size calc -> `size_t` cast or overflow check
- Missing VLA guard -> add `BN_MAX_VLA_ELEMS` check
- Unused local variables (remove)
- Unused static functions (remove)

**NOT Auto-fixable (require manual review):**
- Logic errors in SIMD kernels
- Thread safety violations
- Resource leaks in complex control flow
- GGUF parsing boundary issues
- Architectural changes
