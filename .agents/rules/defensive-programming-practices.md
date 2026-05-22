---
description: Defensive coding practices, buffer safety, pointer discipline, and resource limits.
---

# Defensive Programming Practices

The `milk` project adheres to strong defensive programming practices to guarantee robustness, security, and stability in high-performance environments. These rules complement the `error-handling-practices.md` rule and apply to all human developers and AI agents.

## 1. Buffer and String Safety

- **Ban unbounded functions:** `strcpy()`, `sprintf()`, `strcat()`, and `gets()` are strictly forbidden across the codebase.
- **Use bounded alternatives:** Always use `strncpy()`, `snprintf()`, and `strncat()`. Ensure the destination size is explicitly passed and that the resulting string is null-terminated if the function doesn't guarantee it (e.g., `strncpy()`).

## 2. Pointer Discipline

- **Initialization:** Always initialize pointers to `NULL` or to valid memory immediately upon declaration.
- **Dereference safety:** Validate pointers (especially incoming arguments) against `NULL` before dereferencing them.
- **Dangling pointers:** Immediately after calling `free()` on a pointer, set it to `NULL` to prevent use-after-free bugs.

## 3. Input Validation (Outside the Hot Path)

- **Untrusted input:** Treat all external input—CLI arguments, file contents, environment variables, and SHM parameters—as untrusted. Validate lengths, ranges, and formats prior to processing.
- **Milk context:** Validate FPS parameters in configuration callbacks (e.g., `customCONFcheck` or setup functions), not inside the `fpsexec()` compute loop. Validate `IMGID` or stream dimensions (e.g., `stream.size[0]`) once before entering pixel-processing hot paths to ensure zero overhead in real-time execution.

## 4. Integer Arithmetic and Bounds Checking

- **Safe arithmetic:** Prevent integer overflow and underflow. Be mindful of signed versus unsigned comparisons.
- **Array access:** Validate array indices before accessing memory.
- **Milk context:** Hoist bounds checking and size validation *outside* of `#pragma omp simd` or tight compute loops. Pre-compute safe loop boundaries based on `stream.size[x]` during the initialization phase to guarantee safety without impacting the performance of the hot path.

## 5. State Initialization

- **Zero-initialization:** Prefer `calloc()` over `malloc()` for allocating structs to avoid uninitialized memory bugs. If `malloc()` must be used, explicitly initialize all fields immediately after allocation.
- **Milk context:** Allocate state structures *once* during module initialization (e.g., `init()` phases before real-time control starts) and reuse them across frames. Never use `malloc()` or `calloc()` inside per-frame `fpsexec()` compute loops.

## 6. Resource Limits

- **Avoid exhaustion:** Apply maximum bounds on iterations, file reads, or memory allocations to prevent infinite loops or Out-Of-Memory (OOM) scenarios.
- **Milk context:** Enforce fixed bounds on the number of connected streams, FPS parameters, or loop iterations based on known milk macros (e.g., MAX variables). Do not allow unbounded dynamic allocations or parsing in real-time.
