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

## 7. Format String Safety

- **Safe formatting:** Never pass user input or external data directly as the format string to `printf()`, `snprintf()`, or the `PRINT_ERROR` macros.
- **Why:** Passing untrusted strings directly (e.g., `PRINT_ERROR(untrusted_string)`) exposes the application to format string vulnerabilities and memory corruption if the string contains specifiers like `%x` or `%n`.
- **Enforcement:** Always use a literal format string: `PRINT_ERROR("%s", untrusted_string)`.

## 8. Time-of-Check to Time-of-Use (TOCTOU) in Shared Memory

- **Concurrent mutation:** When dealing with `/dev/shm` IPC, understand that shared memory can change between the time you check it and the time you use it. Another tmux session or compute unit might alter a stream's dimensions or state while you are reading it.
- **Enforcement:** Rely on atomic operations, semaphores (as detailed in `concurrency-practices.md`), or local caching of immutable state before entering your compute loops.

## 9. Safe Signal Handling

- **Minimal handlers:** Keep signal handlers (e.g., for `SIGINT`, `SIGTERM`) strictly minimal.
- **Why:** Signal handlers interrupt the normal flow of the program. Calling non-reentrant functions like `malloc()`, `printf()`, or even `PRINT_ERROR()` inside a signal handler can cause deadlocks or crashes.
- **Enforcement:** Signal handlers should only set a `volatile sig_atomic_t` flag (e.g., `run_status = 0`) that the main loop safely checks.

## 10. Compiler Sanitizers

- **Automated detection:** Automated tooling catches edge cases that rules might miss. Use AddressSanitizer (ASAN) and UndefinedBehaviorSanitizer (UBSAN) to catch buffer overflows, dangling pointers, and undefined behavior at runtime.
- **Enforcement:** Ensure that tests and CI workflows test the code with these sanitizers enabled (e.g., compiling with `-fsanitize=address,undefined`).
