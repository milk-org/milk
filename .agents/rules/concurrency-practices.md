---
description: Thread safety and concurrency patterns
  for multi-process shared memory code.
---

# Concurrency Practices

`milk` is inherently concurrent — multiple
processes communicate via shared memory streams,
FPS parameters, and processinfo. Follow these
patterns to avoid race conditions.

## Semaphore Protocol

- Use `ImageStreamIO` semaphore functions
  (`sem_wait`, `sem_post`) for stream
  synchronization. Do not use raw POSIX
  semaphores directly.
- Post to semaphore index `-1` to wake all
  readers: `ImageStreamIO_sempost(img.im, -1);`
- Wait on a specific semaphore index assigned
  to your reader.

## FPS Parameter Access

- FPS parameters live in shared memory and can
  be modified by `milk-fpsCTRL` at any time.
- Use the `FPSPROCSYNC` mechanism to sync parameter
  values into local variables at well-defined
  points (start of each iteration).
- Never cache FPS pointer dereferences across
  iterations — the value may change.

## Process Coordination

- Use `processinfo` for heartbeat monitoring
  and loop control (`loopcntMax`, `CTRLval`).
- Do **not** use signals (SIGUSR1, etc.) for
  inter-process coordination — use semaphores
  and shared memory flags instead.
- Check `processinfo->CTRLval` at the top of
  each loop iteration for run/stop/pause
  commands.

## `volatile` Keyword

`volatile` prevents the compiler from caching a
value in a register, but it provides **no**
atomicity or memory ordering guarantees across
CPU cores.

- **Required** for signal handler flags — use
  `volatile sig_atomic_t` (POSIX mandate):
  ```c
  static volatile sig_atomic_t got_sigint = 0;
  ```
- **Acceptable** on single-word SHM status fields
  that are polled in a spin loop, where a brief
  stale value is harmless (e.g.,
  `IMAGE_METADATA.write`). The semaphore post
  after the write provides the actual memory
  barrier.
- **Never use** on pixel data pointers or array
  pointers — it disables SIMD vectorization
  entirely. Synchronize via semaphores instead.
- For counters or complex shared state, prefer
  the `processinfo` mechanism which handles
  updates through shared memory with proper
  synchronization.

## Do NOT Use

- `pthread_mutex` across processes — use
  semaphores instead.
- `sched_setaffinity` — use `processinfo` CPU
  mask instead (see `performance-practices.md`).

