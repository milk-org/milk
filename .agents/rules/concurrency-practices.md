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

- A unit writes and waits on streams as
  [procinfo.md §7](../../docs/guide/procinfo.md#7-compute-function-macros) shows:
  `processinfo_update_output_stream()` posts the semaphores, and the `MILK_PROCINFO_*` loop
  macros wait on the input.
- Elsewhere use the ImageStreamIO calls (`ImageStreamIO_semwait()`, `ImageStreamIO_sempost()`),
  never raw POSIX semaphores: `ImageStreamIO_sempost(img.im, -1)` wakes all readers; a reader waits
  on the semaphore index assigned to it.

## FPS Parameter Access

- FPS parameters live in shared memory and can
  be modified by `milk-fpsCTRL` at any time.
- `MILK_PROCINFO_LOOPSTART(self)` copies them into the module variables at the start of each
  iteration ([programmers_guide.md §3](../../docs/arch/programmers_guide.md#3-writing-a-compute-unit)):
  never keep a value across iterations; within one, hoist it out of inner loops
  (`performance-practices.md`).

## Process Coordination

- Use `processinfo` for heartbeat monitoring
  and loop control (`loopcntMax`, `CTRLval`).
- Do **not** use signals (SIGUSR1, etc.) for
  inter-process coordination — use semaphores
  and shared memory flags instead.
- The `MILK_PROCINFO_*` loop macros handle run, stop and pause requests: do not poll
  `processinfo->CTRLval` yourself.

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
