---
description: Safe practices for shared memory
  streams, FPS, and processinfo files.
---

# Shared Memory Safety

`milk` keeps streams (`*.im.shm`), FPS (`*.fps.shm`, `*.fps.d/`) and processinfo (`proc.*.shm`) in
`$MILK_SHM_DIR` (default `/milk/shm`, a tmpfs). Follow these practices to avoid data corruption and
resource leaks.

## Cleanup After Testing

Never leave test SHM files behind: remove them as
[debugging.md §4.2](../../docs/operations/debugging.md#42-shared-memory-leaks) says
(`milk-stream-rm`, `milk-fps-rm -f`, `milk-shmimpurge`), not with `rm`. The pytest suite already
runs in its own `MILK_SHM_DIR`.

## Stale SHM Detection

Before connecting to an existing stream, verify
it is still actively maintained. Symptoms of
stale SHM:

- Stream exists in `$MILK_SHM_DIR` but no process
  is writing to it.
- Semaphore values are stuck (not incrementing).
- `processinfo` shows no heartbeat for the
  owning process.

Use milk-CTRL's streams view (`milk-streamCTRL`, F3, with its inspector) to inspect
stream health.

## Do Not Overwrite Other Users' SHM

When running on a shared system, never overwrite
or delete SHM files owned by other users without
explicit confirmation. Check ownership with:

```bash
ls -la "$MILK_SHM_DIR"/*.im.shm
```

## Unit Cleanup

A unit started with `runstart` keeps running after its starter exits: list and remove leftover
units as [debugging.md §4.2](../../docs/operations/debugging.md#42-shared-memory-leaks) says.

## Stream Creation Best Practices

- Always set `.shared = 1` for streams that will
  be read by other processes.
- Create streams as [streams.md §5](../../docs/guide/streams.md#5-c-api-imgid) says (which creator fits which
  case).
- Post semaphores after writing new data so readers unblock (`concurrency-practices.md`).
