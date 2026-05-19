---
description: Safe practices for shared memory
  streams, FPS, and processinfo files.
---

# Shared Memory Safety

`milk` uses `/dev/shm/` extensively for streams
(`*.im.shm`), FPS (`fps.*`), and processinfo
(`proc.*`). Follow these practices to avoid
data corruption and resource leaks.

## Cleanup After Testing

When running commands that create shared memory
files during testing or debugging, clean up
afterward:

```bash
rm -f /dev/shm/test_*.im.shm
rm -f /dev/shm/fps.test_*.shm
```

Never leave test SHM files behind — they
persist across reboots on tmpfs systems.

## Stale SHM Detection

Before connecting to an existing stream, verify
it is still actively maintained. Symptoms of
stale SHM:

- Stream exists in `/dev/shm/` but no process
  is writing to it.
- Semaphore values are stuck (not incrementing).
- `processinfo` shows no heartbeat for the
  owning process.

Use `streamCTRL` or `milk-streamCTRL` to inspect
stream health.

## Do Not Overwrite Other Users' SHM

When running on a shared system, never overwrite
or delete SHM files owned by other users without
explicit confirmation. Check ownership with:

```bash
ls -la /dev/shm/*.im.shm
```

## tmux Session Cleanup

Standalone executables with `-tmux` create tmux
sessions. Clean up orphaned sessions:

```bash
tmux ls 2>/dev/null
tmux kill-session -t <name>
```

## Stream Creation Best Practices

- Always set `.shared = 1` for streams that will
  be read by other processes.
- Use `imgid_mkimage()` to create streams — it
  handles SHM file creation and semaphore
  initialization.
- Post semaphores after writing new data so
  readers unblock:
  `ImageStreamIO_sempost(img.im, -1);`
