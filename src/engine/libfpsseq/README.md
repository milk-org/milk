# libfpsseq

FPS Sequencer library for orchestrating multi-step
workflows across FPS-managed compute units.

## Purpose

Provides a shared-memory sequencer (`milk-seq`) that
can schedule, queue, and execute commands targeting
FPS parameter entries. Used by `milk-fpsCTRL` and
automated calibration pipelines to coordinate
start/stop/configure operations across multiple
processes.

## Architecture

The sequencer state lives in shared memory
(`/dev/shm/milkseq.*`), enabling multiple clients
to observe and inject commands concurrently.

```
FIFO input → milkseq_fifo_read()
                  ↓
           Task Queue (SHM)
                  ↓
         milkseq_scheduler_step()
                  ↓
          milkseq_exec_cmd()
                  ↓
            FPS parameter write
```

## Files

| File                 | Description                                    |
| -------------------- | ---------------------------------------------- |
| `fpsseq_types.h`     | `MILKSEQ_STATE`, task struct, constants        |
| `fpsseq.h`           | Public API declarations                        |
| `fpsseq_shm.c`       | SHM lifecycle (create, connect, destroy, list) |
| `fpsseq_scheduler.c` | Task scheduling and priority logic             |
| `fpsseq_fifo.c`      | Non-blocking FIFO command reader               |
| `fpsseq_cmdexec.c`   | Command parser and executor                    |
| `fpsseq_script.c`    | `.seq` script loader and compiler              |
| `milk-seq.c`         | `milk-seq` standalone executable               |
| `milk-seq-help.c`    | Help text for `milk-seq`                       |

## Build Tier

Engine tier — built with all configurations.
