# Module: info

Image information, statistics, and stream monitoring.

## Source Files

| File                   | Description                                           |
| ---------------------- | ----------------------------------------------------- |
| `info.c`               | Module registry (`MILK_MODULE_RUNNABLES`)             |
| `cubeMatchMatrix.c`    | Compute pairwise slice-difference matrix              |
| `cubestats.c`          | Per-slice statistics of image cube                    |
| `image_stats.c`        | Basic image statistics (mean, RMS, min, max)          |
| `improfile.c`          | Radial profile of image                               |
| `kbdhit.c`             | Non-blocking keyboard hit detection                   |
| `percentile.c`         | Compute image percentiles                             |
| `print_header.c`       | Print image/stream header info                        |
| `stream_monproc.c`     | Multi-level time-binned stream monitor with histogram |
| `stream_monproc_shm.c` | Shared-memory state accessors for `stream_monproc`    |
| `streamtiming_stats.c` | Stream timing jitter statistics                       |
| `timediff.c`           | Compute timestamp differences                         |
| `bin/milk-shmimmon.c`  | Standalone ANSI shared-memory stream rate monitor     |
| `bin/stream-monproc-disp.c` | Display viewer for `stream_monproc` telemetry    |

## Standalone Executables

| Executable                     | Source File                 | Description                                    |
| ------------------------------ | --------------------------- | ---------------------------------------------- |
| `milk-fpsexec-info-strmonproc` | `stream_monproc.c`          | Multi-level time-binned stream monitor         |
| `milk-shmimmon`                | `bin/milk-shmimmon.c`       | Live ANSI shared-memory stream rate monitor    |
| `stream-monproc-disp`          | `bin/stream-monproc-disp.c` | Display viewer for `stream_monproc` telemetry  |

## Dependencies

- `milk_runnable.h` (`libcomputeunit`), which provides the FPS and processinfo macros
