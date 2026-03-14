# Module: info

Image information, statistics, and stream monitoring.

## Source Files

| File | Description |
|------|-------------|
| `cubeMatchMatrix.c` | Compute pairwise slice-difference matrix |
| `cubestats.c` | Per-slice statistics of image cube |
| `image_stats.c` | Basic image statistics (mean, RMS, min, max) |
| `imagemon.c` | Interactive image monitor |
| `improfile.c` | Radial profile of image |
| `kbdhit.c` | Non-blocking keyboard hit detection |
| `percentile.c` | Compute image percentiles |
| `print_header.c` | Print image/stream header info |
| `stream_monproc.c` | Multi-level time-binned stream monitor with histogram |
| `streamtiming_stats.c` | Stream timing jitter statistics |
| `timediff.c` | Compute timestamp differences |

## Standalone Executables

| Executable | Source File | Description |
|------------|-------------|-------------|
| `milk-fpsexec-info-imagemon` | `imagemon.c` | Interactive image monitor |
| `milk-fpsexec-info-strmonproc` | `stream_monproc.c` | Multi-level time-binned stream monitor |

## Dependencies

- `CLIcore` (includes transitive: `milkfps`, `ImageStreamIO`, `milkdata`)
