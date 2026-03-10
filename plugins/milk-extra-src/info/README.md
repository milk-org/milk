# Module: info

Images information

## Source Files

| File | Description |
|------|-------------|
| `cubeMatchMatrix.c` | Cubematchmatrix module |
| `cubestats.c` | Cubestats module |
| `image_stats.c` | Image stats module |
| `imagemon.c` | image monitor |
| `improfile.c` | Improfile module |
| `kbdhit.c` | Kbdhit module |
| `percentile.c` | Percentile module |
| `print_header.c` | Print header module |
| `stream_monproc.c` | monitor stream with multi-level time binning, circular buffer, and dynamic histogram |
| `streamtiming_stats.c` | Streamtiming stats module |
| `timediff.c` | Timediff module |

## Standalone Executables

| Executable | Source File | Description |
|------------|-------------|-------------|
| `milk-fpsexec-info-imagemon` | `imagemon.c` | image monitor |
| `milk-fpsexec-info-strmonproc` | `stream_monproc.c` | monitor stream with multi-level time binning, circular buffer, and dynamic histogram |

## Dependencies
- `CLIcore` (includes transitive: `milkfps`, `ImageStreamIO`, `milkdata`)
