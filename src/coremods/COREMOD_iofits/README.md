# Module: COREMOD_iofits

I/O routines for FITS format

## Source Files

| File | Description |
|------|-------------|
| `breakcube.c` | Breakcube module |
| `check_fitsio_status.c` | set print to 0 if error message should not be printed to stderr |
| `data_type_code.c` | Data type code module |
| `file_exists.c` | File exists module |
| `images2cube.c` | ========================================== |
| `is_fits_file.c` | Is fits file module |
| `loadfits.c` | load FITS format files |
| `loadmemstream.c` | load memory stream |
| `read_keyword.c` | Read keyword module |
| `savefits.c` | Save image to FITS file |

## Standalone Executables

| Executable | Source File | Description |
|------------|-------------|-------------|
| `milk-fpsexec-iofits-saveFITS` | `savefits.c` | Save image to FITS file |
| `milk-fpsexec-iofits-loadfits` | `loadfits.c` | load FITS format files |
| `milk-fpsexec-iofits-imgs2cube` | `images2cube.c` | ========================================== |

## Dependencies
- `CLIcore` (includes transitive: `milkfps`, `ImageStreamIO`, `milkdata`)
