# Module: COREMOD_iofits

I/O routines for FITS format

## Source Files

| File | Description |
|------|-------------|
| `breakcube.c` | No description available. |
| `check_fitsio_status.c` | No description available. |
| `data_type_code.c` | No description available. |
| `file_exists.c` | No description available. |
| `images2cube.c` | No description available. |
| `is_fits_file.c` | No description available. |
| `loadfits.c` | load FITS format files |
| `loadmemstream.c` | load memory stream |
| `read_keyword.c` | No description available. |
| `savefits.c` | Save image to FITS file |

## Standalone Executables

| Executable | Source File | Description |
|------------|-------------|-------------|
| `milk-fpsexec-iofits-saveFITS` | `savefits.c` | Save image to FITS file |
| `milk-fpsexec-iofits-loadfits` | `loadfits.c` | load FITS format files |
| `milk-fpsexec-iofits-imgs2cube` | `images2cube.c` | No description available. |

## Dependencies
- Implicit standard: `milkdata`, `ImageStreamIO`, `CLIcore`
