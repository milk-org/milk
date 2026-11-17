# Module: COREMOD_iofits

I/O routines for FITS format

## Source Files

| File                    | Description                                                     |
| ----------------------- | --------------------------------------------------------------- |
| `COREMOD_iofits.c`      | Module registry (`MILK_MODULE_RUNNABLES`)                       |
| `breakcube.c`           | Split a 3D image cube into individual 2D slice images           |
| `check_fitsio_status.c` | Report CFITSIO error status                                     |
| `data_type_code.c`      | Map between `ImageStreamIO` and CFITSIO datatype codes          |
| `file_exists.c`         | Check whether a file exists on disk                             |
| `images2cube.c`         | Combine individual 2D images into a 3D cube                     |
| `is_fits_file.c`        | Check whether a path is a readable FITS file                    |
| `loadfits.c`            | Load FITS images/cubes (`.fits`, gzip `.fits.gz`, Rice `.fits.fz`) |
| `loadmemstream.c`       | Load FITS file directly into a shared-memory stream             |
| `read_keyword.c`        | Read FITS header keywords                                       |
| `savefits.c`            | Save image or stream to a FITS file                             |

## Standalone Executables

| Executable                      | Source File     | Description                                                        |
| ------------------------------- | --------------- | ------------------------------------------------------------------ |
| `milk-fpsexec-iofits-saveFITS`  | `savefits.c`    | Save image as a FITS file                                          |
| `milk-fpsexec-iofits-loadfits`  | `loadfits.c`    | Load FITS file (`.fits`, `.fits.gz`, `.fits.fz`) into memory/SHM   |
| `milk-fpsexec-iofits-imgs2cube` | `images2cube.c` | Combine individual 2D images into a 3D cube                        |

## Dependencies

- `milk_runnable.h` (`libcomputeunit`), which provides the FPS and processinfo macros
