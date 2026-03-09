# Module: image_format

Read and write images, supports several image formats.

## Source Files

| File | Description |
|------|-------------|
| `CR2toFITS.c` | No description available. |
| `CR2tomov.c` | No description available. |
| `FITS_to_floatbin_lock.c` | No description available. |
| `FITS_to_ushortintbin_lock.c` | No description available. |
| `FITStorgbFITSsimple.c` | No description available. |
| `combineHDR.c` | No description available. |
| `extract_RGGBchan.c` | Wrapper function, used by all CLI calls |
| `extract_utr.c` | CDS (correlated double sampling) + UTR (sample up-the-ramp) image processing loop for CRED streams |
| `imtoASCII.c` | No description available. |
| `loadCR2toFITSRGB.c` | No description available. |
| `readPGM.c` | No description available. |
| `read_binary32f.c` | No description available. |
| `stream_temporal_stats.c` | Publishes average and standard dev of image stream at regular intervals |
| `writeBMP.c` | No description available. |

## Standalone Executables

| Executable | Source File | Description |
|------------|-------------|-------------|
| `milk-fpsexec-imgfmt-combineHDR` | `combineHDR.c` | No description available. |
| `milk-fpsexec-imgfmt-extractRGGB` | `extract_RGGBchan.c` | Wrapper function, used by all CLI calls |
| `milk-fpsexec-imgfmt-extractutr` | `extract_utr.c` | CDS (correlated double sampling) + UTR (sample up-the-ramp) image processing loop for CRED streams |
| `milk-fpsexec-imgfmt-strmtempstat` | `stream_temporal_stats.c` | Publishes average and standard dev of image stream at regular intervals |
| `milk-fpsexec-imgfmt-writeBMP` | `writeBMP.c` | No description available. |

## Dependencies
- Implicit standard: `milkdata`, `ImageStreamIO`, `CLIcore`
