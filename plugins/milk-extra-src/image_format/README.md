# Module: image_format

Read and write images, supports several image formats.

## Source Files

| File | Description |
|------|-------------|
| `CR2toFITS.c` | Cr2tofits module |
| `CR2tomov.c` | Cr2tomov module |
| `FITS_to_floatbin_lock.c` | ========================================== |
| `FITS_to_ushortintbin_lock.c` | ========================================== |
| `FITStorgbFITSsimple.c` | Fitstorgbfitssimple module |
| `combineHDR.c` | Combinehdr module |
| `extract_RGGBchan.c` | Wrapper function, used by all CLI calls |
| `extract_utr.c` | CDS (correlated double sampling) + UTR (sample up-the-ramp) image processing loop for CRED streams |
| `imtoASCII.c` | ========================================== |
| `loadCR2toFITSRGB.c` | Loadcr2tofitsrgb module |
| `readPGM.c` | Readpgm module |
| `read_binary32f.c` | ========================================== |
| `stream_temporal_stats.c` | Publishes average and standard dev of image stream at regular intervals |
| `writeBMP.c` | Writebmp module |

## Standalone Executables

| Executable | Source File | Description |
|------------|-------------|-------------|
| `milk-fpsexec-imgfmt-combineHDR` | `combineHDR.c` | Combinehdr module |
| `milk-fpsexec-imgfmt-extractRGGB` | `extract_RGGBchan.c` | Wrapper function, used by all CLI calls |
| `milk-fpsexec-imgfmt-extractutr` | `extract_utr.c` | CDS (correlated double sampling) + UTR (sample up-the-ramp) image processing loop for CRED streams |
| `milk-fpsexec-imgfmt-strmtempstat` | `stream_temporal_stats.c` | Publishes average and standard dev of image stream at regular intervals |
| `milk-fpsexec-imgfmt-writeBMP` | `writeBMP.c` | Writebmp module |

## Dependencies
- `CLIcore` (includes transitive: `milkfps`, `ImageStreamIO`, `milkdata`)
