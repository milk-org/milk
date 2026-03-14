# Module: image_format

Read and write images, supports several image formats.

## Source Files

| File | Description |
|------|-------------|
| `CR2toFITS.c` | Convert Canon CR2 raw files to FITS |
| `CR2tomov.c` | Convert CR2 time series to movie stream |
| `FITS_to_floatbin_lock.c` | Convert FITS to locked float32 binary |
| `FITS_to_ushortintbin_lock.c` | Convert FITS to locked uint16 binary |
| `FITStorgbFITSsimple.c` | Convert FITS to simple RGB FITS |
| `combineHDR.c` | Combine multi-exposure HDR image stack |
| `extract_RGGBchan.c` | Extract Bayer RGGB color channels |
| `extract_utr.c` | CDS/UTR image processing loop for CRED streams |
| `imtoASCII.c` | Export image pixels to ASCII text |
| `loadCR2toFITSRGB.c` | Load CR2 directly as RGB FITS |
| `readPGM.c` | Read PGM image format |
| `read_binary32f.c` | Read raw float32 binary file |
| `stream_temporal_stats.c` | Publish temporal average and stdev of stream |
| `writeBMP.c` | Write image to BMP format |

## Standalone Executables

| Executable | Source File | Description |
|------------|-------------|-------------|
| `milk-fpsexec-imgfmt-combineHDR` | `combineHDR.c` | Combine multi-exposure HDR image stack |
| `milk-fpsexec-imgfmt-extractRGGB` | `extract_RGGBchan.c` | Extract Bayer RGGB color channels |
| `milk-fpsexec-imgfmt-extractutr` | `extract_utr.c` | CDS/UTR processing for CRED streams |
| `milk-fpsexec-imgfmt-strmtempstat` | `stream_temporal_stats.c` | Publish temporal stats of image stream |
| `milk-fpsexec-imgfmt-writeBMP` | `writeBMP.c` | Write image to BMP format |

## Dependencies

- `CLIcore` (includes transitive: `milkfps`, `ImageStreamIO`, `milkdata`)
