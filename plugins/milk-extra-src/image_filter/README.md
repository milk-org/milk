# Module: image_filter

Image filtering and convolution

## Source Files

| File | Description |
|------|-------------|
| `cubepercentile.c` | Cubepercentile module |
| `fconvolve.c` | Fconvolve module |
| `fit1D.c` | Fit1d module |
| `fit2DcosKernel.c` | Fit2dcoskernel module |
| `fit2Dcossin.c` | Fit2dcossin module |
| `gaussfilter.c` | Gaussian 2D image filtering |
| `im2Dfilter_1pixbblurr.c` | Apply 1 pixel radius blurr to image |
| `medianfilter.c` | Medianfilter module |
| `percentile_interpolation.c` | Percentile interpolation module |

## Standalone Executables

| Executable | Source File | Description |
|------------|-------------|-------------|
| `milk-fpsexec-imgfilt-gaussfilt` | `gaussfilter.c` | Gaussian 2D image filtering |
| `milk-fpsexec-imgfilt-im2Dfilt1pxbb` | `im2Dfilter_1pixbblurr.c` | Apply 1 pixel radius blurr to image |

## Dependencies
- `CLIcore` (includes transitive: `milkfps`, `ImageStreamIO`, `milkdata`)
