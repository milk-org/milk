# Module: image_filter

Image filtering and convolution

## Source Files

| File | Description |
|------|-------------|
| `cubepercentile.c` | No description available. |
| `fconvolve.c` | No description available. |
| `fit1D.c` | No description available. |
| `fit2DcosKernel.c` | No description available. |
| `fit2Dcossin.c` | No description available. |
| `gaussfilter.c` | Gaussian 2D image filtering |
| `im2Dfilter_1pixbblurr.c` | Apply 1 pixel radius blurr to image |
| `medianfilter.c` | No description available. |
| `percentile_interpolation.c` | No description available. |

## Standalone Executables

| Executable | Source File | Description |
|------------|-------------|-------------|
| `milk-fpsexec-imgfilt-gaussfilt` | `gaussfilter.c` | Gaussian 2D image filtering |
| `milk-fpsexec-imgfilt-im2Dfilt1pxbb` | `im2Dfilter_1pixbblurr.c` | Apply 1 pixel radius blurr to image |

## Dependencies
- Implicit standard: `milkdata`, `ImageStreamIO`, `CLIcore`
