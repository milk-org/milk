# Module: image_filter

Image filtering and convolution

## Source Files

| File                         | Description                                     |
| ---------------------------- | ----------------------------------------------- |
| `cubepercentile.c`           | Compute per-pixel percentile across cube slices |
| `fconvolve.c`                | Fourier-based 2D convolution                    |
| `fit1D.c`                    | 1D polynomial/function fitting                  |
| `fit2DcosKernel.c`           | Fit 2D cosine kernel to image                   |
| `fit2Dcossin.c`              | Fit 2D cosine+sine basis to image               |
| `gaussfilter.c`              | Gaussian 2D image smoothing                     |
| `im2Dfilter_1pixbblurr.c`    | Apply 1-pixel radius box blur                   |
| `medianfilter.c`             | Median filter (spatial)                         |
| `percentile_interpolation.c` | Percentile-based pixel interpolation            |

## Standalone Executables

| Executable                           | Source File               | Description                        |
| ------------------------------------ | ------------------------- | ---------------------------------- |
| `milk-fpsexec-imgfilt-gaussfilt`     | `gaussfilter.c`           | Gaussian 2D image filtering        |
| `milk-fpsexec-imgfilt-im2Dfilt1pxbb` | `im2Dfilter_1pixbblurr.c` | Apply 1 pixel radius blur to image |

## Dependencies

- `CLIcore` (includes transitive: `milkfps`, `ImageStreamIO`, `milkdata`)
