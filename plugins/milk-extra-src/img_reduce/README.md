# Module: img_reduce

Image analysis and reduction routines for astronomy,
including bad pixel cleaning, cube statistics, image
centering and normalization, and correlation analysis.

## Source Files

| File           | Description                                                                           |
| -------------- | ------------------------------------------------------------------------------------- |
| `img_reduce.c` | Bad pixel removal, cube statistics, image centering/normalization, correlation matrix |

## Dependencies

- `CLIcore` (includes transitive: `milkfps`, `ImageStreamIO`, `milkdata`)
- `COREMOD_arith`, `COREMOD_iofits`, `COREMOD_memory`
- `fft`, `image_filter`
