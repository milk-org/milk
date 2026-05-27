# Module: psf

Point Spread Function (PSF) analysis, including chromatic
PSF generation, centroiding, FWHM measurement, encircled
energy computation, and PSF sequence analysis.

## Source Files

| File    | Description                                                                                                                                          |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `psf.c` | Chromatic PSF generation, disk-center finding, photocenter measurement, FWHM measurement, encircled energy, PSF centering, and PSF sequence analysis |

## Dependencies

- `CLIcore` (includes transitive: `milkfps`, `ImageStreamIO`, `milkdata`)
- `COREMOD_arith`, `COREMOD_iofits`, `COREMOD_memory`, `COREMOD_tools`
- `fft`, `image_basic`, `image_filter`, `image_gen`
