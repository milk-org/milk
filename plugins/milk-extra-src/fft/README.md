# Module: fft

Fourier Transforms using FFTW library.

## Source Files

| File                       | Description                                               |
| -------------------------- | --------------------------------------------------------- |
| `DFT.c`                    | Discrete Fourier Transform for focal plane mask insertion |
| `dofft.c`                  | Perform 2D complex FFT on images                          |
| `fft_autocorrelation.c`    | Compute autocorrelation of an image via FFT               |
| `fft_structure_function.c` | Compute structure function via FFT                        |
| `fftcorrelation.c`         | Cross-correlate two images via FFT                        |
| `ffttranslate.c`           | Sub-pixel image translation using Fourier shift           |
| `fftzoom.c`                | Zoom / resample image using FFT                           |
| `init_fftwplan.c`          | Initialize and cache FFTW plans                           |
| `permut.c`                 | Quadrant swap (shift zero-frequency to center)            |
| `pup2foc.c`                | Pupil-to-focal-plane propagation via FFT                  |
| `testfftspeed.c`           | Benchmark FFT execution speed                             |
| `wisdom.c`                 | Manage FFTW wisdom files for plan optimization            |

## Standalone Executables

| Executable                 | Source File | Description                      |
| -------------------------- | ----------- | -------------------------------- |
| `milk-fpsexec-fft-dofft`   | `dofft.c`   | Perform 2D complex FFT           |
| `milk-fpsexec-fft-pup2foc` | `pup2foc.c` | Pupil-to-focal-plane propagation |

## Dependencies

- `CLIcore` (includes transitive: `milkfps`, `ImageStreamIO`, `milkdata`)
- `fftw3`, `fftw3f` (single and double precision)
