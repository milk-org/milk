# Module: fft

Fourier Transforms

## Source Files

| File | Description |
|------|-------------|
| `DFT.c` | Use DFT to insert Focal Plane Mask |
| `dofft.c` | Dofft module |
| `fft_autocorrelation.c` | Compute autocorrelation using FFT |
| `fft_structure_function.c` | Compute structure function using FFT |
| `fftcorrelation.c` | Fftcorrelation module |
| `ffttranslate.c` | Ffttranslate module |
| `fftzoom.c` | Fftzoom module |
| `init_fftwplan.c` | Init fftwplan module |
| `permut.c` | Permut module |
| `pup2foc.c` | Pup2foc module |
| `testfftspeed.c` | Test FFT speed (fftw) |
| `wisdom.c` | Wisdom module |

## Standalone Executables

| Executable | Source File | Description |
|------------|-------------|-------------|
| `milk-fpsexec-fft-dofft` | `dofft.c` | Dofft module |
| `milk-fpsexec-fft-pup2foc` | `pup2foc.c` | Pup2foc module |

## Dependencies
- Implicit standard: `milkdata`, `ImageStreamIO`, `CLIcore`
