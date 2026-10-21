# Module: statistic

Statistics functions and tools, including random number
generation (uniform, Gaussian, Poisson), photon noise
simulation, and BIRCH clustering.

## Source Files

| File          | Description                                                                  |
| ------------- | ---------------------------------------------------------------------------- |
| `statistic.c` | Random number generators, Poisson/Gaussian noise injection, BIRCH clustering |

## Dependencies

- `milk_runnable.h` (`libcomputeunit`), which provides the FPS and processinfo macros
- `COREMOD_memory`
