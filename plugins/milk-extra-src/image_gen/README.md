# Module: image_gen

Creating images (shapes, useful functions, and patterns).

## Source Files

| File           | Description                            |
| -------------- | -------------------------------------- |
| `mkrandomim.c` | Generate random-noise images           |
| `voronoi.c`    | Generate Voronoi tessellation patterns |

## Standalone Executables

| Executable                     | Source File    | Description                            |
| ------------------------------ | -------------- | -------------------------------------- |
| `milk-fpsexec-imggen-mkrandom` | `mkrandomim.c` | Generate random-noise images           |
| `milk-fpsexec-imggen-voronoi`  | `voronoi.c`    | Generate Voronoi tessellation patterns |

## Dependencies

- `milk_runnable.h` (`libcomputeunit`), which provides the FPS and processinfo macros
