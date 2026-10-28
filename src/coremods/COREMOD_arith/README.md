# Module: COREMOD_arith

Image arithmetic, SIMD block kernels, statistical reductions, coordinates, complex math, and BLAS linear algebra.

## Source Files

| File                          | Description                                                            |
| ----------------------------- | ---------------------------------------------------------------------- |
| `im_complex.c`                | Complex image conversions, unary/binary ops, and parts extraction      |
| `image_coord.c` / `.h`        | Coordinate (`ix`, `iy`, `ir`, `itheta`) and PRNG block generators      |
| `image_linalg.c` / `.h`       | Tensor shape (`transpose`, `reshape`) and BLAS Level 1-3 kernels       |
| `image_elem_block.h`          | Shared inline SIMD block kernels and 10-datatype block load/store      |
| `image_medfilt_block.h`       | Shared 3x3, 5x5, and 7x7 Young-tableau SIMD median filter kernels      |
| `milk_calc_ext.h`             | Public C SDK for user-defined `milk.calc` / `@calc` kernels & ops      |
| `image_arith__Cim_Cim__Cim.c` | Legacy complex-to-polar/amph/reim conversion wrappers                  |
| `image_arith__im__im.c`       | Unary image-to-image math functions                                    |
| `image_arith__im_f__im.c`     | Binary image-scalar arithmetic and comparison functions                |
| `image_arith__im_f_f__im.c`   | Ternary image-scalar-scalar (`trunc`/clamp) functions                  |
| `image_arith__im_im__im.c`    | Binary image-image arithmetic and comparison functions                 |
| `image_crop.c`                | Sub-image 2D and 3D cropping functions                                 |
| `image_cropmask.c`            | Masked stream pixel gather (`cropmask`)                                |
| `image_merge3D.c`             | Merge images along an axis (`immerge`)                                 |
| `image_multicrop2D.c`         | Multi-window 2D cropping from stream (`multicrop2D`)                   |
| `image_stats.c` / `.h`        | Whole-image, axis-wise, and masked Chan statistical reductions         |
| `imfunctions.c` / `.h`        | Block-vectorized unary/binary/in-place dispatcher and 2D median filter |
| `mathfuncs.h`                 | Inline scalar and float32 math wrappers                                |

## Standalone Executables

| Executable                       | Source File           | Description                          |
| -------------------------------- | --------------------- | ------------------------------------ |
| `milk-fpsexec-arith-multicrop2D` | `image_multicrop2D.c` | Multi-window 2D cropping from stream |
| `milk-fpsexec-arith-immerge`     | `image_merge3D.c`     | Merge images along an axis           |
| `milk-fpsexec-arith-cropmask`    | `image_cropmask.c`    | Masked stream pixel gather           |

## Dependencies

- `milk_runnable.h` (`libcomputeunit`), which provides the FPS and processinfo macros
