# Module: linopt_imtools

Image linear decomposition and optimization tools.

## Source Files

| File                         | Description                                         |
| ---------------------------- | --------------------------------------------------- |
| `compute_SVDdecomp.c`        | SVD via eigenvalue decomposition                    |
| `compute_SVDpseudoInverse.c` | Compute pseudo-inverse via eigenvalue decomposition |
| `image_construct.c`          | Reconstruct image from modal coefficients           |
| `image_fitModes.c`           | Decompose image as linear sum of modes              |
| `image_to_vec.c`             | Flatten 2D image to 1D vector                       |
| `imcube_crossproduct.c`      | Cross-product between two image cubes               |
| `lin1Dfit.c`                 | Linear 1D curve fitting                             |
| `linRM_from_inout.c`         | Compute response matrix from input/output pairs     |
| `makeCPAmodes.c`             | Generate Cycles Per Aperture modal basis            |
| `makeCosRadModes.c`          | Generate cosine-radial modal basis                  |
| `mask_to_pixtable.c`         | Convert binary mask to pixel coordinate table       |
| `vec_to_2Dimage.c`           | Reshape 1D vector to 2D image                       |

## Standalone Executables

| Executable                          | Source File                  | Description                            |
| ----------------------------------- | ---------------------------- | -------------------------------------- |
| `milk-fpsexec-linopt-SVDdecomp`     | `compute_SVDdecomp.c`        | SVD via eigenvalue decomposition       |
| `milk-fpsexec-linopt-SVDpInverse`   | `compute_SVDpseudoInverse.c` | Compute pseudo-inverse                 |
| `milk-fpsexec-linopt-imgconstruct`  | `image_construct.c`          | Reconstruct image from coefficients    |
| `milk-fpsexec-linopt-imgFitModes`   | `image_fitModes.c`           | Decompose image as linear sum of modes |
| `milk-fpsexec-linopt-img2vec`       | `image_to_vec.c`             | Flatten 2D image to 1D vector          |
| `milk-fpsexec-linopt-imcXprod`      | `imcube_crossproduct.c`      | Cross-product between image cubes      |
| `milk-fpsexec-linopt-lin1Dfit`      | `lin1Dfit.c`                 | Linear 1D curve fitting                |
| `milk-fpsexec-linopt-linRM`         | `linRM_from_inout.c`         | Compute response matrix                |
| `milk-fpsexec-linopt-mkCPAmodes`    | `makeCPAmodes.c`             | Generate CPA modal basis               |
| `milk-fpsexec-linopt-mkCosRadModes` | `makeCosRadModes.c`          | Generate cosine-radial modes           |
| `milk-fpsexec-linopt-mask2pixtab`   | `mask_to_pixtable.c`         | Convert mask to pixel table            |
| `milk-fpsexec-linopt-vec2Dimg`      | `vec_to_2Dimage.c`           | Reshape vector to 2D image             |

## Dependencies

- `CLIcore` (includes transitive: `milkfps`, `ImageStreamIO`, `milkdata`)
