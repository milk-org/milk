# Module: linopt_imtools

Linear analsys of images.

## Source Files

| File | Description |
|------|-------------|
| `compute_SVDdecomp.c` | SVD via eigenvalue decomposition |
| `compute_SVDpseudoInverse.c` | Compute pseudoinverse via eigenvalue |
| `image_construct.c` | No description available. |
| `image_fitModes.c` | Decompose image as linear sum |
| `image_to_vec.c` | No description available. |
| `imcube_crossproduct.c` | Compute product between two image cubes |
| `lin1Dfit.c` | No description available. |
| `linRM_from_inout.c` | No description available. |
| `makeCPAmodes.c` | No description available. |
| `makeCosRadModes.c` | No description available. |
| `mask_to_pixtable.c` | No description available. |
| `vec_to_2Dimage.c` | No description available. |

## Standalone Executables

| Executable | Source File | Description |
|------------|-------------|-------------|
| `milk-fpsexec-linopt-SVDdecomp` | `compute_SVDdecomp.c` | SVD via eigenvalue decomposition |
| `milk-fpsexec-linopt-SVDpInverse` | `compute_SVDpseudoInverse.c` | Compute pseudoinverse via eigenvalue |
| `milk-fpsexec-linopt-imgconstruct` | `image_construct.c` | No description available. |
| `milk-fpsexec-linopt-imgFitModes` | `image_fitModes.c` | Decompose image as linear sum |
| `milk-fpsexec-linopt-img2vec` | `image_to_vec.c` | No description available. |
| `milk-fpsexec-linopt-imcXprod` | `imcube_crossproduct.c` | Compute product between two image cubes |
| `milk-fpsexec-linopt-lin1Dfit` | `lin1Dfit.c` | No description available. |
| `milk-fpsexec-linopt-linRM` | `linRM_from_inout.c` | No description available. |
| `milk-fpsexec-linopt-mkCPAmodes` | `makeCPAmodes.c` | No description available. |
| `milk-fpsexec-linopt-mkCosRadModes` | `makeCosRadModes.c` | No description available. |
| `milk-fpsexec-linopt-mask2pixtab` | `mask_to_pixtable.c` | No description available. |
| `milk-fpsexec-linopt-vec2Dimg` | `vec_to_2Dimage.c` | No description available. |

## Dependencies
- Implicit standard: `milkdata`, `ImageStreamIO`, `CLIcore`
