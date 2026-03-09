# Module: linopt_imtools

Linear analsys of images.

## Source Files

| File | Description |
|------|-------------|
| `compute_SVDdecomp.c` | SVD via eigenvalue decomposition |
| `compute_SVDpseudoInverse.c` | Compute pseudoinverse via eigenvalue |
| `image_construct.c` | Image construct module |
| `image_fitModes.c` | Decompose image as linear sum |
| `image_to_vec.c` | Image to vec module |
| `imcube_crossproduct.c` | Compute product between two image cubes |
| `lin1Dfit.c` | Lin1dfit module |
| `linRM_from_inout.c` | Linrm from inout module |
| `makeCPAmodes.c` | log all debug trace points to file |
| `makeCosRadModes.c` | Makecosradmodes module |
| `mask_to_pixtable.c` | Mask to pixtable module |
| `vec_to_2Dimage.c` | Vec to 2dimage module |

## Standalone Executables

| Executable | Source File | Description |
|------------|-------------|-------------|
| `milk-fpsexec-linopt-SVDdecomp` | `compute_SVDdecomp.c` | SVD via eigenvalue decomposition |
| `milk-fpsexec-linopt-SVDpInverse` | `compute_SVDpseudoInverse.c` | Compute pseudoinverse via eigenvalue |
| `milk-fpsexec-linopt-imgconstruct` | `image_construct.c` | Image construct module |
| `milk-fpsexec-linopt-imgFitModes` | `image_fitModes.c` | Decompose image as linear sum |
| `milk-fpsexec-linopt-img2vec` | `image_to_vec.c` | Image to vec module |
| `milk-fpsexec-linopt-imcXprod` | `imcube_crossproduct.c` | Compute product between two image cubes |
| `milk-fpsexec-linopt-lin1Dfit` | `lin1Dfit.c` | Lin1dfit module |
| `milk-fpsexec-linopt-linRM` | `linRM_from_inout.c` | Linrm from inout module |
| `milk-fpsexec-linopt-mkCPAmodes` | `makeCPAmodes.c` | log all debug trace points to file |
| `milk-fpsexec-linopt-mkCosRadModes` | `makeCosRadModes.c` | Makecosradmodes module |
| `milk-fpsexec-linopt-mask2pixtab` | `mask_to_pixtable.c` | Mask to pixtable module |
| `milk-fpsexec-linopt-vec2Dimg` | `vec_to_2Dimage.c` | Vec to 2dimage module |

## Dependencies
- Implicit standard: `milkdata`, `ImageStreamIO`, `CLIcore`
