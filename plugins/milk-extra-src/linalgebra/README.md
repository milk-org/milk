# Module: linalgebra

Light interface to linea algebra libs (BLAS, CUDA and MAGMA)

## Source Files

| File | Description |
|------|-------------|
| `GPU_SVD_computeControlMatrix.c` | No description available. |
| `GPU_loop_MultMat_execute.c` | No description available. |
| `GPU_loop_MultMat_free.c` | No description available. |
| `GPU_loop_MultMat_setup.c` | No description available. |
| `GPUloadCmat.c` | No description available. |
| `GramSchmidt.c` | No description available. |
| `MVM_CPU.c` | No description available. |
| `MVMextractModes.c` | No description available. |
| `PCAmatch.c` | match two PCA decompositions |
| `Qexpand.c` | match two PCA decompositions |
| `SGEMM.c` | Computes the single-precision general matrix multiplication (SGEMM) of two matrices. |
| `SingularValueDecomp.c` | Compute SVD of indimM x indimN matrix |
| `SingularValueDecomp_mkM.c` | make M from U, S, and V |
| `SingularValueDecomp_mkU.c` | make U from M, V and S |
| `basis_rotate_match.c` | No description available. |
| `cublas_Coeff2Map_Loop.c` | No description available. |
| `cublas_PCA.c` | No description available. |
| `cublas_linalgebra_MVMextractModesLoop.c` | CUDA functions wrapper |
| `cublas_linalgebratest.c` | No description available. |
| `linalgebrainit.c` | Initialize CUDA and MAGMA |
| `magma_MatMatMult_testPseudoInverse.c` | Test pseudo inverse |
| `magma_compute_SVDpseudoInverse.c` | Computes matrix pseudo-inverse (AT A)^-1 AT, using eigenvector/eigenvalue decomposition of AT A |
| `magma_compute_SVDpseudoInverse_SVD.c` | No description available. |
| `modalremap.c` | Use mapping between two spaces to remap input |
| `printGPUMATMULTCONF.c` | No description available. |

## Standalone Executables

| Executable | Source File | Description |
|------------|-------------|-------------|
| `milk-fpsexec-linalg-gramschmidt` | `GramSchmidt.c` | No description available. |
| `milk-fpsexec-linalg-MVMextract` | `MVMextractModes.c` | No description available. |
| `milk-fpsexec-linalg-PCAmatch` | `PCAmatch.c` | match two PCA decompositions |
| `milk-fpsexec-linalg-Qexpand` | `Qexpand.c` | match two PCA decompositions |
| `milk-fpsexec-linalg-SGEMM` | `SGEMM.c` | Computes the single-precision general matrix multiplication (SGEMM) of two matrices. |
| `milk-fpsexec-linalg-SVD` | `SingularValueDecomp.c` | Compute SVD of indimM x indimN matrix |
| `milk-fpsexec-linalg-SVDmkM` | `SingularValueDecomp_mkM.c` | make M from U, S, and V |
| `milk-fpsexec-linalg-SVDmkU` | `SingularValueDecomp_mkU.c` | make U from M, V and S |
| `milk-fpsexec-linalg-basisrotmatch` | `basis_rotate_match.c` | No description available. |
| `milk-fpsexec-linalg-cublasPCA` | `cublas_PCA.c` | No description available. |
| `milk-fpsexec-linalg-modalremap` | `modalremap.c` | Use mapping between two spaces to remap input |

## Dependencies
- Implicit standard: `milkdata`, `ImageStreamIO`, `CLIcore`
