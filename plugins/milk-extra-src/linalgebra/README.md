# Module: linalgebra

Linear algebra operations: SVD, pseudo-inverse, matrix multiplication via BLAS, CUDA, and MAGMA (BLAS, CUDA and MAGMA)

## Source Files

| File | Description |
|------|-------------|
| `GPU_SVD_computeControlMatrix.c` | Compute control matrix via GPU-accelerated SVD |
| `GPU_loop_MultMat_execute.c` | Execute GPU matrix-vector multiply loop |
| `GPU_loop_MultMat_free.c` | Free GPU matrix multiply resources |
| `GPU_loop_MultMat_setup.c` | Set up GPU matrix multiply loop buffers |
| `GPUloadCmat.c` | Load control matrix into GPU memory |
| `GramSchmidt.c` | Gram-Schmidt orthogonalization |
| `MVM_CPU.c` | CPU matrix-vector multiplication |
| `MVMextractModes.c` | Extract modal coefficients via MVM |
| `PCAmatch.c` | Match two PCA decompositions |
| `Qexpand.c` | Expand Q matrix from PCA decomposition |
| `SGEMM.c` | Single-precision general matrix multiply (SGEMM) |
| `SingularValueDecomp.c` | Compute SVD of M×N matrix |
| `SingularValueDecomp_mkM.c` | Reconstruct M from U, S, and V |
| `SingularValueDecomp_mkU.c` | Reconstruct U from M, V, and S |
| `basis_rotate_match.c` | Rotate and match modal basis sets |
| `cublas_Coeff2Map_Loop.c` | cuBLAS coefficient-to-map streaming loop |
| `cublas_PCA.c` | PCA decomposition via cuBLAS |
| `cublas_linalgebratest.c` | cuBLAS linear algebra test/benchmark |
| `linalgebrainit.c` | Initialize CUDA and MAGMA contexts |
| `magma_MatMatMult_testPseudoInverse.c` | Test pseudo-inverse via MAGMA MatMatMult |
| `magma_compute_SVDpseudoInverse.c` | Pseudo-inverse via eigenvalue decomposition (MAGMA) |
| `magma_compute_SVDpseudoInverse_SVD.c` | Pseudo-inverse via full SVD (MAGMA) |
| `modalremap.c` | Remap input using inter-space mapping |
| `printGPUMATMULTCONF.c` | Print GPU matrix multiply configuration |

## Standalone Executables

| Executable | Source File | Description |
|------------|-------------|-------------|
| `milk-fpsexec-linalg-gramschmidt` | `GramSchmidt.c` | Gram-Schmidt orthogonalization |
| `milk-fpsexec-linalg-MVMextract` | `MVMextractModes.c` | Extract modal coefficients via MVM |
| `milk-fpsexec-linalg-PCAmatch` | `PCAmatch.c` | Match two PCA decompositions |
| `milk-fpsexec-linalg-Qexpand` | `Qexpand.c` | Expand Q matrix from PCA decomposition |
| `milk-fpsexec-linalg-SGEMM` | `SGEMM.c` | Single-precision general matrix multiply |
| `milk-fpsexec-linalg-SVD` | `SingularValueDecomp.c` | Compute SVD of M×N matrix |
| `milk-fpsexec-linalg-SVDmkM` | `SingularValueDecomp_mkM.c` | Reconstruct M from U, S, and V |
| `milk-fpsexec-linalg-SVDmkU` | `SingularValueDecomp_mkU.c` | Reconstruct U from M, V, and S |
| `milk-fpsexec-linalg-basisrotmatch` | `basis_rotate_match.c` | Rotate and match modal basis sets |
| `milk-fpsexec-linalg-cublasPCA` | `cublas_PCA.c` | PCA decomposition via cuBLAS |
| `milk-fpsexec-linalg-modalremap` | `modalremap.c` | Remap input using inter-space mapping |

## Dependencies

- `CLIcore` (includes transitive: `milkfps`, `ImageStreamIO`, `milkdata`)
