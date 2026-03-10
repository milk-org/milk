# Module: linalgebra

Linear algebra operations: SVD, pseudo-inverse, matrix multiplication via BLAS, CUDA, and MAGMA (BLAS, CUDA and MAGMA)

## Source Files

| File | Description |
|------|-------------|
| `GPU_SVD_computeControlMatrix.c` | Gpu svd computecontrolmatrix module |
| `GPU_loop_MultMat_execute.c` | Gpu loop multmat execute module |
| `GPU_loop_MultMat_free.c` | Gpu loop multmat free module |
| `GPU_loop_MultMat_setup.c` | Gpu loop multmat setup module |
| `GPUloadCmat.c` | Gpuloadcmat module |
| `GramSchmidt.c` | Gramschmidt module |
| `MVM_CPU.c` | Mvm cpu module |
| `MVMextractModes.c` | Mvmextractmodes module |
| `PCAmatch.c` | match two PCA decompositions |
| `Qexpand.c` | match two PCA decompositions |
| `SGEMM.c` | Computes the single-precision general matrix multiplication (SGEMM) of two matrices. |
| `SingularValueDecomp.c` | Compute SVD of indimM x indimN matrix |
| `SingularValueDecomp_mkM.c` | make M from U, S, and V |
| `SingularValueDecomp_mkU.c` | make U from M, V and S |
| `basis_rotate_match.c` | Basis rotate match module |
| `cublas_Coeff2Map_Loop.c` | Cublas coeff2map loop module |
| `cublas_PCA.c` | Cublas pca module |
| `cublas_linalgebra_MVMextractModesLoop.c` | CUDA functions wrapper |
| `cublas_linalgebratest.c` | Cublas linalgebratest module |
| `linalgebrainit.c` | Initialize CUDA and MAGMA |
| `magma_MatMatMult_testPseudoInverse.c` | Test pseudo inverse |
| `magma_compute_SVDpseudoInverse.c` | Computes matrix pseudo-inverse (AT A)^-1 AT, using eigenvector/eigenvalue decomposition of AT A |
| `magma_compute_SVDpseudoInverse_SVD.c` | Magma compute svdpseudoinverse svd module |
| `modalremap.c` | Use mapping between two spaces to remap input |
| `printGPUMATMULTCONF.c` | Printgpumatmultconf module |

## Standalone Executables

| Executable | Source File | Description |
|------------|-------------|-------------|
| `milk-fpsexec-linalg-gramschmidt` | `GramSchmidt.c` | Gramschmidt module |
| `milk-fpsexec-linalg-MVMextract` | `MVMextractModes.c` | Mvmextractmodes module |
| `milk-fpsexec-linalg-PCAmatch` | `PCAmatch.c` | match two PCA decompositions |
| `milk-fpsexec-linalg-Qexpand` | `Qexpand.c` | match two PCA decompositions |
| `milk-fpsexec-linalg-SGEMM` | `SGEMM.c` | Computes the single-precision general matrix multiplication (SGEMM) of two matrices. |
| `milk-fpsexec-linalg-SVD` | `SingularValueDecomp.c` | Compute SVD of indimM x indimN matrix |
| `milk-fpsexec-linalg-SVDmkM` | `SingularValueDecomp_mkM.c` | make M from U, S, and V |
| `milk-fpsexec-linalg-SVDmkU` | `SingularValueDecomp_mkU.c` | make U from M, V and S |
| `milk-fpsexec-linalg-basisrotmatch` | `basis_rotate_match.c` | Basis rotate match module |
| `milk-fpsexec-linalg-cublasPCA` | `cublas_PCA.c` | Cublas pca module |
| `milk-fpsexec-linalg-modalremap` | `modalremap.c` | Use mapping between two spaces to remap input |

## Dependencies
- `CLIcore` (includes transitive: `milkfps`, `ImageStreamIO`, `milkdata`)
