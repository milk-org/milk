/** @file magma_compute_SVDpseudoInverse_SVD.h
 */

#ifdef HAVE_CUDA

errno_t magma_compute_SVDpseudoInverse_SVD_addCLIcmd();

int CUDACOMP_magma_compute_SVDpseudoInverse_SVD(const char *ID_Rmatrix_name,
        const char *ID_Cmatrix_name,
        double      SVDeps,
        long        MaxNBmodes,
        const char *ID_VTmatrix_name);

#endif
