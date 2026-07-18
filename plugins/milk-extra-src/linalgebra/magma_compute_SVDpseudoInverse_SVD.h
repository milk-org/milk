/**
 * @file magma_compute_SVDpseudoInverse_SVD.h
 * @brief Magma compute svdpseudoinverse svd module
 */

MILK_WEAK errno_t magma_compute_SVDpseudoInverse_SVD_addCLIcmd() {};

MILK_WEAK int LINALGEBRA_magma_compute_SVDpseudoInverse_SVD(const char *ID_Rmatrix_name,
                                                            const char *ID_Cmatrix_name,
                                                            double      SVDeps,
                                                            long        MaxNBmodes,
                                                            const char *ID_VTmatrix_name)
    MILK_WEAK_FUNCDEF;
