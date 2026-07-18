/**
 * @file cublas_Coeff2Map_Loop.h
 * @brief Cublas coeff2map loop module
 */

/** @file Coeff2Map_Loop.h
 */


MILK_WEAK errno_t Coeff2Map_Loop_addCLIcmd() {};


MILK_WEAK errno_t LINALGEBRA_Coeff2Map_Loop(const char *IDmodes_name,
                                            const char *IDcoeff_name,
                                            int         GPUindex,
                                            const char *IDoutmap_name,
                                            int         offsetmode,
                                            const char *IDoffset_name) MILK_WEAK_FUNCDEF;
