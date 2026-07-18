/** @file GPU_loop_MultMat_setup.c
 */

#ifdef HAVE_CUDA

/** @brief Setup memory and process for GPU-based matrix-vector multiply
 */
MILK_WEAK int GPU_loop_MultMat_setup(int         index,
                                     const char *IDcontrM_name,
                                     const char *IDwfsim_name,
                                     const char *IDoutdmmodes_name,
                                     long        NBGPUs,
                                     int        *GPUdevice,
                                     int         orientation,
                                     int         USEsem,
                                     int         initWFSref,
                                     long        loopnb) MILK_WEAK_FUNCDEF;

#endif
