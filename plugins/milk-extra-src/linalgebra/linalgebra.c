// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    linalgebra.c
 * @brief   Linear Algebra functions wrapper
 *
 *
 */

#define MODULE_SHORTNAME_DEFAULT "linalg"
#define MODULE_DESCRIPTION "linear algebra"

#ifdef HAVE_CUDA
#    include <cublas_v2.h>
#endif

#ifdef HAVE_MAGMA
#    include "magma_lapack.h"
#    include "magma_v2.h"
#endif

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "linalgebra_types.h"

// Units of the module. Units of files dropped by an unmet MILK_CMAKE_MANDATE_* are weak (NULL).
extern const MILK_RUNNABLE runnable_linalgebra_basisrotmatch;
extern const MILK_RUNNABLE runnable_linalgebra_GramSchmidt;
extern const MILK_RUNNABLE runnable_linalgebra_MVMextract;
extern const MILK_RUNNABLE runnable_linalgebra_PCAmatch;
extern const MILK_RUNNABLE runnable_linalgebra_Qexpand;
extern const MILK_RUNNABLE runnable_linalgebra_SGEMM;
extern const MILK_RUNNABLE runnable_linalgebra_SVDmkM;
extern const MILK_RUNNABLE runnable_linalgebra_SVDmkU;
extern const MILK_RUNNABLE runnable_linalgebra_ModalRemap;
MILK_RUNNABLE_WEAK_DECL(runnable_linalgebra_PCAdecomp);
MILK_RUNNABLE_WEAK_DECL(runnable_linalgebra_cudacoeff2map);
MILK_RUNNABLE_WEAK_DECL(runnable_linalgebra_cudacoeffo2map);
MILK_RUNNABLE_WEAK_DECL(runnable_linalgebra_linalgebrainit);
MILK_RUNNABLE_WEAK_DECL(runnable_linalgebra_linalgebrapsinv);
MILK_RUNNABLE_WEAK_DECL(runnable_linalgebra_linalgebrapsinvSVD);
MILK_RUNNABLE_WEAK_DECL(runnable_linalgebra_cudatestpsinv);
MILK_RUNNABLE_WEAK_DECL(runnable_linalgebra_compSVD);

// globals

imageID IDtimerinit = 0;
imageID IDtiming    = -1; // index to image where timing should be written

#ifdef HAVE_CUDA
int            cuda_deviceCount;
GPUMATMULTCONF gpumatmultconf[20]; // supports up to 20 cfgs per process
float          cublasSgemv_alpha = 1.0;
float          cublasSgemv_beta  = 0.0;
#endif

#ifdef HAVE_MAGMA
int           INIT_MAGMA = 0;
magma_queue_t magmaqueue;
#endif

static void __attribute__((constructor)) libinit_linalgebra_printinfo()
{
#ifdef HAVE_CUDA
    if (!getenv("MILK_QUIET"))
    {
        printf("[CUDA %d]", dcquiet);
    }

    for (int i = 0; i < 20; i++)
    {
        gpumatmultconf[i].init  = 0;
        gpumatmultconf[i].alloc = 0;
    }
#endif

#ifdef HAVE_MAGMA
    if (!getenv("MILK_QUIET"))
    {
        printf("[MAGMA]");
    }
#endif
}

// Module registry: CLI exposure = membership here. linalgebratest is not exposed (as before).
static const MILK_RUNNABLE *const runnables[] = {
    &runnable_linalgebra_linalgebrainit,
    &runnable_linalgebra_basisrotmatch,
    &runnable_linalgebra_GramSchmidt,
    &runnable_linalgebra_PCAdecomp,
    &runnable_linalgebra_cudatestpsinv,
    &runnable_linalgebra_linalgebrapsinv,
    &runnable_linalgebra_linalgebrapsinvSVD,
    &runnable_linalgebra_cudacoeff2map,
    &runnable_linalgebra_cudacoeffo2map,
    &runnable_linalgebra_MVMextract,
    &runnable_linalgebra_PCAmatch,
    &runnable_linalgebra_Qexpand,
    &runnable_linalgebra_compSVD,
    &runnable_linalgebra_SVDmkU,
    &runnable_linalgebra_SVDmkM,
    &runnable_linalgebra_SGEMM,
    &runnable_linalgebra_ModalRemap,
};

MILK_MODULE_RUNNABLES(linalgebra, NULL, runnables);


// WEAK declarations for all functions exported by the module
// These declarations are used to serve the linker in case
// of missing MILK_CMAKE_MANDATE and when the actual declaration
// does therefore not exist.

#define MW MILK_WEAK
#define MWF MILK_WEAK_FUNCDEF

errno_t compute_basis_rotate_match(IMGID imginAB, IMGID *imgArot, int optmode);

MW errno_t LINALGEBRA_Coeff2Map_Loop(const char *IDmodes_name,
                                     const char *IDcoeff_name,
                                     int         GPUindex,
                                     const char *IDoutmap_name,
                                     int         offsetmode,
                                     const char *IDoffset_name) MWF;

MW errno_t GPUcomp_test(long NBact, long NBmodes, long WFSsize, long GPUcnt) MWF;

MW int GPU_loop_MultMat_free(int index) MWF;

MW int GPU_loop_MultMat_execute(int   index,
                                int  *status,
                                int  *GPUstatus,
                                float alpha,
                                float beta,
                                int   timing,
                                int   TimerOffsetIndex) MWF;


errno_t SVDmkM(IMGID imgU, IMGID imgS, IMGID imgV, IMGID *imgM, int GPUdev);

errno_t Qexpand(IMGID incoeffM, IMGID *outcoeffM, int axis);

errno_t compute_SVDU(IMGID imgM, IMGID imgV, IMGID imgS, IMGID *imgU, IMGID *imgUS, int GPUdev);

errno_t computeSGEMM(IMGID  imginA,
                     IMGID  imginB,
                     IMGID *outimg,
                     int    TranspA,
                     int    TranspB,
                     int    GPUdev);

MW errno_t compute_SVD(IMGID    imgin,
                       IMGID   *imgU,
                       IMGID   *imgeigenval,
                       IMGID   *imgV,
                       uint32_t Vdim0,
                       float    SVlimit,
                       uint32_t SVDmaxNBmode,
                       int      GPUdev,
                       uint64_t compSVDmode,
                       char    *SVDunmodesname,
                       char    *SVDvnmodesname) MWF;

MW errno_t LINALGEBRA_printGPUMATMULTCONF(int index) MWF;

MW errno_t LINALGEBRA_magma_compute_SVDpseudoInverse(const char *ID_Rmatrix_name,
                                                     const char *ID_Cmatrix_name,
                                                     double      SVDeps,
                                                     long        MaxNBmodes,
                                                     const char *ID_VTmatrix_name,
                                                     int         LOOPmode,
                                                     int         testmode,
                                                     int         precision,
                                                     int         GPUdevice,
                                                     imageID    *outID) MWF;

errno_t PCAmatch(IMGID  imgmodesA,
                 IMGID  imgmodesB,
                 IMGID *imgoutcA,
                 IMGID *imgoutcB,
                 IMGID *imgoutimA,
                 IMGID *imgoutimB,
                 int    GPUdev);

MW long CLINALGEBRA_MatMatMult_testPseudoInverse(const char *IDmatA_name,
                                                 const char *IDmatAinv_name,
                                                 const char *IDmatOut_name) MWF;

MW int LINALGEBRA_magma_compute_SVDpseudoInverse_SVD(const char *ID_Rmatrix_name,
                                                     const char *ID_Cmatrix_name,
                                                     double      SVDeps,
                                                     long        MaxNBmodes,
                                                     const char *ID_VTmatrix_name) MWF;

MW int LINALGEBRA_init() MWF;

MW void *GPU_scanDevices(void *deviceCount_void_ptr) MWF;

errno_t GramSchmidt(IMGID imginm, IMGID *imgoutm, int GPUdev);

MW errno_t GPUloadCmat(int index) MWF;

MW errno_t GPU_SVD_computeControlMatrix(int         device,
                                        const char *ID_Rmatrix_name,
                                        const char *ID_Cmatrix_name,
                                        double      SVDeps,
                                        const char *ID_VTmatrix_name) MWF;

MW int GPU_loop_MultMat_setup(int         index,
                              const char *IDcontrM_name,
                              const char *IDwfsim_name,
                              const char *IDoutdmmodes_name,
                              long        NBGPUs,
                              int        *GPUdevice,
                              int         orientation,
                              int         USEsem,
                              int         initWFSref,
                              long        loopnb) MWF;

errno_t ModalRemap(IMGID imgM0, IMGID imgU0, IMGID imgU1, IMGID *imgM1, int GPUdev);

void    matrixMulCPU(float *cMat, float *wfsVec, float *dmVec, int M, int N);
MW void matrixMulCPU_BLAS(float *cMat, float *wfsVec, float *dmVec, int M, int N) MWF;
void    matrixMulCPU_plain_or_OPENMP(float *cMat, float *wfsVec, float *dmVec, int M, int N);

#undef MW
#undef MWF
