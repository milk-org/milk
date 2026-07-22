// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    linalgebra.h
 * @brief   Function prototypes linalgebra
 *
 */

#ifndef _LINALGEBRA_H
#define _LINALGEBRA_H

// Prototypes from all useful exported functions
// No definitions, this is for export inside and outside of the module.

errno_t compute_basis_rotate_match(IMGID imginAB, IMGID *imgArot, int optmode);

errno_t LINALGEBRA_Coeff2Map_Loop(const char *IDmodes_name,
                                  const char *IDcoeff_name,
                                  int         GPUindex,
                                  const char *IDoutmap_name,
                                  int         offsetmode,
                                  const char *IDoffset_name);

errno_t GPUcomp_test(long NBact, long NBmodes, long WFSsize, long GPUcnt);

int GPU_loop_MultMat_free(int index);

int GPU_loop_MultMat_execute(int   index,
                             int  *status,
                             int  *GPUstatus,
                             float alpha,
                             float beta,
                             int   timing,
                             int   TimerOffsetIndex);


errno_t SVDmkM(IMGID imgU, IMGID imgS, IMGID imgV, IMGID *imgM, int GPUdev);

errno_t Qexpand(IMGID incoeffM, IMGID *outcoeffM, int axis);

errno_t compute_SVDU(IMGID imgM, IMGID imgV, IMGID imgS, IMGID *imgU, IMGID *imgUS, int GPUdev);

errno_t computeSGEMM(IMGID  imginA,
                     IMGID  imginB,
                     IMGID *outimg,
                     int    TranspA,
                     int    TranspB,
                     int    GPUdev);

errno_t compute_SVD(IMGID    imgin,
                    IMGID   *imgU,
                    IMGID   *imgeigenval,
                    IMGID   *imgV,
                    uint32_t Vdim0,
                    float    SVlimit,
                    uint32_t SVDmaxNBmode,
                    int      GPUdev,
                    uint64_t compSVDmode,
                    char    *SVDunmodesname,
                    char    *SVDvnmodesname);

errno_t LINALGEBRA_printGPUMATMULTCONF(int index);

errno_t LINALGEBRA_magma_compute_SVDpseudoInverse(const char *ID_Rmatrix_name,
                                                  const char *ID_Cmatrix_name,
                                                  double      SVDeps,
                                                  long        MaxNBmodes,
                                                  const char *ID_VTmatrix_name,
                                                  int         LOOPmode,
                                                  int         testmode,
                                                  int         precision,
                                                  int         GPUdevice,
                                                  imageID    *outID);

errno_t PCAmatch(IMGID  imgmodesA,
                 IMGID  imgmodesB,
                 IMGID *imgoutcA,
                 IMGID *imgoutcB,
                 IMGID *imgoutimA,
                 IMGID *imgoutimB,
                 int    GPUdev);

errno_t CLIADDCMD_linalgebra__ModalRemap();

long CLINALGEBRA_MatMatMult_testPseudoInverse(const char *IDmatA_name,
                                              const char *IDmatAinv_name,
                                              const char *IDmatOut_name);

int LINALGEBRA_magma_compute_SVDpseudoInverse_SVD(const char *ID_Rmatrix_name,
                                                  const char *ID_Cmatrix_name,
                                                  double      SVDeps,
                                                  long        MaxNBmodes,
                                                  const char *ID_VTmatrix_name);

int LINALGEBRA_init();

void *GPU_scanDevices(void *deviceCount_void_ptr);

errno_t GramSchmidt(IMGID imginm, IMGID *imgoutm, int GPUdev);

errno_t GPUloadCmat(int index);

errno_t GPU_SVD_computeControlMatrix(int         device,
                                     const char *ID_Rmatrix_name,
                                     const char *ID_Cmatrix_name,
                                     double      SVDeps,
                                     const char *ID_VTmatrix_name);

int GPU_loop_MultMat_setup(int         index,
                           const char *IDcontrM_name,
                           const char *IDwfsim_name,
                           const char *IDoutdmmodes_name,
                           long        NBGPUs,
                           int        *GPUdevice,
                           int         orientation,
                           int         USEsem,
                           int         initWFSref,
                           long        loopnb);

errno_t ModalRemap(IMGID imgM0, IMGID imgU0, IMGID imgU1, IMGID *imgM1, int GPUdev);

void matrixMulCPU(float *cMat, float *wfsVec, float *dmVec, int M, int N);
void matrixMulCPU_BLAS(float *cMat, float *wfsVec, float *dmVec, int M, int N);
void matrixMulCPU_plain_or_OPENMP(float *cMat, float *wfsVec, float *dmVec, int M, int N);

#endif
