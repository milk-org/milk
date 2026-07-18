/**
 * @file SingularValueDecomp.h
 * @brief Singularvaluedecomp module
 */

#ifndef LINALGEBRA_COMPSVD_H
#define LINALGEBRA_COMPSVD_H

MILK_WEAK errno_t compute_SVD(IMGID    imgin,
                              IMGID   *imgU,
                              IMGID   *imgeigenval,
                              IMGID   *imgV,
                              uint32_t Vdim0,
                              float    SVlimit,
                              uint32_t SVDmaxNBmode,
                              int      GPUdev,
                              uint64_t compSVDmode,
                              char    *SVDunmodesname,
                              char    *SVDvnmodesname) MILK_WEAK_FUNCDEF;

MILK_WEAK errno_t CLIADDCMD_linalgebra__compSVD() {};


#endif
