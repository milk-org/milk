#ifndef LINALGEBRA_COMPSVD_H
#define LINALGEBRA_COMPSVD_H


#define COMPSVD_SKIP_BIGMAT (1 << 0)

errno_t CLIADDCMD_linalgebra__compSVD();

errno_t compute_SVD(
    IMGID imgin,
    IMGID imgU,
    IMGID imgeigenval,
    IMGID imgV,
    int GPUdev,
    uint64_t compSVDmode
);

#endif
