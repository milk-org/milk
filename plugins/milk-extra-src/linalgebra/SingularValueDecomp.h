#ifndef LINALGEBRA_COMPSVD_H
#define LINALGEBRA_COMPSVD_H


#define COMPSVD_SKIP_BIGMAT     0x00000001UL
#define COMPSVD_COMP_PSINV      0x00000002UL
#define COMPSVD_COMP_CHECKPSINV 0x00000004UL

errno_t CLIADDCMD_linalgebra__compSVD();

errno_t compute_SVD(
    IMGID imgin,
    IMGID imgU,
    IMGID imgeigenval,
    IMGID imgV,
    float SVlimit,
    int GPUdev,
    uint64_t compSVDmode
);

#endif
