#ifndef LINALGEBRA_COMPSVD_MKM_H
#define LINALGEBRA_COMPSVD_MKM_H



errno_t SVDmkM(
    IMGID    imgU,
    IMGID    imgS,
    IMGID    imgV,
    IMGID    *imgM,
    int      GPUdev
);

errno_t CLIADDCMD_linalgebra__SVDmkM();


#endif
