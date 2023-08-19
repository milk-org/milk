#ifndef LINALGEBRA_COMPSVDU_H
#define LINALGEBRA_COMPSVDU_H



errno_t compute_SVDU(
    IMGID    imgM,
    IMGID    imgV,
    IMGID    imgS,
    IMGID    *imgU,
    IMGID    *imgUS,
    int      GPUdev
);

errno_t CLIADDCMD_linalgebra__compSVDU();


#endif
